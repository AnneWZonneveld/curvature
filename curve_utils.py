import os
import torch 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython import embed as shell
from tqdm import tqdm
import time
import math
from multiprocessing import current_process, cpu_count, shared_memory

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from iopath.common.file_io import PathManager
pathmgr = PathManager()
pathmgr.set_logging(False) 


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def transform_clips(video, model_name):
    """
    Compose video data transforms --> specific to model

    Parameters
    ---------
    video: tensor
    model_name: str
    
    """

    # Define parameters based on model
    if model_name in  ['slow_r50','i3d_r50', 'c2d_r50','resnet50']:
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30
    
    elif model_name == 'slowfast_r50':
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32 # different than documentation
        sampling_rate = 2
        frames_per_second = 30
    
    elif model_name == 'r2plus1d_r50':
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 16
        sampling_rate = 4
        frames_per_second = 30
    
    elif model_name == 'csn_r101':
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        sampling_rate = 2
        frames_per_second = 30
    
    elif model_name == 'x3d_s':
        side_size = 182
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 182
        num_frames = 13
        sampling_rate = 6
        frames_per_second = 30
    
    elif model_name == 'mvit_base_16x4':
        side_size = 224
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 224
        num_frames = 16
        sampling_rate = 4
        frames_per_second = 30



    transforms_list = [
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size=(crop_size, crop_size)),
    ]

    if model_name == 'slowfast_r50':
        transforms_list.append(PackPathway())

    # Compose transformation
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(transforms_list),
    )

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    # Get clip and transform
    start_sec = 0.0 
    end_sec = start_sec + clip_duration
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec) 
    video_data= transform(video_data)

    return video_data

def encode_videos(files, model_name):
    """
    Encodes video files to tenstors 

    Parameters
    ----------

    files: list of file strings
    model: string of model name
    
    returns 5 dimensional tensor (# of videos, RGB, # of frames, #height, #width)

    """
    if model_name == 'slowfast_r50':
        slow_inputs = []
        fast_inputs = []
        for file in files:
            video =  EncodedVideo.from_path(file)
            video_data = transform_clips(video=video, model_name=model_name)
            del video
            slow_inputs.append(video_data['video'][0])
            fast_inputs.append(video_data['video'][1])
        
        fast_inputs = torch.stack(fast_inputs)
        slow_inputs = torch.stack(slow_inputs)

        inputs = np.array((slow_inputs, fast_inputs), dtype='object')
    else:
        inputs = []
        for file in files:
            video =  EncodedVideo.from_path(file)
            video_data = transform_clips(video=video, model_name=model_name)
            del video
            inputs.append(video_data['video'])
        
        inputs = torch.stack(inputs)

    return inputs


def layer_activations(model, layer, inputs):
    """
    Get activations from specified layers

    Parameters
    ----------

    layers: list of strings with layer names
    inputs:  5 dimensional tensor of encoded videos (# of videos, RGB, # of frames, #height, #width)

    returns convoluted tensor (# of videos, # of frames, #height, #width)

    """

    # Get intermediate activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
            try:
                output = output.detach()
            except:
                pass
        return hook

    # Register hook
    model_layer = dict([*model.named_modules()])[layer]
    model_layer.register_forward_hook(get_activation(layer))

    # Pass the video through the model
    with torch.no_grad():
        output = model(inputs)
    
    return activations


def comp_speed(vid_array):

    """
    Compute velocity (=difference vectors) and norm for specified sequence of images

    Parameters
    ----------

    vid_array: 4 dimensional tensor of image sequence (# of frames, # channels, height, width)

    returns
    - diff_vectors: list of length vid_array.shape[0] - 1
    - norms: list of length vid_array.shape[0] - 1

    """

    num_frames = vid_array.shape[1]
    dif_vectors = [vid_array[:, i , :, :].flatten() -  vid_array[:, i+1, : , :].flatten().flatten() for i in range(num_frames-1)]
    dif_vectors =  np.vstack(tuple(dif_vectors))
    norms = np.zeros(num_frames - 1) 

    for i in range(num_frames - 1):
        norms[i] = np.linalg.norm(dif_vectors[i]) + 1e-8  # Compute the norm (magnitude) of each difference vector
        dif_vectors[i] /= norms[i]  # Normalize the difference vector
    
    return dif_vectors, norms

def comp_mean_curv(vid_array):
    """ Compute curvatures based on angles of velocity vectors. """

    num_frames = vid_array.shape[1]
    dif_vectors, norms = comp_speed(vid_array)

    curvs = np.zeros(num_frames - 2)
    angles = np.zeros(num_frames - 2)
    for i in range(num_frames - 2):
        cos = np.clip(np.dot(dif_vectors[i], dif_vectors[i+1]), -1, 1)
        angles[i] = math.acos(cos)
        curvs[i] = math.degrees(math.acos(cos)) #angle in degrees
    
    mean_curve = np.mean(curvs)
    
    return mean_curve


def comp_curves(batch, model, model_name, layer, batches, data_shape, dtype, shm_name):

    print(f'starting batch {batch}')
    start_time = time.time()

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    serialized_data_copy = bytes(np.ndarray((data_shape,), dtype='B', buffer=existing_shm.buf))
    encoded_videos = pickle.loads(serialized_data_copy)

    # Select videos current batch
    if model_name == 'slowfast_r50':
        batch_size = int(encoded_videos[0].shape[0]/batches)
        slow_batch = torch.tensor(encoded_videos[0][int(batch*batch_size):int((batch+1)*batch_size), :, :, :, :])
        fast_batch = torch.tensor(encoded_videos[1][int(batch*batch_size):int((batch+1)*batch_size), :, :, :, :])
        batch_videos = [slow_batch, fast_batch]

        # Get pixel values
        pixel_list = [slow_batch[i] for i in range(slow_batch.shape[0])]
    else:
        batch_size = int(encoded_videos.shape[0]/batches)
        batch_videos = torch.tensor(encoded_videos[int(batch*batch_size):int((batch+1)*batch_size), :, :, :, :])

        # Get pixel values
        pixel_list = [batch_videos[i] for i in range(batch_videos.shape[0])]

    # Get activations
    if model_name in ['i3d_r50', 'c2d_r50', 'slow_r50', 'slowfast_r50', 'r2plus1d_r50', 'csn_r101', 'x3d_s', 'mvit_base_16x4']: # video models
        activations = layer_activations(model=model, layer=layer, inputs=batch_videos)
        activation_list = [activations[layer][i] for i in range(activations[layer].shape[0])]
    
    elif model_name in ['resnet50']: # static models
        batch_activations = []
        for video in range(batch_videos.shape[0]):
            video_activation = []
            for frame in range(batch_videos.shape[2]):
                frame_activation = layer_activations(model=model, layer=layer, inputs=batch_videos[video, :, frame, :, :].unsqueeze(0))
                video_activation.append(frame_activation[layer])
            video_activation = torch.cat(video_activation, dim=0)
            batch_activations.append(video_activation)
        batch_activations = torch.stack(batch_activations)
        activation_list = [batch_activations[i] for i in range(batch_activations.shape[0])]
    
    # Calculate curvature
    curves = []
    pixel_curves = []
    for video in tqdm(range(len(batch_videos))):

        activations = activation_list[video].numpy()
        pixels = pixel_list[video]
        
        curv = comp_mean_curv(activations)
        pixel_curv = comp_mean_curv(pixels)
        curves.append(curv)
        pixel_curves.append(pixel_curv)
    
    end_time = time.time()
    print(f'b {batch} done: {(end_time - start_time)/60} mins')
    
    return (curves, pixel_curves)