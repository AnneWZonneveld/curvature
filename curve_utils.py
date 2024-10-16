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


def transform_clips(video, model_name):
    """
    Compose video data transforms --> specific to model

    Parameters
    ---------
    video: tensor
    model_name: str
    
    """

    # Define parameters based on model
    if model_name in  ['slow_r50', 'i3d_r50']:
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30

    # Compose transformation
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(sampling_rate),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size=(crop_size, crop_size))
            ]
        ),
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
            activations[name] = output.detach()
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


def comp_curv(batch, model, model_name, layer, files, batch_size):

    print(f'Starting b {batch}')
    start_time = time.time()

    # Encode batch of videos
    batch_files = files[int(batch*batch_size):int((batch+1)*batch_size)]
    encoded_videos = encode_videos(model_name=model_name, files=batch_files)

    # Get activations
    activations = layer_activations(model=model, layer=layer, inputs=encoded_videos)

    activation_list = [activations[layer][i] for i in range(activations[layer].shape[0])]
    pixel_list = [encoded_videos[i] for i in range(encoded_videos.shape[0])]

    # Calculate curvature
    curves = []
    pixel_curves = []
    for video in tqdm(range(len(batch_files))):

        activations = activation_list[video].numpy()
        pixels = pixel_list[video].numpy()
        
        curv = comp_mean_curv(activations)
        pixel_curv = comp_mean_curv(pixels)
        curves.append(curv)
        pixel_curves.append(pixel_curv)
    
    end_time = time.time()
    print(f'b {batch} done: {(end_time - start_time)/60} mins')
    
    return (curves, pixel_curves)

