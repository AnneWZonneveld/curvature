"""
Video-feature extracting

uses python 3.7

"""

import glob
import numpy as np
import pickle
import sys
import shutil
import argparse
import time
import os
import json
import urllib
import torch 
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

# ------------------- Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='slow_r50', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

#### ----------------

def transform_clips(video, model_name='slow_r50'):
    """
    Compose video data transforms --> specific to model

    Parameters
    ---------
    video: tensor
    model_name: str
    
    """

    # Define parameters based on model
    if model == 'slow_r50':
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
    clip_start_sec = 0.0 
    clip_duration = 3.0 
    video_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration) 
    video_data= transform(video_data)

    return video_data

def encode_videos(files, model='slow_r50'):
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
        video_data = transform_clips(video=video, model=model)
        inputs.append(video_data['video'])
    inputs = torch.stack(inputs)

    return inputs

def top5_preds(inputs):
    """Generate the top-5 predictions based on output 

    Parameters
    ----------

    inputs: 5 dimensional tensor (# of videos, RGB, # of frames, #height, #width)
    
    """

    preds = model(inputs) #i
    preds = torch.nn.functional.softmax(preds, dim=1) 
    pred_class_ids = preds.topk(k=5).indices

    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    try: urllib.URLopener().retrieve(json_url, json_filename)
    except: urllib.request.urlretrieve(json_url, json_filename)
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    for j in range(len(pred_class_ids)):
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_class_ids[j]]
        print(f"video {j}: {pred_class_names}")


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def layers_activations(model, layers, inputs):
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

    # Register hooks to the layers you are interested in
    for name in layers:
        layer = dict([*model.named_modules()])[name]
        layer.register_forward_hook(get_activation(name))

    # Pass the video through the model
    with torch.no_grad():
        output = model(inputs)

    
    return activations

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calc_curve(representations):
    num_frames = representations.shape[2]
    vectors = [representations[:, :, i, :, :].flatten() -  representations[:, :, i + 1, :, :].flatten() for i in range(num_frames-1)]
    cos_sims = [cosine_similarity(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)]
    curve = np.mean(cos_sims) if cos_sims else 0
    return curve

def calc_norm(representations):
    num_frames = representations.shape[2]
    vectors = [representations[:, :, i, :, :].flatten() -  representations[:, :, i + 1, :, :].flatten() for i in range(num_frames-1)]
    norms = [np.linalg.norm(vectors[i]) for i in range(len(vectors))]
    norm = np.mean(norms)
    return norm


# ------------------- MAIN
# Load model
# model = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=True)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

# Load videos
# data_dir = args.data_dir
data_dir = '/Users/annewzonneveld/Documents/phd/projects/curvature/test_data/'
files = sorted(glob.glob(os.path.join(data_dir, '*mp4')))
encoded_videos = encode_videos(model='slow_r50', files=files)
# encoded_videos = encode_videos(model=args.model_name, files=files)

# Get activations
layers = ["blocks.4.res_blocks.2.activation"]
activations = layers_activations(model=model, layers=layers, inputs=encoded_videos)