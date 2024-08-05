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

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='slow_r50', type=str)
parser.add_argument('--data_split', default='train', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Load model
# model = torch.hub.load('facebookresearch/pytorchvideo', args.model, pretrained=True)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

# Load video
video = EncodedVideo.from_path('/Users/annewzonneveld/Documents/phd/projects/curvature/0001.mp4')
#video = EncodedVideo.from_path('/home/azonnev/data/boldmoments/stimulus_set/mp4_h264/0001.mp4')

# Compose video data transforms --> specific to model
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30

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

# Get clip
clip_start_sec = 0.0 
clip_duration = 3.0 
video_data = video.get_clip(start_sec=clip_start_sec, end_sec=clip_start_sec + clip_duration) 
video_data = transform(video_data)
input = video_data['video'] #torch.Size([3, 8, 256, 256])

# Generate top 5 predictions
preds = model(input[None, ...]) #input should be 5 dimensional (# of videos, RGB, # of frames, #height, #width)
preds = torch.nn.functional.softmax(preds, dim=1) 
pred_class_ids = preds.topk(k=5).indices[0]

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

pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_class_ids]

# Get intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks to the layers you are interested in
layer_names = ["blocks.4.res_blocks.2.activation"]
for name in layer_names:
    layer = dict([*model.named_modules()])[name]
    layer.register_forward_hook(get_activation(name))

# Pass the video through the model
with torch.no_grad():
    output = model(input[None, ...])


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

