"""
Video-feature extracting
"""

import glob
import numpy as np
import pickle
import sys
import shutil
import argparse
import time
import os
import torch 
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from pytorchvideo.data.encoded_video import EncodedVideo
import torchvision.transforms.functional.to_tensor as F_t
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='slowfast_r50', type=str)
parser.add_argument('--data_split', default='train', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Load model
model = torch.hub.load('facebookresearch/pytorchvideo', args.model, pretrained=True)

# Load video
video = EncodedVideo.from_path('/home/azonnev/data/boldmoments/stimulus_set/mp4_h264/0001.mp4')

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
print(f"video_data {video_data.shape}")

# Generate top 5 predictions
preds = torch.nn.functional.softmax(preds)
pred_class_ids = preds.topk(k=5).indices
print(f"classes: {pred_class_ids}")