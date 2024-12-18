"""
Video-feature extracting

uses python 3.7 --> use py37 env

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
import pandas as pd
from IPython import embed as shell
from tqdm import tqdm
import torch 
import psutil
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
parser.add_argument('--model_name', default='i3d_r50', type=str)
parser.add_argument('--pretrained', default=1, type=int)
parser.add_argument('--layer', default='late', type=str)
parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/data/mp4_h264', type=str)
parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

start_time = time.time()

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

def encode_videos(files, model_name='slow_r50'):
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

def top5_preds(inputs):
    """Generate the top-5 predictions based on output 

    Parameters
    ----------

    inputs: 5 dimensional tensor (# of videos, RGB, # of frames, #height, #width)
    
    """

    preds = model(inputs) 
    preds = torch.nn.functional.softmax(preds, dim=1) 
    pred_class_ids = preds.topk(k=5).indices

    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    try: 
        urllib.URLopener().retrieve(json_url, json_filename)
    except: 
        urllib.request.urlretrieve(json_url, json_filename)
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    for j in range(len(pred_class_ids)):
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_class_ids[j]]
        print(f"video {j}: {pred_class_names}")


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

def perform_pca(activations, n_components=20): #bring back to 3 dimensions, similar to RGB?

    for i in range(activations.shape[0]):
        pca_frames = []
        for j in range(activations.shape[2]):
            frame =  activations[i, :, j, :, :]  # [n channels,  height,  width]
            frame_reshaped = frame.reshape(-1, frame.size(0)) # [height x width, n channels]
            
            pca = PCA(n_components=n_components)
            frame_pca = pca.fit_transform(frame_reshaped.numpy()) # [n channels, n_componenets]
            frame_pca = torch.tensor(frame_pca)

            pca_frames.append(frame_pca)









# ------------------- MAIN
# Setting and checking cache
os.environ['TORCH_HOME'] = '/ivi/zfs/s0/original_homes/azonnev/.cache'
print("cache log: ")
print(os.getenv('TORCH_HOME'))

# Load model
model = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=args.pretrained, force_reload=True)
model.eval()
print(f'Model {args.model_name} loaded succesfully')

# Set to according layer 
if args.model_name == 'slow_r50':
    if args.layer == 'early': 
        args.layer = '...'
    elif args.layer == 'mid':
        args.layer = '...'
    elif args.layer == 'late':
        args.layer = 'blocks.4.res_blocks.2.activation'
elif args.model_name == 'i3d_r50':
    if args.layer == 'early': 
        args.layer = 'blocks.1.res_blocks.2.activation'''
    elif args.layer == 'mid':
        args.layer = 'blocks.3.res_blocks.3.activation'
    elif args.layer == 'late':
        args.layer = 'blocks.5.res_blocks.2.activation'

# Load videos
files = sorted(glob.glob(os.path.join(args.data_dir, '*mp4')))
batches = 551
batch_size = int(len(files)/batches)

results_df = pd.DataFrame()

for batch in tqdm(range(batches)):

    # Encode batch of videos
    batch_files = files[int(batch*batch_size):int((batch+1)*batch_size)]
    encoded_videos = encode_videos(model_name=args.model_name, files=batch_files)

    # Get activations
    activations = layer_activations(model=model, layer=args.layer, inputs=encoded_videos)

    extract_df = pd.DataFrame()
    extract_df['activation'] = [activations[args.layer][i] for i in range(activations[args.layer].shape[0])]
    extract_df['pixels'] = [encoded_videos[i] for i in range(encoded_videos.shape[0])]

    # Perform PCA / dimensionality reduction

    results_df = pd.concat([results_df, extract_df], axis=0)


end_time = time.time()
elapsed_time = end_time - start_time

print(f'Done extracting features in {elapsed_time/60} minutes')

# Save results
res_folder = args.res_dir + f'/features/{args.model_name}/pt_{args.pretrained}/{args.layer}'
if not os.path.exists(res_folder) == True:
    os.makedirs(res_folder)

file_path = res_folder + '/res_df.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(results_df, f)

print(f'Results stored at {file_path}')

