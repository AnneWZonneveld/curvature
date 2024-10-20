
"""
Uses python 3.7

"""

import glob
import numpy as np
import pickle
import sys
import shutil
import argparse
import time
import os
from functools import partial
import multiprocessing as mp
import json
import urllib
import torch 
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import embed as shell
import math
from curve_mp import *
from curve_utils import *

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


# ---------------- Input 
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='resnet50', type=str)
parser.add_argument('--pretrained', default=1, type=int)
parser.add_argument('--layer', default='early', type=str)
# parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/data/mp4_h264', type=str)
# parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/data/mp4_h264', type=str)
parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
parser.add_argument('--out_batch', default=1, type=int)
parser.add_argument('--out_batches', default=19, type=int)
parser.add_argument('--in_batches', default=29, type=int)
parser.add_argument('--n_cpus', default=2, type=int)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

start_time = time.time()

#### ----------------

def load_model():
    """
    Load according model and settings.
    """

    if args.model_name in ['i3d_r50']:
        model = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=args.pretrained, force_reload=True)
    elif args.model_name in ['alexnet', 'vgg19', 'resnet50']:
        model = torch.hub.load('pytorch/vision:v0.10.0', args.model_name, pretrained=True)

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

    # Static models
    elif args.model_name == 'alexnet':
        if args.layer == 'early': 
            args.layer = 'features.2'
        elif args.layer == 'mid':
            args.layer = 'features.5'
        elif args.layer == 'late':
            args.layer = 'features.12'
    elif args.model_name == 'vgg19':
        if args.layer == 'early': 
            args.layer = 'features.4'
        elif args.layer == 'mid':
            args.layer = 'features.16'
        elif args.layer == 'late':
            args.layer = 'features.30'
    elif args.model_name == 'resnet50':
        if args.layer == 'early': 
            args.layer = 'layer1.2.relu'
        elif args.layer == 'mid':
            args.layer = 'layer2.3.relu'
        elif args.layer == 'late':
            args.layer = 'layer4.2.relu'
        
    return model


def compute_all_curvature(out_batch, out_batches=19, in_batches=29, n_cpus=1):

    # Load model
    model = load_model()

    # Find video files
    files = sorted(glob.glob(os.path.join(args.data_dir, '*mp4')))
    # files = files[0:32] #test
    out_batch_size = int(len(files)/out_batches)
    print(f'n files {len(files)}')
    print(f'out batches {out_batches}')
    print(f'out batch size {out_batch_size}')

    # Select batch 
    out_batch_files = files[int(out_batch*out_batch_size):int((out_batch+1)*out_batch_size)]

    # Encode batch
    encoded_videos = encode_videos(model_name=args.model_name, files=out_batch_files)
    encoded_videos = np.array(encoded_videos)
    print(f'Encoded all videos')

    results = comp_curv_mp(model=model, model_name=args.model_name, layer=args.layer, encoded_videos=encoded_videos, out_batch=out_batch, in_batches=in_batches, n_cpus=n_cpus)

    return results


# ------------------- MAIN
if __name__ == '__main__':
    # # Setting and checking cache
    # os.environ['TORCH_HOME'] = '/ivi/zfs/s0/original_homes/azonnev/.cache'
    # print("cache log: ")
    # print(os.getenv('TORCH_HOME'))

    # Compute curvature 
    results = compute_all_curvature(out_batch = args.out_batch, out_batches=args.out_batches, n_cpus=args.n_cpus)

    # Reformat results
    results_df = pd.DataFrame()
    all_curvs = []
    all_pixel_curvs = []
    for i in range(len(results)):
        batch_res = results[i]
        all_curvs.extend(batch_res[0])
        all_pixel_curvs.extend(batch_res[1])

    rel_curvs = np.array(all_curvs) - np.array(all_pixel_curvs)
    results_df['curve'] = all_curvs
    results_df['pixel_curve'] = all_pixel_curvs
    results_df['rel_curve'] = rel_curvs

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Done computing curvature out batch {args.out_batch}: {elapsed_time/60} mins')

    # Save results
    res_folder = args.res_dir + f'/features/{args.model_name}/pt_{args.pretrained}/{args.layer}'
    if not os.path.exists(res_folder) == True:
        os.makedirs(res_folder)

    file_path = res_folder + f'/curve_props_df_b{args.out_batch}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(results_df, f)

    print(f'Results stored at {file_path}')


    





