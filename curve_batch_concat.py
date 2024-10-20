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
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import embed as shell
import math

# ---------------- Input 
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='resnet50', type=str)
parser.add_argument('--pretrained', default=1, type=int)
parser.add_argument('--layer', default='early', type=str)
# parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/data/mp4_h264', type=str)
# parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results/features', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# ------------------ Set parameters    
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

# ------------------ Load data
res_dir = args.data_dir + f'/{args.model_name}/pt_{args.pretrained}/{args.layer}/'
files = sorted(glob.glob(os.path.join(res_dir, 'curve_props_df_b*')))

res_df = pd.DataFrame()
for i in range(len(files)):

    file = files[i]
    with open(file, 'rb') as f: 
        data = pickle.load(f)
    
    res_df = pd.concat([res_df, data], axis=0)

res_df = res_df.reset_index()

# Save results
file_path = res_dir + f'/curve_props_df_all.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(res_df, f)