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
import seaborn as sns
import matplotlib.pyplot as plt 

# ------------------- Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='slow_r50', type=str)
parser.add_argument('--layer', default='blocks.4.res_blocks.2.activation', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--wd', default='/home/azonnev/analyses/curvature', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# ------------------- Load in data
print('Loading curvature results')
curve_dir = args.wd + f'/results/features/{args.model_name}/{args.layer}/'
curve_file = curve_dir + 'res_df.pkl'

with open(curve_file, 'rb') as f:
    curve_data = pickle.load(f)


# ------------------- Plot
sns.set_style('white')
sns.set_style("ticks")
sns.set_context('paper', 
                rc={'font.size': 14, 
                    'xtick.labelsize': 10, 
                    'ytick.labelsize':10, 
                    'axes.titlesize' : 13,
                    'figure.titleweight': 'bold', 
                    'axes.labelsize': 13, 
                    'legend.fontsize': 8, 
                    'font.family': 'Arial',
                    'axes.spines.right' : False,
                    'axes.spines.top' : False})

fig, ax = plt.subplots(dpi=300)
sns.scatterplot(data=curve_data, x='curve', y="norm")
ax.set_title(f'{args.model_name} {args.layer}')
ax.set_xlabel('curve')
ax.set_ylabel(f'norm')
sns.despine(offset= 10, top=True, right=True)
fig.tight_layout()

output_dir = args.wd + f'/results/figures/'
if not os.path.exists(output_dir) == True:
    os.makedirs(output_dir)

img_path = output_dir + f'scatter_{args.model_name}_{args.layer}.png'
plt.savefig(img_path)
plt.clf()