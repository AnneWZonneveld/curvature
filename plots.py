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
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.utils import resample
from IPython import embed as shell
from scipy.stats import pearsonr, spearmanr
import math
import tensorflow as tf
import tensorflow_hub as hub
from datasets import Dataset
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool, ColumnarDataSource, LinearColorMapper, ColumnDataSource, CategoricalColorMapper
from bokeh.transform import factor_cmap
from bokeh.palettes import Turbo256


# ------------------- Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='i3d_r50', type=str)
# parser.add_argument('--wd', default='/home/azonnev/analyses/curvature', type=str)
parser.add_argument('--wd', default='/Users/annewzonneveld/Documents/phd/projects/curvature', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

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

# ----------------------------
def load_data(layer, pretrained=1):
    """ 
    Loads df of curvature properties
    
    Parameters
    ----------

    layer: str
    pretrained: bool

    returns df with curvature properties
    """

    print(f'Loading curvature data for {layer} layer pt {pretrained}')

    # Set to according layer  
    if args.model_name == 'slow_r50':
        if layer == 'late': 
            layer = 'blocks.4.res_blocks.2.activation'
    elif args.model_name == 'i3d_r50':
        if layer == 'early': 
            layer = 'blocks.1.res_blocks.2.activation'''
        elif layer == 'mid':
            layer = 'blocks.3.res_blocks.3.activation'
        elif layer == 'late':
            layer = 'blocks.5.res_blocks.2.activation'

    # Static model
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

    curv_dir = args.wd + f'/results/features/{args.model_name}/pt_{pretrained}/{layer}/'
    curv_file = curv_dir + 'curve_props_df.pkl'

    with open(curv_file, 'rb') as f:
        curv_data = pickle.load(f)
    
    return curv_data


def distr_plot(layer, pretrained):
    """ 
    Plots 2 figures:
    - pixel vs feature curvature distribution 
    - relative curvature distribution
    
    Parameters
    ----------

    layer: str
    pretrained: bool

    """

    # Load data
    print('Loading data')
    df = load_data(layer=layer, pretrained=pretrained)

    # Reformat data
    df_melted = pd.melt(df, value_vars=['mean_pixel_curv', 'mean_curv'], var_name='type', value_name='curve')

    # Set to according layer name 
    if args.model_name == 'slow_r50':
        if layer == 'late': 
            layer = 'blocks.4.res_blocks.2.activation'
    elif args.model_name == 'i3d_r50':
        if layer == 'early': 
            layer = 'blocks.1.res_blocks.2.activation'''
        elif layer == 'mid':
            layer = 'blocks.3.res_blocks.3.activation'
        elif layer == 'late':
            layer = 'blocks.5.res_blocks.2.activation'

    print('Distribution plots')
    # Feature + pixel distr plot
    fig, ax = plt.subplots(dpi=300) 
    sns.histplot(data=df_melted, x='curve', hue='type', kde=True,ax=ax) 
    ax.set_title(f'{args.model_name} {layer} pt {pretrained} ')
    # ax.legend(title='type', labels=['feature space', 'pixel space'])
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/pt_{pretrained}/{layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/distr_plot.png'
    plt.savefig(img_path)
    plt.clf()

    # Relative curve plot
    fig, ax = plt.subplots(dpi=300) 
    sns.histplot(data=df, x='rel_curve', kde=True,ax=ax) 
    ax.set_title(f'Rel curve {args.model_name} {layer} pt {pretrained}')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    img_path = output_dir + f'/rel_distr_plot.png'
    plt.savefig(img_path)
    plt.clf()


def pt0_vs_pt1_rel_distr_plot(layer):
    """ 
    Plots the relative curvature for pretrained vs untrained activations.

    Parameters
    ----------

    layer: str

    """

    # Load data
    print('Loading data')
    pt1_df = load_data(layer=layer, pretrained=1)
    pt0_df = load_data(layer=layer, pretrained=0)

    if args.model_name == 'slow_r50':
        if layer == 'late': 
            layer = 'blocks.4.res_blocks.2.activation'
    elif args.model_name == 'i3d_r50':
        if layer == 'early': 
            layer = 'blocks.1.res_blocks.2.activation'''
        elif layer == 'mid':
            layer = 'blocks.3.res_blocks.3.activation'
        elif layer == 'late':
            layer = 'blocks.5.res_blocks.2.activation'


    # Reformat data
    df = pd.DataFrame()
    df['rel_curve'] = pt1_df['rel_curve'].to_list() + pt0_df['rel_curve'].to_list()
    df['type'] = ['pretrained'] * len(pt1_df) + ['untrained'] *  len(pt1_df)

    # Compare trainen vs untrained plot
    fig, ax = plt.subplots(dpi=300) 
    sns.histplot(data=df, x='rel_curve', hue='type', kde=True, ax=ax) 
    ax.set_title(f'{args.model_name} {layer}')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/pt0_vs_pt1/{layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/comp_distr_plot.png'
    plt.savefig(img_path)
    plt.clf()


def time_series_plot(df):
    """
    Not done.
    """

    # Reformat data
    n_videos = len(df)
    n_curvs = df['all_curvs'].iloc[0].shape[0]
    vid_id = np.repeat(np.array([*range(len(df))]) + 1, n_curvs)
    curv_id = [*range(n_curvs)] * len(df)
    curves = [item for sublist in df['all_curvs'] for item in sublist]
    
    rf_df = pd.DataFrame()
    rf_df['vid_id'] = vid_id
    rf_df['curv_id'] = curv_id
    rf_df['curve'] = curves

    # Plot
    fig, ax = plt.subplots(dpi=500)
    sns.boxplot(data=rf_df, x='curv_id', y='curve', ax=ax, width = 0.3)
    # sns.stripplot(data=rf_df, x='curv_id', y='curve', hue='vid_id', ax=ax, 
    #             dodge=True, jitter=True, palette='husl', alpha=0.7, size=5)

    # Plot transparent lines connecting individual measurements across timepoints
    for vid_id in rf_df['vid_id'].unique():
        individual_data = rf_df[rf_df['vid_id'] == vid_id]
        plt.plot(individual_data['curv_id'], individual_data['curve'], 
                color='gray', alpha=0.2, linewidth=1, zorder=1)

    # Customize labels and title
    ax.set_title('Curve over time')
    ax.set_xlabel('time')
    ax.set_ylabel('curve')

    # # Optional: remove legend for 'measurement'
    # ax.legend_.remove()
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/pt_{args.pretrained}/{args.layer}/'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/time_series_plot.png'
    plt.savefig(img_path)
    plt.clf()


def scatter_plot(df, x, y):

    # Test correlation
    res = spearmanr(df[x], df[y])
    stat = round(res[0], 4)
    p_val = round(res[1], 4)

    # Plot
    fig, ax = plt.subplots(dpi=300)
    sns.scatterplot(data=df, x=x, y=y)
    ax.set_title(f'{args.model_name} {args.layer}''\n'f'spearman: {stat}, p: {p_val}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    sns.despine(offset= 10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/scatter_{x}_{y}.png'
    plt.savefig(img_path)
    plt.clf()


# ---------------- MAIN

# Plot distributions
distr_plot(layer = 'late', pretrained=1)

# Plot untrained vs pretrained relative curve plots
pt0_vs_pt1_rel_distr_plot(layer = 'late')

# Plot relationships variables of interest 
# scatter_plot(df=curve_data, x='curve', y='norm')
# scatter_plot(df=curve_data, x='lse', y='norm')
# scatter_plot(df=curve_data, x='curve', y='lse')
# scatter_plot(df=curve_data, x='pixel_curve', y='pixel_norm')
# scatter_plot(df=curve_data, x='pixel_lse', y='pixel_norm')
# scatter_plot(df=curve_data, x='pixel_curve', y='pixel_lse')
# scatter_plot(df=curve_data, x='rel_curve', y='norm')
# scatter_plot(df=curve_data, x='rel_lse', y='norm')