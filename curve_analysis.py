"""
Uses python 3.8

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
parser.add_argument('--model_name', default='slow_r50', type=str)
parser.add_argument('--layer', default='blocks.4.res_blocks.2.activation', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--data_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/', type=str)
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
def load_data(): 
    print('Loading activations')

    # Set to according layer (late layer)
    if args.model_name == 'slow_r50':
        args.layer = 'blocks.4.res_blocks.2.activation'
    elif args.model_name == 'i3d_r50':
        args.layer = 'blocks.5.res_blocks.2.activation'

    activation_dir = args.wd + f'/results/features/{args.model_name}/{args.layer}/'
    activation_file = activation_dir + 'res_df.pkl'

    with open(activation_file, 'rb') as f:
        activation_data = pickle.load(f)
    
    return activation_data

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

def distr_plot(df):

    # Reformat data
    df_melted = pd.melt(df, value_vars=['mean_pixel_curv', 'mean_curv'], var_name='type', value_name='curve')

    # Both distr plot
    fig, ax = plt.subplots(dpi=300) 
    sns.histplot(data=df_melted, x='curve', hue='type', kde=True,ax=ax) 
    ax.set_title(f'{args.model_name} {args.layer} ')
    ax.legend(title='type', labels=['feature space', 'pixel space'])
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/distr_plot.png'
    plt.savefig(img_path)
    plt.clf()

    # Rel plot
    fig, ax = plt.subplots(dpi=300) 
    sns.histplot(data=df, x='rel_curve', kde=True,ax=ax) 
    ax.set_title(f'Rel curve {args.model_name} {args.layer} ')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/rel_distr_plot.png'
    plt.savefig(img_path)
    plt.clf()


def time_series_plot(df):

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

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/time_series_plot.png'
    plt.savefig(img_path)
    plt.clf()



def comp_speed(vid_array):
    '''Compute
    - velocity = difference vectors
    - norm of difference vectors
    for specificied sequence of images'''

    # vid_array = img_array[0, :, :, :, :]
    num_frames = vid_array.shape[1]
    dif_vectors = [vid_array[:, i , :, :].flatten() -  vid_array[:, i + 1 , :, :].flatten().flatten() for i in range(num_frames-1)]
    dif_vectors =  np.vstack(tuple(dif_vectors))
    norms = np.zeros(num_frames - 1)

    for i in range(num_frames - 1):
        norms[i] = np.linalg.norm(dif_vectors[i]) + 1e-8  # Compute the norm (magnitude) of each difference vector
        dif_vectors[i] /= norms[i]  # Normalize the difference vector
    
    return dif_vectors, norms

def comp_curvatures(vid_array):
    """ Compute curvatures based on angles of velocity vectors. """
    num_frames = vid_array.shape[1]
    dif_vectors, norms = comp_speed(vid_array)

    curvs = np.zeros(num_frames - 2)
    angles = np.zeros(num_frames - 2)
    for i in range(num_frames - 2):
        cos = np.clip(np.dot(dif_vectors[i], dif_vectors[i+1]), -1, 1)
        angles[i] = math.acos(cos)
        curvs[i] = math.degrees(math.acos(cos)) #angle in degrees
    
    # mean_curve_angle = sum(angles) * 180 / (len(angles)*math.pi)

    mean_curve = np.mean(curvs)
    mean_norm = np.mean(norms)

    return curvs, mean_curve, norms, mean_norm

def calc_lse(activation):
    """
    Calculate least squared error between actualy temporal trajectory 
    and shortest straight interpolated trajectory.

    Not yet adjusted for new data structure
    """

    activation = np.array(activation)
    num_timepoints = activation.shape[1]
    start_rep = activation[:, 0, :, :]
    end_rep =  activation[:, -1, :, :]

    # Interpolate shortest mean trajectory 
    mean_trajectory = np.array([
    start_rep + (t / (num_timepoints - 1)) * (end_rep - start_rep)
    for t in range(num_timepoints)])
    mean_trajectory = np.stack(mean_trajectory, axis=0).transpose(1, 0, 2, 3)

    # Calculate squared errors relative to the mean trajectory
    lse= np.sum(abs(activation - mean_trajectory) ** 2)

    # Normalize
    lse = lse / num_timepoints

    return lse

def calc_deviation(activation):

    """
    Not yet adjusted for new data structure
    """

    num_frames = activation.shape[1]
    vectors = [(activation[:, i, :, :].flatten() -  activation[:, i + 1, :, :].flatten()) 
    / np.linalg.norm(activation[:, i, :, :].flatten() - activation[:, i + 1, :, :].flatten()) for i in range(num_frames-1)]
    shell()
    mean_vector = np.mean(activation, axis=1).flatten()
    deviations = [mean_vector - vector for vector in vectors]
    deviations = np.stack(deviations, axis=0)
    
    norms = np.linalg.norm(deviations, axis=1)
    # Not done yet

def calc_properties(activation_data):
    """
    Calculate geometric properties for pixel and feature space as
    specified for further analysis.
    """

    mean_curvs = []
    all_curvs = []
    mean_norms = []
    all_norms = []
    mean_pixel_curvs = []
    all_pixel_curvs = []
    mean_pixel_norms = []
    all_pixel_norms = []
    rel_curves = [] 
    rel_norms = []

    for video in range(len(activation_data)):
        if video % 100 == 0:
            print(f'Calculating properties for video {video} ')

        activation = activation_data['activation'].iloc[video].numpy()
        pixels = activation_data['pixels'].iloc[video].numpy()
        
        # Calculate properties for feature space
        curvs, mean_curv, norms, mean_norm = comp_curvatures(activation)
        mean_curvs.append(mean_curv)
        all_curvs.append(curvs)
        mean_norms.append(mean_norm)
        all_norms.append(norms)

        # Calculate properties for pixel space
        pixel_curvs, mean_pixel_curv, pixel_norms, mean_pixel_norm = comp_curvatures(pixels)
        mean_pixel_curvs.append(mean_pixel_curv)
        all_pixel_curvs.append(pixel_curvs)
        mean_pixel_norms.append(mean_pixel_norm)
        all_pixel_norms.append(pixel_norms)
        
        # Compute relative curve / norm
        rel_curve = mean_curv - mean_pixel_curv
        rel_norm = mean_norm -  mean_pixel_norm
        rel_curves.append(rel_curve)
        rel_norms.append(rel_norm)

    activation_data['all_curvs'] = all_curvs
    activation_data['mean_curv'] = mean_curvs
    activation_data['all_norms'] = all_norms
    activation_data['mean_norm'] = mean_norms
    activation_data['all_pixel_curvs'] = all_pixel_curvs
    activation_data['mean_pixel_curv'] = mean_pixel_curvs
    activation_data['all_pixel_norms'] = all_pixel_norms
    activation_data['mean_pixel_norm'] = mean_pixel_norms
    activation_data['rel_curve'] = rel_curves
    activation_data['rel_norm'] = rel_norms
    
    return activation_data


def static_control(activation_data): 
    """
    Not yet adjusted for new data structure
    """

    curves = []
    norms = []
    lses = []
    pixel_curves = []
    pixel_norms = []
    pixel_lses = []

    for video in range(len(activation_data)):
        first_t = activation_data['activation'].iloc[video][:, 0, :, :].unsqueeze(1)
        first_t_pixel = activation_data['pixels'].iloc[video][:, 0, :, :].unsqueeze(1)

        static_activation = first_t.repeat(1, 8, 1, 1)
        static_pixel = first_t_pixel.repeat(1, 8, 1, 1)

        curve = calc_curve(static_activation)
        norm = calc_norm(static_activation)
        lse = calc_lse(static_activation)

        pixel_curve =  calc_curve(static_pixel)
        pixel_norm =  calc_norm(static_pixel)
        pixel_lse = calc_lse(static_pixel)
    
        curves.append(curve)
        norms.append(norm)
        lses.append(lse)
        pixel_curves.append(pixel_curve)
        pixel_norms.append(pixel_norm)
        pixel_lses.append(pixel_lse)

    shell()
    if sum(curves) + sum(norms) + sum(lses) == 0:
        print("Static control: no changes in representational feature space model")
    if sum(pixel_curves) + sum(pixel_norms) + sum(pixel_lses) == 0: 
        print("Static control: no changes in pixel space")

# ------------------ MAIN
activation_data = load_data()
activation_data = activation_data.reset_index()   

# static_control(activation_data)
curve_data = calc_properties(activation_data=activation_data)

# Plot relationships variables of interest 
scatter_plot(df=curve_data, x='curve', y='norm')
scatter_plot(df=curve_data, x='lse', y='norm')
scatter_plot(df=curve_data, x='curve', y='lse')
scatter_plot(df=curve_data, x='pixel_curve', y='pixel_norm')
scatter_plot(df=curve_data, x='pixel_lse', y='pixel_norm')
scatter_plot(df=curve_data, x='pixel_curve', y='pixel_lse')
scatter_plot(df=curve_data, x='rel_curve', y='norm')
scatter_plot(df=curve_data, x='rel_lse', y='norm')


# Qualitative inspections
# scatter_control(curve_data=curve_data, n_samples=1000, n_control_videos=25, x1='curve', x2='norm')
# cluster_control(curve_data=curve_data, x1='curve', x2='norm')


