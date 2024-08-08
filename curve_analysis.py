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

# ------------------- Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='slow_r50', type=str)
parser.add_argument('--layer', default='blocks.4.res_blocks.2.activation', type=str)
parser.add_argument('--data_split', default='train', type=str)
parser.add_argument('--data_dir', type=str)
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
    print('Loading curvature results')
    curve_dir = args.wd + f'/results/features/{args.model_name}/{args.layer}/'
    curve_file = curve_dir + 'res_df.pkl'

    with open(curve_file, 'rb') as f:
        curve_data = pickle.load(f)
    
    return curve_data

def scatter_plot(df, data_type='stand'):
    if data_type == 'stand':
        x ='curve'
        y ='norm'
    elif data_type =='pixel':
        x = 'pixel_curve'
        y ='pixel_norm'
    elif data_type == 'rel':
        x = 'rel_curve'
        y = 'rel_norm'

    fig, ax = plt.subplots(dpi=300)
    sns.scatterplot(data=df, x=x, y=y)
    ax.set_title(f'{args.model_name} {args.layer}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    sns.despine(offset= 10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/scatter_{data_type}.png'
    plt.savefig(img_path)
    plt.clf()


def baseline_control(curve_data, n_samples=1000, n_control_videos=10, x1='curvature', x2='norm'):
    """
    Based on RNC algorithm Ale

    curve_data: df
        df with all relevant curvature measurement
    n_samples: int
        n of samples to base null distribution on 
    n_control_videos:
        n of control images to select
    x1: str
        measurement 1 to use
    x2: stre
        measurement 2 to use
    """

    n_videos = len(curve_data)
    measurements = [x1, x2]

    # Null distribution videos 2-D array of shape:
    # (N Null distribution samples × Y Videos)
    null_dist_vids = np.zeros((n_samples, n_control_videos), dtype=np.int32)
    # Null distribution scores 2-D arrays of shape:
    # (N Null distribution samples × n of measurements)
    null_dist_select = np.zeros((n_samples, len(measurements)), dtype=np.float32)

    # Create the null distribution
    for i in tqdm(range(n_samples), desc='Null distribution samples'):
        sample = resample(np.arange(n_videos), replace=False, n_samples=n_control_videos)
        sample.sort()
        null_dist_vids[i] = sample
        for r in range(len(measurements)):
            measurement = measurements[r]
            null_dist_select[i,r] = np.mean(curve_data[measurement].iloc[sample])
        del sample
    
    # Select the video sample closest to the selection split null distribution mean,
    # and use its score as the univariate RNC baseline
    baseline_score_select = np.zeros((len(measurements)))
    for r in range(len(measurements)):
        null_dist_mean = np.mean(null_dist_select[:,r])
        idx = np.argsort(abs(null_dist_select[:,r] - null_dist_mean))[0]
        baseline_score_select[r] = null_dist_select[idx,r]
    
    return baseline_score_select

def control_videos(curve_data, n_samples=1000, n_control_videos=10, x1='curvature', x2='norm', margin=0.05):
    """
    Based on RNC algorithm Ale

    curve_data: df
        df with all relevant curvature measurement
    n_samples: int
        n of samples to base null distribution on 
    n_control_videos:
        n of control images to select
    x1: str
        measurement 1 to use
    x2: stre
        measurement 2 to use
    """

    baseline_score_select = baseline_control(curve_data=curve_data, n_samples=n_samples, n_control_videos=n_control_videos, x1=x1, x2=x2)

    # Select the top N videos that align the two measurements: 
    # i.e., that lead both high values or both low values
    measure_sum = curve_data[x1] + curve_data[x2]

    # High / high corner
    high_1_high_2 = np.argsort(measure_sum)[::-1]

    # Ignore images conditions with univariate responses below the baseline scores
    # (plus a margin)
    idx_bad_x1 = curve_data[x1][high_1_high_2] < (baseline_score_select[0] + margin)
    idx_bad_x2 = curve_data[x2][high_1_high_2] < (baseline_score_select[1] + margin)
    idx_bad = np.where(idx_bad_x1 + idx_bad_x2)[0]
    high_1_high_2 = np.delete(np.array(high_1_high_2), idx_bad, axis=0)[:n_control_videos]

    # Low / low corner
    low_1_low_2 = np.argsort(measure_sum)

    # Ignore images conditions with univariate responses below the baseline scores
    # (minus a margin)
    idx_bad_x1 = curve_data[x1][high_1_high_2] > (baseline_score_select[0] + margin)
    idx_bad_x2 = curve_data[x2][high_1_high_2] < (baseline_score_select[1] + margin)
    idx_bad = np.where(idx_bad_x1 + idx_bad_x2)[0]
    low_1_low_2 = np.delete(np.array(low_1_low_2), idx_bad)[:n_control_videos]

    # Select the top N videos that differentiate the two measurements:
    # i.e., that lead to one measurement with high and the other one with low values
    measure_diff = curve_data[x1] - curve_data[x2]

    # High / low corner
    high_1_low_2 = np.argsort(measure_diff)[::-1]
    # Ignore images conditions with responses below (x1) or above
    # (x2) the baseline scores (plus/minus a margin)
    idx_bad_x1 = curve_data[x1][high_1_low_2] < (baseline_score_select[0] + margin)
    idx_bad_x2 = curve_data[x2][high_1_low_2] > (baseline_score_select[1] - margin)
    idx_bad = np.where(idx_bad_x1 + idx_bad_x2)[0]
    high_1_low_2 = np.delete(np.array(high_1_low_2), idx_bad)[:n_control_videos]

    # Low / high corner
    low_1_high_2 = np.argsort(measure_diff)
    # Ignore images conditions with responses above (x1) or below
    # (x2) the baseline scores (plus/minus a margin)
    idx_bad_x1 = curve_data[x1][low_1_high_2] > (baseline_score_select[0] - margin)
    idx_bad_x2 = curve_data[x2][low_1_high_2] < (baseline_score_select[1] + margin)
    idx_bad = np.where(idx_bad_x1 + idx_bad_x2)[0]
    low_1_high_2 = np.delete(np.array(low_1_high_2), idx_bad)[:n_control_videos]

    final_idx = {}
    final_idx['h1-h2'] = high_1_high_2
    final_idx['l1-l2'] = low_1_low_2
    final_idx['h1-l2'] = high_1_low_2
    final_idx['l1-h2'] = low_1_high_2

    return final_idx, baseline_score_select

def scatter_control(curve_data, n_samples=1000, n_control_videos=10, x1='curvature', x2='norm'):

    curve_data = curve_data.reset_index()    
    outlier_idx, baselines = control_videos(curve_data=curve_data, n_samples=n_samples, n_control_videos=n_control_videos, x1=x1, x2=x1)
    high_1_high_2 = curve_data.iloc[outlier_idx['h1-h2']]
    low_1_low_2 = curve_data.iloc[outlier_idx['l1-l2']]
    high_1_low_2 = curve_data.iloc[outlier_idx['h1-l2']] #empty
    low_1_high_2 = curve_data.iloc[outlier_idx['l1-h2']] #empty

    # Plot parameters
    fontsize = 15
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = fontsize
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['lines.markersize'] = 3
    colors = [(4/255, 178/255, 153/255), (130/255, 201/255, 240/255),
        (217/255, 214/255, 111/255), (214/255, 83/255, 117/255)]

    max_x1 = np.max(curve_data[x1])
    min_x1 = np.min(curve_data[x1])
    max_x2 = np.max(curve_data[x2])
    min_x2 = np.min(curve_data[x2])
    x1_upper = round(1.2 * max_x1)
    x1_lower = round(0.8 * min_x1)
    x2_upper = round(1.2 * max_x2)
    x2_lower = round(0.8 * min_x2)

    fig, ax = plt.subplots(dpi=300)
    # # Diagonal dashed line
    # plt.plot(np.arange(x1_lower,x1_upper), np.arange(x2_lower, x2_upper), '--k', linewidth=2, alpha=.4, label='_nolegend_')
    # Null distribution dashed lines
    # plt.plot([baselines[0], baselines[0]], [x1_lower, x1_upper], '--w', linewidth=2, alpha=.6, label='_nolegend_')
    # plt.plot([x2_lower, x2_upper], [baselines[1], baselines[1]], '--w', linewidth=2, alpha=.6, label='_nolegend_')
    # Scatter plot of all images
    plt.scatter(curve_data[x1], curve_data[x2], c='w', alpha=.1, edgecolors='k', label='_nolegend_')

    # Highlight outliers
    plt.scatter(high_1_high_2[x1], high_1_high_2[x2], color=colors[0], s=s, alpha=0.8)
    plt.scatter(low_1_low_2[x1], low_1_low_2[x2], color=colors[1], s=s, alpha=0.8)
    plt.scatter(high_1_low_2[x1], high_1_low_2[x2], color=colors[2], s=s, alpha=0.8)
    plt.scatter(low_1_high_2[x1], low_1_high_2[x2], color=colors[3], s=s, alpha=0.8)
    
    plt.xlabel(f'{x1}', fontsize=fontsize)
    plt.ylabel(f'{x2}', fontsize=fontsize)

    output_dir = args.wd + f'/results/figures/{args.model_name}/{args.layer}'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/scatter_control.png'
    plt.savefig(img_path)
    plt.clf()


# ------------------ MAIN
curve_data = load_data()
scatter_plot(df=curve_data, data_type='stand')
scatter_plot(df=curve_data, data_type='pixel')
scatter_plot(df=curve_data, data_type='rel')
scatter_control(curve_data=curve_data, n_samples=1000, n_control_videos=25, x1='curve', x2='norm')


