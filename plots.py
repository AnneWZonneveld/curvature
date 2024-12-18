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

# ------------------- Input arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--wd', default='/home/azonnev/analyses/curvature', type=str)
parser.add_argument('--wd', default='/Users/annewzonneveld/Documents/phd/projects/curvature', type=str)
parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
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
def load_data(model_name, layer, pretrained=1):
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
    if args.model_name == 'slowfast_r50':
        if layer == 'early': 
            layer = 'blocks.0.multipathway_fusion.activation'
        elif layer == 'mid':
            layer = 'blocks.2.multipathway_fusion.activation'
        elif layer == 'late':
            layer = 'blocks.4.multipathway_fusion'
    elif model_name in  ['slow_r50', 'r2plus1d_r50']:
        if layer == 'early': 
            layer = 'blocks.0.activation'
        elif layer == 'mid':
            layer = 'blocks.2.res_blocks.3.activation'
        elif layer == 'late':
            layer = 'blocks.4.res_blocks.2.activation'
    elif model_name in ['i3d_r50', 'c2d_r50']:
        if layer == 'early': 
            layer = 'blocks.1.res_blocks.2.activation'
        elif layer == 'mid':
            layer = 'blocks.3.res_blocks.3.activation'
        elif layer == 'late':
            layer = 'blocks.5.res_blocks.2.activation'
    elif model_name == 'csn_r101':
        if model_name == 'early': 
            model_name = 'blocks.0.activation'
        elif model_name == 'mid':
            model_name = 'blocks.3.res_blocks.8.activation'
        elif model_name == 'late':
            model_name = 'blocks.4.res_blocks.2.activation'
    elif model_name == 'x3d_s':
        if model_name == 'early': 
            model_name = 'blocks.0.activation'
        elif model_name == 'mid':
            model_name = 'blocks.3.res_blocks.4.activation'
        elif model_name == 'late':
            model_name = 'blocks.4.res_blocks.6.activation'
    elif model_name == 'mvit_base_16x4':
        if model_name == 'early': 
            model_name = 'blocks.0.mlp'
        elif model_name == 'mid':
            model_name = 'blocks.7.mlp.fc2'
        elif model_name == 'late':
            model_name = 'blocks.15.mlp.fc2'
    
    # Static models
    elif model_name == 'resnet50':
        if layer == 'early': 
            layer = 'layer1.2.relu'
        elif layer == 'mid':
            layer = 'layer2.3.relu'
        elif layer == 'late':
            layer = 'layer4.2.relu'

    curv_dir = args.wd + f'/results/features/{model_name}/pt_{pretrained}/{layer}/'
    curv_file = curv_dir + 'curve_props_df.pkl'

    with open(curv_file, 'rb') as f:
        curv_data = pickle.load(f)
    
    return curv_data


def distr_plot(model_name, layer, pretrained):
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
    df = load_data(model_name=model_name, layer=layer, pretrained=pretrained)

    # Reformat data
    df_melted = pd.melt(df, value_vars=['pixel_curve', 'curve'], var_name='type', value_name='curve')

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


def pt0_vs_pt1_rel_distr_plot(model_name, layer):
    """ 
    Plots the relative curvature for pretrained vs untrained activations.

    Parameters
    ----------

    layer: str

    """

    # Load data
    print('Loading data')
    pt1_df = load_data(model_name=model_name, layer=layer, pretrained=1)
    pt0_df = load_data(model_name=model_name, layer=layer, pretrained=0)

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


def layers_rel_curve_plot(model_name, pretrained=1):
    """ 
    Plots the relative curvature for different layers.

    Parameters
    ----------

    pretrained: bool

    """

    # Load data
    print('Loading data')
    data_dir = args.wd + f'/results/features/{model_name}/pt_{pretrained}/'
    layers = os.listdir(data_dir)
    data_dict = {layer: load_data(model_name=model_name, layer=layer, pretrained=1) for layer in layers}

    all_rel_curvs = []
    all_layer_names = []
    for i in range(len(layers)):
        layer = layers[i]
        all_rel_curvs.extend(data_dict[layer]['rel_curve'])

        if i == 0:
            layer_name = 'early'
        elif i == 1:
            layer_name = 'mid'
        elif i == 2:
            layer_name = 'late'

        layer_names = [layer_name]*len(data_dict[layer]['rel_curve'])
        all_layer_names.extend(layer_names)

    # Append 'pixel' category with value 0
    all_rel_curvs.append(0)
    all_layer_names.append('pixel')

    # Reformat data
    df = pd.DataFrame()
    df['rel_curve'] = all_rel_curvs
    df['layer'] = all_layer_names
    df['layer'] = pd.Categorical(df['layer'], categories=['pixel', 'early', 'mid', 'late'], ordered=True)

    # Plot
    fig, ax = plt.subplots(dpi=300)
    sns.pointplot(data=df[df['layer'] != 'pixel'], x="layer", y="rel_curve", join=False, markers='o', dodge=True, ax=ax)

    # Add the 'pixel' category as a single gray point
    pixel_y = df[df['layer'] == 'pixel']['rel_curve'].values[0]
    ax.scatter('pixel', pixel_y, color='gray', s=100, label='pixel', zorder=5)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)

    ax.set_title(f'{model_name}')
    ax.set_xlabel('layer')
    ax.set_ylabel('Relative curve')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    # Save the figure
    output_dir = args.res_dir + f'/figures/{model_name}/pt_{pretrained}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = output_dir + f'/rel_curve_layers.png'
    plt.savefig(img_path)
    plt.clf()


def layers_abs_curve_plot(pretrained=1, model_name):
    """ 
    Plots the absolute pixel and feature curvature for different layers.

    Parameters
    ----------

    pretrained: bool

    """

    # Load data
    print('Loading data')
    data_dir = args.wd + f'/results/features/{model_name}/pt_{pretrained}/'
    layers = os.listdir(data_dir)
    data_dict = {layer: load_data(model_name=model_name, layer=layer, pretrained=1) for layer in layers}

    all_curvs = []
    all_data_types = []
    all_layer_names = []
    for i in range(len(layers)):
        layer = layers[i]

        if i == 0:
            layer_name = 'early'
        elif i == 1:
            layer_name = 'mid'
        elif i == 2:
            layer_name = 'late'

        layer_names = [layer_name] * len(data_dict[layer]['rel_curve'])

        # Add pixel_curve data
        all_curvs.extend(data_dict[layer]['pixel_curve'])
        all_layer_names.extend(layer_names)
        all_data_types.extend(['pixel'] * len(data_dict[layer]['pixel_curve']))

        # Add feature curve data
        all_curvs.extend(data_dict[layer]['curve'])
        all_layer_names.extend(layer_names)
        all_data_types.extend(['feature'] * len(data_dict[layer]['curve']))

    # Append the pixel category with value 0
    all_curvs.append(0)
    all_layer_names.append('pixel')
    all_data_types.append('pixel')

    # Reformat data
    df = pd.DataFrame()
    df['curve'] = all_curvs
    df['layer'] = all_layer_names
    df['data_type'] = all_data_types

    # Ensure correct categorical order for layers
    df['layer'] = pd.Categorical(df['layer'], categories=['pixel', 'early', 'mid', 'late'], ordered=True)

    # Plot
    fig, ax = plt.subplots(dpi=300)

    # Plot the feature data points excluding the pixel category
    sns.pointplot(data=df[df['layer'] != 'pixel'], x="layer", y="curve", hue="data_type", join=False, markers='o', dodge=True, ax=ax)
    ax.scatter('pixel', 0, color='gray', s=100, label='pixel', zorder=5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('layer')
    ax.set_ylabel('Relative curve')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    # Save
    output_dir = args.res_dir + f'/figures/{model_name}/pt_{pretrained}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = output_dir + f'/abs_curve_layers.png'
    plt.savefig(img_path)
    plt.clf()


def layers_dyn_vs_static(static_model, dynamic_model, pretrained=1):
    """ 
    Plots the relative curvature for different layers.

    Parameters
    ----------

    pretrained: bool

    """

    # Load data
    print('Loading data')
    static_dir = args.wd + f'/results/features/{static_model}/pt_{pretrained}/'
    dynamic_dir = args.wd + f'/results/features/{dynamic_model}/pt_{pretrained}/'
    static_layers = os.listdir(static_dir)
    dynamic_layers = os.listdir(dynamic_dir)
    static_dict = {layer: load_data(model_name=static_model, layer=layer, pretrained=1) for layer in static_layers}
    dynamic_dict = {layer: load_data(model_name=dynamic_model, layer=layer, pretrained=1) for layer in dynamic_layers}

    all_curvs = []
    all_model_types = []
    all_layer_names = []
    for i in range(len(layers)):
        static_layer = static_layers[i]
        dynamic_layer = dynamic_layers[i]

        if i == 0:
            layer_name = 'early'
        elif i == 1:
            layer_name = 'mid'
        elif i == 2:
            layer_name = 'late'

        layer_names = [layer_name] * len(static_dict[static_layer]['rel_curve'])

        # Add dynamic curve data
        all_curvs.extend(dynamic_dict[dynamic_layer]['rel_curve'])
        all_layer_names.extend(layer_names)
        all_model_types.extend(['dynamic'] * len(dynamic_dict[dynamic_layer]['rel_curve']))

        # Add static curve data
        all_curvs.extend(static_dict[static_layer]['rel_curve'])
        all_layer_names.extend(layer_names)
        all_model_types.extend(['static'] * len(static_dict[static_layer]['rel_curve']))

    # Append the pixel category with value 0
    all_curvs.append(0)
    all_layer_names.append('pixel')
    all_model_types.append('pixel')

    # Reformat data
    df = pd.DataFrame()
    df['curve'] = all_curvs
    df['layer'] = all_layer_names
    df['model'] = all_data_types
    df['layer'] = pd.Categorical(df['layer'], categories=['pixel', 'early', 'mid', 'late'], ordered=True)

    # Plot
    fig, ax = plt.subplots(dpi=300)
    sns.pointplot(data=df[df['layer'] != 'pixel'], x="layer", y="curve", hue="", join=False, markers='o', dodge=True, ax=ax)
    ax.scatter('pixel', 0, color='gray', s=100, label='pixel', zorder=5)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_title(f'{dynamic_model}')
    ax.set_xlabel('layer')
    ax.set_ylabel('Relative curve')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    # Save
    output_dir = args.res_dir + f'/figures/{dynamic_model}/pt_{pretrained}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = output_dir + f'/dyn_vs_static_layers.png'
    plt.savefig(img_path)
    plt.clf()


def compare_models(pretrained=1)

    models = ['resnet50','i3d_r50']

    # Load data
    print('Loading data')
    all_models = []
    all_layers = []
    all_curves = []
    all_model_types = []
    for model_name in models:
        model_dir = args.wd + f'/results/features/{model_name}/pt_{pretrained}/'

        if model_name in ['resnet50']:
            model_type = 'static'
        elif model_name in ['i3d_r50']:
            model_type = 'dynamic'

        layers = sorted(os.listdir(model_dir))
        for i in range(len(layers)):
            
            if i == 0:
                layer_name = 'early'
            elif i == 1:
                layer_name = 'mid'
            elif i == 2:
                layer_name = 'late'

            layer = layers[i]
            data = load_data(model_name=model_name, layer=layer, pretrained=1) 
            all_models.extend(len(data)*[model_name])
            all_layers.extend(len(data)*[layer_name])
            all_model_types.extend(len(data)*[model_type])
            all_curves.extend(list(data['rel_curve']))
 
    df = pd.DataFrame()
    df['model'] = all_models
    df['layer'] = all_layers
    df['rel_curve'] = all_curves
    df['model_type'] = all_model_types

    # Create subplots
    layers = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, len(layers), figsize=(15, 5), dpi=300, sharey=True)

    # Iterate over layers and corresponding subplot axes
    for i, layer in enumerate(layers):
        ax = axes[i]

        # Filter data for the current layer
        layer_data = df[df["layer"] == layer]
        
        # Create boxplot for the current layer
        sns.boxplot(
            data=layer_data,
            x="model",
            y="rel_curve",
            hue="model_type",
            ax=ax,
            dodge=True,
        )
        
        # Customize the axis and title
        ax.set_title(f"{layer.capitalize()} Layer", fontsize=10)
        ax.set_xlabel("Model")  # Set x-axis label as "Model"
        if i == 0:
            ax.set_ylabel("Straightening")  # Add y-axis label only to the first subplot
        else:
            ax.set_ylabel("")  # Remove y-axis label from other subplots
        
        # Move the legend to the last subplot
        if i < len(layers) - 1:
            ax.get_legend().remove()

        sns.despine(offset=10, top=True, right=True)

    # Add a single legend on the last subplot
    handles, labels = ax.get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Model Type", loc="upper right", fontsize=8)

    # Adjust subplot layout
    fig.tight_layout()

    # Save
    output_dir = args.res_dir + f'/figures/compare_models/pt_{pretrained}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = output_dir + f'/all_models.png'
    plt.savefig(img_path)
    plt.clf()

# ---------------- MAIN

shell()

# # Plot distributions
# distr_plot(model_name='i3d_r50', layer='late', pretrained=1)

# # Plot untrained vs pretrained relative curve plots
# pt0_vs_pt1_rel_distr_plot(model_name='i3d_r50',layer = 'late')

# Plot relative curve for different layers
layers_rel_curv_plot(model_name='i3d_r50', pretrained=1)
layers_abs_curv_plot(model_name='i3d_r50', pretrained=1)

# Plot relationships variables of interest 
# scatter_plot(df=curve_data, x='curve', y='norm')
# scatter_plot(df=curve_data, x='lse', y='norm')
# scatter_plot(df=curve_data, x='curve', y='lse')
# scatter_plot(df=curve_data, x='pixel_curve', y='pixel_norm')
# scatter_plot(df=curve_data, x='pixel_lse', y='pixel_norm')
# scatter_plot(df=curve_data, x='pixel_curve', y='pixel_lse')
# scatter_plot(df=curve_data, x='rel_curve', y='norm')
# scatter_plot(df=curve_data, x='rel_lse', y='norm')