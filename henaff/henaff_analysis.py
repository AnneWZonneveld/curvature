"""
Use python 3.7

Code to replicate henaff curvature calculations in python. 
Based on repository : https://github.com/olivierhenaff/neural-straightening/tree/master
Able to replicate mean curvature measurements as given in 'acute data -combined_235x.csc' file 
https://osf.io/vf2xk/

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
import math
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import embed as shell
from PIL import Image
import torch
from torchvision import transforms

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
parser.add_argument('--model_name', default='resnet50', type=str)
parser.add_argument('--data_dir', default='/home/azonnev/analyses/curvature/henaff/henaff_stimuli', type=str)
parser.add_argument('--res_dir', default='/Users/annewzonneveld/Documents/phd/projects/curvature/results', type=str)
args = parser.parse_args()

print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

start_time = time.time()

#### ----------------

def comp_speed(vid_array):
    """
    Compute velocity (=difference vectors) and norm for specified sequence of images

    Parameters
    ----------

    vid_array: 4 dimensional tensor of image sequence (# of frames, # channels, height, width)

    returns
    - diff_vectors: list of length vid_array.shape[0] - 1
    - norms: list of length vid_array.shape[0] - 1

    """

    num_frames = vid_array.shape[0]
    dif_vectors = [vid_array[i, : , :, :].flatten() -  vid_array[i+1, : , :, :].flatten().flatten() for i in range(num_frames-1)]
    dif_vectors =  np.vstack(tuple(dif_vectors))
    norms = np.zeros(num_frames - 1) 

    for i in range(num_frames - 1):
        norms[i] = np.linalg.norm(dif_vectors[i]) + 1e-8  # Compute the norm (magnitude) of each difference vector
        dif_vectors[i] /= norms[i]  # Normalize the difference vector
    
    return dif_vectors, norms

def comp_curvatures(vid_array):
    """
    Compute curvature based on angles of velocity vectors,
    as calculated from sequnec of images.

    Parameters
    ----------

    vid_array: 4 dimensional tensor of image sequence (# of frames, # channels, height, width)

    returns
    - curvs: list of length vid_array.shape[0] - 2
    - mean_curve: float of mean curvature over image sequence

    """

    num_frames = vid_array.shape[0]
    dif_vectors, norms = comp_speed(vid_array)

    curvs = np.zeros(num_frames - 2)
    angles = np.zeros(num_frames - 2)
    for i in range(num_frames - 2):
        cos = np.clip(np.dot(dif_vectors[i], dif_vectors[i+1]), -1, 1)
        angles[i] = math.acos(cos)
        curvs[i] = math.degrees(math.acos(cos)) #angle in degrees
    
    # mean_curve_angle = sum(angles) * 180 / (len(angles)*math.pi)

    mean_curve = np.mean(curvs)

    return curvs, mean_curve


def load_image(image_path, goal='pixel'):
    """
    Load and preprocess a single image with according settings.

    Parameters
    ----------

    img_path: str that specifies image path
    goal: str that specifies preprocessing settings

    returns
    - image_tensor: torch.tensor of size (1, # channels, height , width)

    """

    if goal == 'pixel':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif goal in ['alexnet', 'vgg19', 'resnet50', 'c2d_r50']:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    img = Image.open(image_path).convert('L')  
    img_tensor = preprocess(img)  

    return img_tensor.unsqueeze(0)

def load_video(image_path):
    """
    Load and preprocess a single image as 'video' by constructing a tensor made of 
    repititions of same frame with correct size (3 channels x 8 frames x height x width)

    Parameters
    ----------

    img_path: str that specifies image path

    returns
    - image_rf: torch.tensor of size (# channels, #frames, height , width)

    """
    
    img = load_image(image_path, goal='resnet50')
    img_rgb = img.squeeze(0).repeat(3, 1, 1)
    img_rf = torch.cat([img_rgb.unsqueeze(0)] * 8, dim=0)
    img_rf = img_rf.permute(1, 0, 2, 3) 

    return img_rf


def load_images(goal='pixel', sequence='natural'):
    """
    Load images for all sequences of specified sequence type into numpy array.

    Parameters
    ----------

    goal: str that specifies preprocess settings
    sequence: str taht specifies sequence type ('natural', 'synthetic', 'contrast')

    returns
    - imag array: torch.tensor of size (# sequences, #frames, #channels, height , width)
                  or in case of c2d_r50 (# sequences, #frames, #channels, #time steps, height , width)

    """

    # Specify individual image names
    if sequence == 'natural':
        frame_names = ['natural01',   'natural02',   'natural03',   'natural04',   'natural05',   'natural06',   'natural07',   'natural08',   'natural09',   'natural10', 'natural11']
    elif sequence == 'synthetic':
        frame_names = ['natural01', 'synthetic02', 'synthetic03', 'synthetic04', 'synthetic05', 'synthetic06', 'synthetic07', 'synthetic08', 'synthetic09', 'synthetic10', 'natural11']
    elif sequence == 'contrast':
        frame_names = ['contrast01',  'contrast02',  'contrast03',  'contrast04',  'contrast05',  'contrast06',  'contrast07',  'contrast08',  'contrast09',  'contrast10', 'natural01']

    # Load images (n_videos x n_frames x n_channels x height x width)
    if goal == 'pixel':
        img_array = np.zeros((10, 11, 3, 512, 512)) 
    elif goal in ['alexnet', 'vgg19', 'resnet50']:
        img_array = np.zeros((10, 11, 3, 224, 224)) 
    elif goal == 'c2d_r50':
        img_array = np.zeros((10, 11, 3, 8, 224, 224)) 

    if sequence in ['natural', 'synthetic']:
        for video in range(10):
            vid_name = f'movie{video+1}/'
            vid_folder = args.data_dir + vid_name
            for i in range(len(frame_names)): 
                img_path = vid_folder + f'{frame_names[i]}.png'
                if goal == 'c2d_r50':
                    # Load image in as video 
                    img = load_video(img_path)
                    shell()
                    img_array[video, i, :, :, :, :] = img
                else:
                    # Load image
                    img = load_image(img_path, goal=goal)
                    img_array[video, i, :, :, :] = img[np.newaxis, ...]
    else: 
        contrast_videos = [1, 3, 6, 9]
        for video in range(10):
            vid_name = f'movie{video+1}/'
            vid_folder = args.data_dir + vid_name
            if video in contrast_videos:
                for i in range(len(frame_names)): 
                    img_path = vid_folder + f'{frame_names[i]}.png'
                    if goal == 'c2d_r50':
                        # Load image in as video 
                        img = load_video(img_path)
                        img_array[video, i, :, :, :, :] = img
                    else:
                        # Load image
                        img = load_image(img_path, goal=goal)
                        img_array[video, i, :, :, :] = img[np.newaxis, ...]

    return img_array

def curvature_test(sequence='natural'):
    """
    Calculate curvature for specified sequencees based on pixel values.

    Parameters
    ----------
    sequence: str tht specifies sequence type ('natural', 'synthetic', 'contrast')

    returns
    - mean_curvs: list of float of vids_array.shape[0] = 10 sequences

    """

    # Load videos 
    vids_array = load_images(goal='pixel', sequence=sequence)
    vids_curvs = np.zeros((vids_array.shape[0], vids_array.shape[1] - 2))
    mean_curvs = np.zeros(vids_array.shape[0])


    if sequence in ['natural', 'synthetic']:
        for i in range(vids_array.shape[0]): 
            curvs, mean_curve = comp_curvatures(vids_array[i, :, :, :])
            vids_curvs[i, :] = curvs
            mean_curvs[i] = mean_curve
    else:
        contrast_videos = [1, 3, 6, 9]  #only contrast images available for specified sequences
        for i in range(vids_array.shape[0]): 
            if i in contrast_videos:
                curvs, mean_curve = comp_curvatures(vids_array[i, :, :, :])
                vids_curvs[i, :] = curvs
                mean_curvs[i] = mean_curve
            else:
                mean_curvs[i] = np.nan

    return(mean_curvs)


def model_curves(model, model_name = 'alexnet', sequence='natural'):
    """
    Calculate curvature for specified sequencees based on model activations 
    for several layers.

    Parameters
    ----------
    model: model object
    model_name: str of model name
    sequence: str that specifies sequence type ('natural', 'synthetic', 'contrast')

    returns
    - curve_dict: dict for which keys correspond to model layers, which each map to list of mean_curv values (size 10 = # sequences)
    """

    # Dictionay to save model features to 
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Preproces sequences and load model
    vids_array = load_images(goal=model_name, sequence=sequence)

    # Register layers of interest
    if model_name == 'alexnet':
        model.features[2].register_forward_hook(get_features('max1'))
        model.features[5].register_forward_hook(get_features('max2'))
        model.features[12].register_forward_hook(get_features('max3'))

        curve_dict = {
            'max1': [],
            'max2': [],
            'max3': []
        }

    elif model_name == 'vgg19':

        model.features[4].register_forward_hook(get_features('max1'))
        model.features[9].register_forward_hook(get_features('max2'))
        model.features[16].register_forward_hook(get_features('max3'))
        model.features[23].register_forward_hook(get_features('max4'))
        model.features[30].register_forward_hook(get_features('max5'))

        curve_dict = {
            'max1': [],
            'max2': [],
            'max3': [],
            'max4': [],
            'max5': []
        }
    
    elif model_name == 'resnet50':

        model.layer1[2].relu.register_forward_hook(get_features('layer1'))
        model.layer2[3].relu.register_forward_hook(get_features('layer2'))
        model.layer3[5].relu.register_forward_hook(get_features('layer3'))
        model.layer4[2].relu.register_forward_hook(get_features('layer4'))

        curve_dict = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': []
        }
    
    elif model_name == 'c2d_r50':

        model.blocks[1].res_blocks[2].activation.register_forward_hook(get_features('layer1'))
        model.blocks[3].res_blocks[3].activation.register_forward_hook(get_features('layer2'))
        model.blocks[4].res_blocks[5].activation.register_forward_hook(get_features('layer3'))
        model.blocks[5].res_blocks[2].activation.register_forward_hook(get_features('layer4'))

        curve_dict = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'layer4': []
        }

    
    # Extract features
    if sequence in ['natural', 'synthetic']:
        for video in range(vids_array.shape[0]):
            print(f'Processing vid {video}')

            images = torch.from_numpy(vids_array[video]).to(torch.float32)
            
            with torch.no_grad():
                _ = model(images) 
        
            # Calc curvature
            for key in curve_dict:
                activations = features[key]
                if model_name == 'c2d_r50':
                    activations  = activations.mean(dim=2) # average over model time steps
                _, curve = comp_curvatures(activations)
                curve_dict[key].append(curve)
    else: 
        contrast_videos = [1, 3, 6, 9] #only contrast images available for specified sequences
        for video in range(vids_array.shape[0]): 
            print(f'Processing vid {video}')
            if video in contrast_videos:

                images = torch.from_numpy(vids_array[video]).to(torch.float32)

                with torch.no_grad():
                    _ = model(images) 
            
                # Calc curvature
                for key in curve_dict:
                    activations = features[key]
                    if model_name == 'c2d_r50':
                        activations  = activations.mean(dim=2) # average over model time steps
                    _, curve = comp_curvatures(activations)
                    curve_dict[key].append(curve)
            else:
                for key in curve_dict:
                    curve_dict[key].append(np.nan)
     
    return curve_dict


def model_test(model_name='alexnet'):
    """
    Calculate relative curvature, i.e. straightening, for specified model. 

    Parameters
    ----------
    model_name: str tht specifies model name

    returns
    - plot of relative curvature over layers for specified model

    """

    # Load model
    if model_name == 'c2d_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True, force_reload=True)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    model.eval()

    sequences = ['natural', 'synthetic', 'contrast']
    sequence_names = []
    layer_names = []
    all_rel_curves = []
    for sequence in sequences:

        print(f'Processing {sequence} sequences')

        # Calculate mean pixel curves
        mean_pix_curves = curvature_test(sequence=sequence)

        # Calculate mean model curves
        mean_model_curves = model_curves(model = model, model_name=model_name, sequence=sequence)

        # Calculate rel curve per layer
        rel_curves_dict = {}
        for key in mean_model_curves.keys():
            layer_curves = mean_model_curves[key]
            rel_curves =  np.array(layer_curves) - np.array(mean_pix_curves)

            layer_names.extend([key]*len(rel_curves))
            sequence_names.extend([sequence]*len(rel_curves))
            all_rel_curves.extend(rel_curves)
        
        layer_names.append('pixel')
        sequence_names.append(sequence)
        all_rel_curves.append(0) 

    # Reformat data
    df = pd.DataFrame()
    df['rel_curve'] = all_rel_curves
    df['layer'] = layer_names
    df['sequence'] = sequence_names
    df['layer'] = pd.Categorical(df['layer'], categories=['pixel'] + sorted(sorted(df['layer'].unique().tolist(), reverse=True)[1:]))

    # Create plot 
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
    sns.pointplot(x="layer", y="rel_curve", hue="sequence", hue_order=['natural', 'contrast', 'synthetic'], data=df[df['layer'] != 'pixel'], join=False, markers='o', dodge=True)
    plt.scatter(x=['pixel'], y=[0], color='gray', s=100, zorder=5, label='Pixel (0)')
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.7)
    ax.set_title(f'{model_name}')
    ax.set_xlabel('layer')
    ax.set_ylabel('Relative curve')
    sns.despine(offset=10, top=True, right=True)
    fig.tight_layout()

    output_dir = args.res_dir + f'/figures/{model_name}/'
    if not os.path.exists(output_dir) == True:
        os.makedirs(output_dir)

    img_path = output_dir + f'/henaff_model.png'
    plt.savefig(img_path)
    plt.clf()


# ------------------ MAIN
# Setting and checking cache
os.environ['TORCH_HOME'] = '/ivi/zfs/s0/original_homes/azonnev/.cache'
print("cache log: ")
print(os.getenv('TORCH_HOME'))

# Curvatures calculation test
# mean_pix_curves = curvature_test()
# print('mean pixel curves')
# print(f'{mean_pix_curves}')

# Modelling test
# model_test(model_name='alexnet')
# # model_test(model_name='vgg19')
# # model_test(model_name='resnet50')
model_test(model_name=args.model_name)

end_time = time.time()
print(f'Script done in {(end_time - start_time)/60} mins')
