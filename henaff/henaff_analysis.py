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
from sklearn.utils import resample
from IPython import embed as shell
from PIL import Image
import torch
from torchvision import transforms

def comp_speed(vid_array):
    '''Compute
    - velocity = difference vectors
    - norm of difference vectors
    for specificied sequence of images'''

    # vid_array = img_array[0, :, :, :, :]
    num_frames = vid_array.shape[0]
    dif_vectors = [vid_array[i, : , :, :].flatten() -  vid_array[i+1, : , :, :].flatten().flatten() for i in range(num_frames-1)]
    dif_vectors =  np.vstack(tuple(dif_vectors))
    norms = np.zeros(num_frames)

    for i in range(num_frames - 1):
        norms[i] = np.linalg.norm(dif_vectors[i]) + 1e-8  # Compute the norm (magnitude) of each difference vector
        dif_vectors[i] /= norms[i]  # Normalize the difference vector
    
    return dif_vectors, norms

def comp_curvatures(vid_array):
    """ Compute curvatures based on angles of velocity vectors. """
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


def load_image(image_path, goal='curve_test'):

    # Define the image transformation (Resize, Normalize, etc.)
    if goal == 'curve_test':
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif goal == 'alex':
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    img = Image.open(image_path).convert('L')  
    img_tensor = preprocess(img)  # Preprocess it
    return img_tensor.unsqueeze(0)

def load_images(goal='curve_test'):

    # Load images (n_videos x n_frames x n_channels x height x width)
    stim_dir = '/Users/annewzonneveld/Documents/phd/projects/curvature/henaff/henaff_stimuli/'
    if goal == 'curve_test':
        img_array = np.zeros((10, 11, 3, 512, 512)) 
    elif goal == 'alex':
        img_array = np.zeros((10, 11, 3, 224, 224)) 

    for video in range(10):
        vid_name = f'movie{video+1}/'
        vid_folder = stim_dir + vid_name
        img_paths = sorted(glob.glob(f'{vid_folder}/natural*'))
        for img_idx in range(len(img_paths)):
            img = load_image(img_paths[img_idx], goal=goal)
            img_array[video, img_idx, :, :, :] = img[np.newaxis, ...]
    
    return img_array

def curvature_test():

    # Load videos 
    vids_array = load_images(goal='curve_test')
    vids_curvs = np.zeros((vids_array.shape[0], vids_array.shape[1] - 2))
    mean_curvs = np.zeros(vids_array.shape[0])

    for i in range(vids_array.shape[0]):
        curvs, mean_curve = comp_curvatures(vids_array[i, :, :, :])
        vids_curvs[i, :] = curvs
        mean_curvs[i] = mean_curve

    print('mean curves for natural videos')
    print(mean_curvs)


features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def model_test():
    vids_array = load_images(goal='alex')

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    model.eval()

    # Register layers of interest
    model.features[2].register_forward_hook(get_features('max1'))
    model.features[5].register_forward_hook(get_features('max2'))
    model.features[12].register_forward_hook(get_features('max3'))

    curve_dict = {
        'max1': [],
        'max2': [],
        'max3': []
    }

    # Extract features
    for video in range(vids_array.shape[0]):
        images = torch.from_numpy(vids_array[video]).to(torch.float32)

        with torch.no_grad():
            _ = model(images) 
    
        # Calc curvature
        for key in features:
            activations = features[key]
            _, curve = comp_curvatures(activations)
            curve_dict[key].append(curve)



# ------------------ MAIN

#Curvatures calculation test
curvature_test()

# Modelling test
shell()
