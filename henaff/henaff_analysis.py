"""
Use python 3.7

Code to replicate henaff curvature analysis in python. 
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
    """ Compute curvatures """
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

# Define the image transformation (Resize, Normalize, etc.)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_image(image_path):
    img = Image.open(image_path).convert('L')  
    img_tensor = preprocess(img)  # Preprocess it
    return img_tensor.unsqueeze(0)

def load_images():

    # Load images (n_videos x n_frames x n_channels x height x width)
    stim_dir = '/Users/annewzonneveld/Documents/phd/projects/curvature/henaff/henaff_stimuli/'
    img_array = np.zeros((10, 11, 3, 512, 512)) # or one channel?
    for video in range(10):
        vid_name = f'movie{video+1}/'
        vid_folder = stim_dir + vid_name
        img_paths = sorted(glob.glob(f'{vid_folder}/natural*'))
        for img_idx in range(len(img_paths)):
            img = load_image(img_paths[img_idx])
            img_array[video, img_idx, :, :, :] = img[np.newaxis, ...]
    
    return img_array


# ------------------ MAIN
vids_array = load_images()

# Compute curvatures
vids_curvs = np.zeros((vids_array.shape[0], vids_array.shape[1] - 2))
mean_curvs = np.zeros(vids_array.shape[0])

for i in range(vids_array.shape[0]):
    curvs, mean_curve = comp_curvatures(vids_array[i, :, :, :])
    vids_curvs[i, :] = curvs
    mean_curvs[i] = mean_curve
