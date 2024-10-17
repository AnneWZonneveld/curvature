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
from multiprocessing import current_process, cpu_count, shared_memory
import json
import urllib
import torch 
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from IPython import embed as shell
import math
from curve_utils import *

from pytorchvideo.data.encoded_video import EncodedVideo

def comp_curv_mp(model, model_name, layer, encoded_videos, batches, batch_size, n_cpus):

    start_time = time.time()

    # # Encode batch of videos
    # batch_files = files[int(batch*batch_size):int((batch+1)*batch_size)]
    # print(f'Batch files {batch_files}')
    # encoded_videos = encode_videos(model_name=model_name, files=batch_files)

    # Create shared memory
    time_stamp = time.time()
    shm_name = f'shm_{model_name}_{time_stamp}'

    try:
        shm = shared_memory.SharedMemory(create=True, size=encoded_videos.nbytes, name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=encoded_videos.nbytes, name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_videos = np.recarray(shape=encoded_videos.shape, dtype=encoded_videos.dtype, buf=shm.buf)

    # Copy the data into the shared memory
    np.copyto(shm_videos, encoded_videos)
    
    partial_curv = partial(comp_curves, 
                        model = model,
                        model_name = model_name,
                        layer = layer, 
                        batches = batches,
                        batch_size = batch_size,
                        data_shape = encoded_videos.shape,
                        dtype = encoded_videos.dtype,
                        shm_name = shm_name)

    bs = range(batches)
    pool = mp.Pool(n_cpus)

    try:
        results = pool.map(partial_curv, bs)
    finally:
        pool.close()
        pool.join()

    shm.close()
    shm.unlink()

    return list(results)














