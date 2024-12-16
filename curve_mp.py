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

def comp_curv_mp(model, model_name, layer, encoded_videos, out_batch, in_batches, n_cpus):

    start_time = time.time()

    shm_name = f'{model_name}_{layer[0:8]}_{out_batch}'

    serialized_data = pickle.dumps(encoded_videos)

    try:
        shm = shared_memory.SharedMemory(create=True, size=len(serialized_data), name=shm_name)
    except FileExistsError:
        shm_old = shared_memory.SharedMemory(shm_name, create=False)
        shm_old.close()
        shm_old.unlink()
        shm = shared_memory.SharedMemory(create=True, size=len(serialized_data), name=shm_name)

    # Create a np.recarray using the buffer of shm
    shm_videos = np.ndarray(len(serialized_data), dtype='B', buffer=shm.buf)

    # Copy the data into the shared memory
    shm_videos[:] = np.frombuffer(serialized_data, dtype='B')

    partial_curv = partial(comp_curves, 
                        model = model,
                        model_name = model_name,
                        layer = layer, 
                        batches = in_batches,
                        data_shape = len(serialized_data),
                        dtype = encoded_videos.dtype,
                        shm_name = shm_name)

    # inner batch 
    bs = range(in_batches)
    pool = mp.Pool(n_cpus)

    try:
        results = pool.map(partial_curv, bs)
    finally:
        pool.close()
        pool.join()

    shm.close()
    shm.unlink()

    return list(results)














