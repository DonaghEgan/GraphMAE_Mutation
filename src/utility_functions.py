from torch_geometric.data import download_url, extract_zip, extract_gz
from typing import Optional, List
import matplotlib.pyplot as plt
import psutil
import os
import torch
import numpy as np

def explore_structure(d, indent=0):
    spacing = '  ' * indent
    if isinstance(d, dict):
        for key, value in d.items():
            print(f"{spacing}{key}: {type(value).__name__}")
            explore_structure(value, indent + 1)
    elif isinstance(d, list):
        print(f"{spacing}List of {len(d)} items")
        if d:  # Only recurse if list is non-empty
            explore_structure(d[0], indent + 1)

def move_batch_to_device(batch, device):
    batch.omics = batch.omics.to(device)
    batch.clin = batch.clin.to(device)
    batch.osurv = batch.osurv.to(device)
    batch.sample_meta = batch.sample_meta.to(device)
    return batch

def merge_last_two_dims(x):
    # x.shape == (1661, 554, D, 12)
    n0, n1, D, C = x.shape
    return x.reshape(n0, n1, D*C)

def log_memory(step):
    process = psutil.Process(os.getpid())
    cpu_mem_mb = process.memory_info().rss / (1024 ** 2)
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_info = f", GPU Memory: {gpu_mem_mb:.2f} MB"
    print(f"{step}: CPU Memory: {cpu_mem_mb:.2f} MB{gpu_info}")


