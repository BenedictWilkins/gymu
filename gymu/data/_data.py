#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 03-02-2022 15:42:18

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


import h5py
import numpy as np 
from collections import OrderedDict
from tqdm.auto import tqdm

__all__ = ('write_episodes', 'read_episodes')

# DEPRECATED? use _tar instead?

def write_episodes(path, episodes, compression="gzip", write_mode="w", show_progress=False):
    with h5py.File(path, write_mode) as f:
        offset = len(list(f.keys()))
        iter = tqdm(episodes, "Writing episodes:") if show_progress else episodes
        for i, data in enumerate(iter):
            g = f.create_group(f"ep-{str(offset + i).zfill(4)}")
            for k,v in data.items():
                g.create_dataset(k.lower(), data=v[...], compression=compression)

def read_episodes(path, lazy=False, show_progress=False):
    f = h5py.File(path)
    def _load_lazy(group):
        return {k:group[k] for k in group.keys()}
    def _load(group):
        return {k:group[k][...] for k in group.keys()}
    
    iter = tqdm(f.keys(), desc="Reading episodes:") if show_progress else f.keys()
    if lazy:
        return [_load_lazy(f[k]) for k in iter], f
    else:
        return [_load(f[k]) for k in iter]






    





