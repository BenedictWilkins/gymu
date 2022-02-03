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
from tqdm.auto import tqdm

def write_episodes(path, episodes):
    with h5py.File(path, "a") as f:
        offset = len(list(f.keys()))
        for i, data in enumerate(tqdm(episodes)):
            g = f.create_group(f"ep-{str(offset + i).zfill(4)}")
            for k,v in data.items():
                g.create_dataset(k.lower(), data=v, compression="gzip")