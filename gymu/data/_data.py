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

from ..mode import mode

__all__ = ('write_episodes', 'read_episodes')

def write_episodes(path, episodes, compression="gzip", write_mode="w", show_progress=False):
    with h5py.File(path, write_mode) as f:
        offset = len(list(f.keys()))
        iter = tqdm(episodes, "Writing episodes:") if show_progress else episodes
        for i, data in enumerate(iter):
            g = f.create_group(f"ep-{str(offset + i).zfill(4)}")
            for k,v in data.items():
                g.create_dataset(k.lower(), data=v, compression=compression)

def read_episodes(path, lazy=True, show_progress=False):
    with h5py.File(path) as f:
        def _load_lazy(group):
            cls = mode(list(group.keys()))
            return cls(**{k:group[k] for k in group.keys()})
        def _load(group):
            cls = mode(list(group.keys()))
            return cls(**{k:group[k][...] for k in group.keys()})
        load = _load_lazy if lazy else _load
        iter = tqdm(f.keys(), desc="Reading episodes:") if show_progress else f.keys()
        return [load(f[k]) for k in iter]


