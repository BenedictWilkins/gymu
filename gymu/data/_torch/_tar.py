#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import torch
import pathlib
import itertools
import tarfile
import io

from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, TensorDataset

from itertools import groupby
from more_itertools import pairwise, windowed

from ...mode import STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO

__all__ = ("write_episode",)

def write_episode(iterator, path="./episode/"):
    """
        Write iterator data to a tar archive. May be used with the standard gymu iterator to create a dataset on disk. 

        Example:
        '''
            env = gymu.make("SpaceInvaders-v0")
            iterator = gymu.iterator(env, mode=gymu.mode.sar)
            gymu.data.write_episode(iterator)
        '''
    """
    path = pathlib.Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    path_tar = pathlib.Path(path, f"../{path.stem}.tar.gz").resolve()
    with tarfile.open(str(path_tar), "w:gz") as tar:
        for i, x in enumerate(tqdm(iterator)):
            file = pathlib.Path(path, str(i).zfill(8) + ".npz")
            for k, v in dict(x).items():        
                if v is None:
                    del x[k]
            np.savez(file, **x)
            tar.add(str(file), arcname=file.name)
            file.unlink() # delete file
    path.rmdir()



        






