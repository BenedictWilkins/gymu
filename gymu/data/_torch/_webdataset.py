#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"



import webdataset as wb
import numpy as np
import torch
import itertools
import more_itertools
import io

from collections.abc import Iterable
from typing import List
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset, TensorDataset, DataLoader

from ...mode import STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO
from ...utils import overload
from . import iterators

__all__ = ("dataset",)

class GymuShorthands:

    @property
    def gymu(self):
        return _GymuShorthands(self)

# patch the wb.Process class to inherit from our custom Shorthands
wb.Processor.__bases__ += (GymuShorthands,) 
#type(wb.Processor.__name__ + "WithGymuShorthands", (wb.Processor, GymuShorthands), {})

class _GymuShorthands:

    def __init__(self, source):
        super().__init__()
        self.source = source

    # access like dataset.gymu.decode
    def decode(self, keep_meta=False):
        return self.source.then(iterators.decode, keep_meta=keep_meta)
    
    def mode(self, mode, ignore_last=True):
        return self.source.then(iterators.mode, mode, ignore_last=ignore_last)

    def keep(self, keys=[STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO]):
        return self.source.then(iterators.keep, keys=keys)

    def window(self, window_size=2):
        return self.source.then(iterators.window, window_size=window_size)

    def to_tensor_dataset(self, num_workers=0, show_progress=False): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')                                  
        source = DataLoader(self.source, batch_size=512, shuffle=False, num_workers=num_workers, drop_last=False)   
        source = source if not show_progress else tqdm(source, desc="Loading Tensor Dataset")                                                                          
        source = iter(source)                                                                       
        tensors = [[y] for y in next(source)]                     
        for x in source:                                                                          
            for z,t in zip(x, tensors):                                          
                t.append(z)                             
        tensors = [torch.cat(z,dim=0) for z in tensors]  # TODO this uses double memory... perhaps we need to specify a max size?                                                     
        return TensorDataset(*tensors)      

class _WebDatasetIterable(IterableDataset, GymuShorthands, wb.Composable, wb.Shorthands):
   
    def __init__(self, iterator):
        self.iterator = iterator
    
    def __iter__(self):
        return iter(self.iterator)

@overload
def dataset(*args, **kwargs):
    pass 

@dataset.args(List)
def dataset(path: List, **kwargs):
    path = [str(p) for p in path]
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs))

@dataset.args(str)
def dataset(path : str, **kwargs):
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs))

@dataset.args(Iterable)
def dataset(iterator : Iterable, **kwargs):
    return _WebDatasetIterable(iterator, **kwargs)