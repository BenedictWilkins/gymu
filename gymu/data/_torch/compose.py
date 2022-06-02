#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 01-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
from typing import Union, List, Dict, Any
from . import iterators

from webdataset import iterators as wbiterators

def decode(dataset, keep_meta=False):
    return dataset.then(iterators.decode, keep_meta=keep_meta)

def mode(dataset, mode, ignore_last=True):
    return dataset.then(iterators.mode, mode, ignore_last=ignore_last)

def keep(dataset, *keys): #[STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO]
    return dataset.then(iterators.keep, keys=keys)

def discard(dataset, *keys):
    return dataset.then(iterators.discard, keys=keys)

def window(dataset, window_size=2, **kwargs):
    return dataset.then(iterators.window, window_size=window_size, **kwargs)

def unpack_info(dataset, *keys : List[str]):
    return dataset.then(iterators.unpack_info, *keys)

def numpy(dataset):
    return dataset.then(iterators.numpy)

def mask(dataset, **mask : Union[slice, np.ndarray]):
    return dataset.then(iterators.mask, mask)

def to_dict(dataset, *keys : List[str]):
    return dataset.then(iterators.to_dict, *keys)

def to_tensor_dataset(dataset, num_workers : int = 0, show_progress : bool = False, order : List[str] = None): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')                                  
    source = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=num_workers, drop_last=False)   
    source = source if not show_progress else tqdm(source, desc="Loading Tensor Dataset")                                                                          
    source = iter(source)
    tensors = [[x] for x in next(source)]
    #print([(type(x[0])) for x in tensors])
    for z in tensors: # validate input... 
        if not torch.is_tensor(z[0]):
            raise ValueError(f"Expected torch.Tensor but found {type(z[0])}, choose a gymu.mode or convert to tuple before converting to a TensorDataset for example: 'dataset.decode().mode(gymu.sa).to_tensor_dataset()'.")
    for x in source:
        for z,t in zip(x, tensors):
            t.append(z)
    tensors = [torch.cat(z,dim=0) for z in tensors]  # TODO this uses double memory... perhaps we need to specify a max size?                                                     
    return TensorDataset(*tensors) 


# webdataset
def to_tuple(dataset, *keys):
    return dataset.then(wbiterators.to_tuple, *keys)


