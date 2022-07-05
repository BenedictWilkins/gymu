#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 01-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from itertools import islice

import numpy as np
from typing import Union, List, Dict, Any
from .. import iterators

from webdataset import iterators as wbiterators

__all__ = ("decode", "mode", "keep", "discard", "window", "unpack_info", "to_numpy", "mask", "to_dict", "to_tuple", "map_each", "skip")

# functional composition

def skip(dataset, n):
    def _skip_iterator(source, n=n):
        yield from islice(source, n, None)
    return dataset.then(_skip_iterator, n=n)

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

def to_numpy(dataset):
    return dataset.then(iterators.to_numpy)

def mask(dataset, **mask : Union[slice, np.ndarray]):
    return dataset.then(iterators.mask, mask)

def to_dict(dataset, *keys : List[str]):
    return dataset.then(iterators.to_dict, *keys)

def to_tuple(dataset, *keys):
    return dataset.then(iterators.to_tuple, *keys)

def map_each(dataset, fun):
    return dataset.then(iterators.map_each, fun)



