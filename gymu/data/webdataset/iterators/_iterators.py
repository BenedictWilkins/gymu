#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 20-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from typing import Dict, Any, List, Iterable, Union
import numpy as np
import io
import itertools
import more_itertools
from copy import deepcopy
from collections import defaultdict

from ....mode import STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO

__all__ = ("decode", "mode", "keep", "discard", "window", "unpack_info", "numpy", "mask", "to_dict")

def decode(source, keep_meta=False):
    def _decode_keep_meta(source):
        for data in source:
            x = {k:v for k,v in  np.load(io.BytesIO(data['npz']), allow_pickle=True).items()}
            del data['npz']
            x.update(data)
            yield x
    def _decode(source):
        for data in source:
            yield {k:v for k,v in  np.load(io.BytesIO(data['npz']), allow_pickle=True).items()}
    if keep_meta:
        yield from _decode_keep_meta(source)
    else:
        yield from _decode(source)

def mode(source : Iterable, mode , ignore_last : bool = True):

    def _mode_without_next_state(source):
        for x in source:
            x = {k:v for k,v in x.items() if k in mode.keys()}
            yield mode(**x)
    def _mode_with_next_state(source):
        x1 = next(source)
        if NEXT_STATE in x1: # next state is already present in the data
            yield mode(**{k:v for k,v in x1.items() if k in mode.keys()})
            yield from _mode_without_next_state(source)
        else: # next_state is not present, use the state from the next sample, assuming data stream is ordered!
            for x2 in source:
                done = x1.get(DONE, False) # TODO if done isnt present... give a warning? we might overlap into other episodes!
                x = {k:v for k,v in x1.items() if k in mode.keys()}
                x[NEXT_STATE] = x2[STATE] if not done else None
                x1 = x2
                if not done or not ignore_last:
                    yield mode(**x)   
    if NEXT_STATE in mode.keys():
        yield from _mode_with_next_state(source)
    else:
        yield from _mode_without_next_state(source)

def keep(source : Iterable, keys : List[Any]):
    """ Keep the specified keys, discard the rest.
    Args:
        source (Iterable): source iterable.
        keys (List[Any], optional): keys to keep.
    Yields:
        dict: dictionary containing only the specified keys and their associated values.
    """
    for x in source:
        yield {k:x[k] for k in keys} # TODO missing error handling?

def discard(source : Iterable, keys : List[Any]):
    """ Discard the specified keys, keep the rest.
    Args:
        source (Iterable): source iterable.
        keys (List[Any], optional): keys to discard.
    Yields:
        dict: dictionary containing only keys not specified in 'keys' and their associated values.
    """
    for x in source:
        yield {k:v for k,v in x.items() if k not in keys}

def mask(source : Iterable, mask : Dict[Any,Union[slice,np.ndarray]]={}):
    mask_map = defaultdict(lambda : (lambda x: x))
    # is fancy indexing being used? if so, convert element to numpy array.
    mask_map.update({k:(lambda x, m=mask[k]: np.array(x)[m] if isinstance(m, np.ndarray) else x[m]) for k in mask.keys()})
    for x in source:
        yield {k:mask_map[k](z) for k,z in x.items()}

def numpy(source : Iterable):
    for x in source:
        yield {k:np.array(v) for k,v in x.items()}

def window(source : Iterable, window_size : int = 2, default : Any = 'none'):
    """ Create a window over data in the iterable, uses `more_itertools.windowed` under the hood. 
        The resulting windows are not stacked together, this should be done in the next processing step. 
    
    Example: 
        dataset = dataset.decode().window(window_size=4)    # window dataset
        dataset = dataset.mask(state = slice(2))            # keep only the first two states
        dataset = datset.numpy()                            # stack

    Example: 
        dataset = dataset.decode().keep('state')
        >> [1,2,3,4,5,...]
        dataset.window(window_size=2)
        >> [[1,2],[2,3],[3,4],...]
        dataset.window(window_size=2, default={'state' : -1})
        >> [[-1,1],[1,2],[2,3],...]
        dataset.window(window_size=2, default='zeros')
        >> [[0,1],[1,2],[2,3],...]
        dataset.window(window_size=2, default='repeat_initial')
        >> [[1,1],[1,2],[2,3],...]
        dataset.window(window_size=2, step=2)
        >> [[0,1],[2,3],[4,5],...]
        
    Args:
        source (Iterable): source iterable.
        window_size (int, optional): size of the window. Defaults to 2.
        default (Union[str, dict], optional): default (left) value to pad, should follow the same format elements in the iterable. Options: ['none', 'repeat_initial', 'zeros_like', <dict>]. Defaults to 'none' (no left padding).
    """
    if default == 'none':
        default_l = None
    elif default == 'repeat_initial':
        default_l = lambda x : x
    elif default == 'zeros_like':
        default_l = lambda x : {k:np.zeros_like(v) for k,v in x.items()}
    #elif default == 'gaussian_noise': 
    #    default_l = lambda x: {k:np.random.normal(size=np.array(v).shape) for k,v in x.items()}
    elif isinstance(default, dict):
        def _make_callable(v):
            return (lambda *_: v) if not callable(v) else v
        default = {k:_make_callable(v) for k,v in default}
        default_l = lambda x, d=default: {k:d.get(k, (lambda *_:None))(k,v) for k,v in x.items()}
    elif callable(default):
        default_l = lambda x, d=default: d(x)
    else:
        raise ValueError(f"Invalid default value {default} specified, must be {str} or {dict}")

    def windowed(iterable, default): # construct windowed
        for ep in episode(iterable):
            if default is not None:
                first = next(ep)
                d = default(first)
                ep = itertools.chain([deepcopy(x) for x in ([d] * (window_size - 1))], [first], ep)
            
            yield from more_itertools.windowed(ep, window_size)
    for x in windowed(source,default_l):
        yield dict(zip(x[0].keys(), zip(*[z.values() for z in x]))) # group values by keys
        
def unpack_info(source : Iterable, *keys : List[str]):
    for x in source:
        info = x['info'] 
        if len(keys) > 0:
            x = dict(**x, **{k:info[k] for k in keys})
        else:
            x = dict(**x, **info)
        del x['info'] 
        yield x

def to_dict(source, *keys):
    if len(keys) == 0: # try to convert to dictionary without keys...
        for data in source:
            yield dict(**data) 
    else:
        for data in source:
            yield dict(zip(keys,data))




# utility functions

def episode(iterable):
    def whilenotdone(x, iterable):
        try: 
            yield x
            while not x.get(DONE, False):
                x = next(iterable)
                yield x
        except StopIteration:
            return
    try:
        while True:
            x = next(iterable) # stop iteration? (last done reached)
            yield whilenotdone(x, iterable)
    except StopIteration as e:
        return 