#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 20-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import io
import itertools
import more_itertools

from ...mode import STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO

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

def mode(source, mode, ignore_last=True): 
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

def keep(source, keys=[STATE, NEXT_STATE, ACTION, REWARD, DONE, INFO]): # keep only these keys
    for x in source:
        yield {k:x[k] for k in keys}

def window(source, window_size=2):
    def consume(iterator, n):
        next(itertools.islice(iterator, n, n), None)
    iterator = more_itertools.windowed(source, window_size)
    for x in iterator:
        k = x[0].keys()
        print([z['__key__'] for z in x])
        done = x[-1].get(DONE, False) # TODO if done isnt present... give a warning? we might overlap into other episodes!
        x = [z.values() for z in x]
        yield dict(zip(k, [np.stack(z) for z in zip(*x)]))
        if done: # skip the last examples and start again at the next episode.
            consume(iterator, window_size-1)
 
def unpack_info(source, keys=None):
    # TODO info may contain collections of data, to work with other Composables it should be unpackaged into the data dictionary
    raise NotImplementedError()