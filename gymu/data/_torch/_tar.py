#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import pathlib
import itertools
import tarfile
import io

from tqdm.auto import tqdm

from itertools import groupby
from more_itertools import pairwise, windowed

from ...mode import STATE, NEXT_STATE, DONE

__all__ = ("write_episode", "Composable")

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

class Composable:

    def decode():
        def _decode(source):
            for data in source:
                x = {k:v for k,v in  np.load(io.BytesIO(data['npz']), allow_pickle=True).items()}
                del data['npz']
                x.update(data)
                yield x
        return _decode

    def mode(mode, ignore_last=True): 
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
            return _mode_with_next_state
        else:
            return _mode_without_next_state

    def keep(keys): # keep only these keys
        def _keep(source):
            for x in source:
                yield {k:x[k] for k in keys}
        return _keep

    def window(window_size=2):
        def consume(iterator, n):
            next(itertools.islice(iterator, n, n), None)
        def _window(source):
            iterator = windowed(source, window_size)
            for x in iterator:
                k = x[0].keys()
                print([z['__key__'] for z in x])
                done = x[-1].get(DONE, False) # TODO if done isnt present... give a warning? we might overlap into other episodes!
                x = [z.values() for z in x]
                yield dict(zip(k, [np.stack(z) for z in zip(*x)]))
                if done: # skip the last examples and start again at the next episode.
                    consume(iterator, window_size-1)
        return _window

    def unpack_info(keys=None):
        # TODO info may contain collections of data, to work with other Composables it should be unpackaged into the data dictionary
        def _unpack_info(keys):
            pass 
        return _unpack_info 





        






