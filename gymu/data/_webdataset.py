#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import logging
Logger = logging.getLogger('gymu')

import webdataset as wb
import numpy as np
import torch
import math
import pathlib
import glob

from collections.abc import Iterable, Sequence
from typing import List, Union, Dict, Any, Callable
from plum import dispatch

from ..typing import Mode
from ..iter import Iterator
from .. import mode as m

from .compose import Shorthands


__all__ = ("dataset", "Shorthands", "Composable", "Processor")

def local_urls(path : str , recursive : bool = True, extension : str = ".tar*"):
    """ Get (local) urls that have the given extension (default .tar*).

    Args:
        path (str): 
        recursive (bool, optional): whether to search sub directories. Defaults to True.
        extension (str, optional): search for files with this extension. Defaults to ".tar*".

    Returns:
        List[str]: local file paths with the given extension.
    """
    path = str(pathlib.Path(path).expanduser().resolve())
    path += ("*" * (int(recursive) + 1)) + extension
    return [url for url in glob.glob(path, recursive=recursive)]

class Composable(wb.Composable):

    def then(self, f, *args, **kw):
        assert callable(f)
        assert "source" not in kw
        return Processor(self, f, *args, **kw)

    def compose_functional(self, *fs):
        result = self
        for f in fs:
            result = f(result)
        return result

class Processor(Shorthands, Composable, wb.Processor):
    pass

class _WebDatasetIterable(Processor): # wrapper for IterableDataset...
   
    def __init__(self, iterator, **kwargs):
       
        self.iterator = iterator
    
    def __iter__(self):
        return iter(self.iterator)

class _WebDatasetEnvironmentIterable(Processor):

    def __init__(self, env : Callable, policy : Callable, mode : Mode = m.sa, max_episode_length : int = 4096, num_episodes : int = 1, **kwargs):
        self.env = env
        self.policy = policy
        self._mode = mode
        self._max_episode_length = max(1, max_episode_length)
        self._num_episodes = max(1, num_episodes)

    def __iter__(self):
        # the process is the same for each worker if using multiprocess
        env = self.env()
        policy = self.policy() if self.policy is not None else None # TODO if this should be shared accross workers things are more complicated
        iterator = Iterator(env, policy, max_length=self._max_episode_length, mode=self._mode)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            for _ in range(self._num_episodes):
                yield from iterator
        else: 
            n = math.ceil(self._num_episodes / worker_info.num_workers)
            dif = (n * worker_info.num_workers ) - self._num_episodes 
            if worker_info.id < dif:
                n -= 1
            for _ in range(n):
                yield from iterator


# NOTE for some reason adding type hints to 'transforms' breaks plum-dispatch on importing this module... TODO bug report?

@dispatch
def dataset(path : List[str], transforms = [], **kwargs):
    path = [str(p) for p in path]
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs)).compose_functional(*transforms)

@dispatch
def dataset(path : str, transforms  = [], **kwargs):
    return _WebDatasetIterable(wb.WebDataset(path, **kwargs)).compose_functional(*transforms)

@dispatch
def dataset(iterator : Iterable, transforms = [],  **kwargs):
    return _WebDatasetIterable(iterator, **kwargs).compose_functional(*transforms)

@dispatch
def dataset(iterator : Sequence, transforms = [],  **kwargs):
    return _WebDatasetIterable(iterator, **kwargs).compose_functional(*transforms)

@dispatch
def dataset(env : Callable, policy : Union[Callable,None] = None, mode = m.sa, max_episode_length : int = 4096, num_episodes : int = 1, transforms = [], **kwargs):
    return _WebDatasetEnvironmentIterable(env, policy, mode=mode, max_episode_length=max_episode_length, num_episodes=num_episodes, **kwargs).compose_functional(*transforms)

try:
    from torch.utils.data import Dataset
    @dispatch
    def dataset(iterator : Dataset, transforms = [],  **kwargs):
        return _WebDatasetIterable(iterator, **kwargs).compose_functional(*transforms)
except ModuleNotFoundError as e:
    pass
