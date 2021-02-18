#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 18-02-2021 09:50:57

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import torch
from torch.utils.data import IterableDataset, TensorDataset, ConcatDataset

from .. import mode 
from .base import iterator, episode

import tqdm

class GymDataset(IterableDataset):

    def __init__(self, env, policy=None, mode=mode.s, **kwargs):
        self.env = env
        self.policy = policy
        self.mode = mode

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        def _iter():
            while True:
                for data in iterator(self.env, policy=self.policy, mode=self.mode):
                    yield data.tuple()

        if worker_info is None: # single worker
            yield from _iter()
        else: # in worker process
            # it uses multiprocess which sucks, so probably wont be implemeting this, use ray instead!!!
            raise NotImplementedError("TODO? (see source)") 


class EpisodeDataset(ConcatDataset):

    def __init__(self, env, policy, mode=mode.s, n=1, max_length=1000, progress=True):
        if progress:
            _iter = tqdm.tqdm(range(n), "Collecting episodes")
        else:
            _iter = range(n)

        def _episode():
            ep = episode(env, policy, mode=mode, max_length=max_length)
            ep = [torch.from_numpy(e) for e in ep]
            return TensorDataset(*ep) 

        data = [_episode() for _ in _iter]
        super(EpisodeDataset, self).__init__(data)

        

