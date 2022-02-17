#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 03-02-2022 19:02:45

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from re import L
from typing import Callable, List, Union
from tqdm.auto import tqdm

import h5py
import pathlib

import torch
import numpy as np
import gym

from torch.utils.data import Dataset, TensorDataset, ConcatDataset, DataLoader
from pytorch_lightning import LightningDataModule

from ._data import read_episodes, write_episodes

__all__ = ("GymDataModule",)

class GymDataModule(LightningDataModule):

    def __init__(self, 
                env_id : Union[str, Callable],
                policy = None,
                mode : List[str] = ['state', 'action'], 
                num_train_episodes : int = 50,
                num_validation_episodes: int = 20,
                num_test_episodes: int = 10,
                max_episode_length : int = 5000,
                batch_size : int = 256):

        super().__init__()
        self.env_id = env_id
        if isinstance(self.env_id, str):
            env = gym.make(self.env_id)
        elif callable(env_id):
            env = env_id()
        else:
            raise TypeError(f"Invalid environment {env_id}.")

        self.state_space = env.observation_space
        self.action_space = env.action_space
        try:
            env.close()
        except:
            pass 
    
        self.policy = policy
        self.mode = mode

        self.num_train_episodes = num_train_episodes
        self.num_validation_episodes = num_validation_episodes
        self.num_test_episodes = num_test_episodes
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size

    def episodes(self, n, workers=1, show_progress=False):
        iterator = iterator.episodes(self.env_id, policy=self.policy, n=n, mode=self.mode, max_length=self.max_episode_length, workers=workers)     
        iterator = tqdm(iterator, total=n) if show_progress else iterator
        episodes = [type(ep)(*[torch.from_numpy(x) for x in ep]) for ep in iterator]
        return episodes

    def write_dataset(self, path, force=False, show_progress=False, compression=None):        
        # generate a dataset from the given environment and save it to disk.
        path = pathlib.Path(path)
        assert path.suffix == ".hdf5"
        if not path.exists() or force:
            write_episodes(path, [], write_mode="w") # overwrite the original file
            # create and save one episode at a time to avoid running out of memory...
            iterator = iterator.episodes(self.env_id, policy=self.policy, n=n, mode=self.mode, max_length=self.max_episode_length, workers=workers)
            write_episodes(path, iterator, compression=compression, write_mode="w", show_progress=show_progress)
        else:
            raise FileExistsError(f"File {str(path)} already exists.")

