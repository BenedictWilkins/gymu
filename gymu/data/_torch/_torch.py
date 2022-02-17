#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 15-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch
from torch.utils.data import Dataset

__all__ = ("SubSequenceDataset",)

class SubSequenceDataset(Dataset):
    """ 
        A dataset of subsequences of a given length from a dataset of sequences. 
        Batches have the following shape: [batch_size, subsequence_length, ...]
    """
    
    def __init__(self, *tensors, subsequence_length = 16):
        super().__init__()
        self.tensors = [self.subsequence(x, subsequence_length, False) for x in tensors]
    
    def __getitem__(self, i):
        return [x[i] for x in self.tensors]
    
    def __len__(self):
        return self.tensors[0].shape[0]
        
    def subsequence(self, x, sub_sequence_length, seq_first=False): 
        # this is always a view, pretty efficient!
        x = x.unfold(0, sub_sequence_length, 1)
        p = torch.roll(torch.arange(len(x.shape)), 1)
        if not seq_first:
            p[[0,1]] = p[[1,0]]
        return x.permute(*p)


'''
class HDF5EpisodeDataset(Dataset):

    """ Contains episode data that follows the mode API"""
    
    def __init__(self, mode, **kwargs):
        self._mode_data = {}
        self._additional_data = {}

        for k,v in kwargs.items():
            if k in MODE_PROPERTIES:
                self._mode_data[MODE_PROPERTIES_DEFAULT[MODE_PROPERTIES[k]]] = v # normalise keys
            else:
                self._additional_data[k] = v 

        _len = len(self) #
        assert all([_len == len(x) for x in self._mode_data.values()])
        assert all([_len == len(x) for x in self._additional_data.values()])

        # unpack _data into mode using keys correctly, handle NEXT_STATE if not present in dataset
        if NEXT_STATE in mode.keys() and not NEXT_STATE in self._mode_data: # do some indexing, use STATE[1:] as the NEXT_STATE
            _next_state = self._mode_data[STATE][1:] # use region reference? lazy loading?
            self._mode_data = {k:v[:-1] for k,v in self._mode_data.items()}
            self._mode_data[NEXT_STATE] = _next_state
            self._additional_data = {k:v[:-1] for k,v in self._additional_data.items()}

        # TODO check info, this should also be a time-series, or a collection of time-series?
        self._mode_data = mode(**self._mode_data)
        self._additional_data = tuple([v for k,v in sorted(self._additional_data.items())])
        self._data = [*self._mode_data, *self._additional_data]

    def __getitem__(self, i):
        return tuple(x[i] for x in self._data)

    def __len__(self):
        return len(self._mode_data[STATE])

    @property
    def state(self):
        return self._mode_data.state 
    @property
    def next_state(self):
        return self._mode_data.next_state
    @property   
    def action(self):
        return self._mode_data.action
    @property
    def reward(self):
        return self._mode_data.reward
    @property
    def done(self):
        return self._mode_data.done
    @property
    def info(self):
        if INFO in self._mode_data.keys():
            return (self._mode_data.info, *self._additional_data)
        else:
            return self._additional_data
'''     