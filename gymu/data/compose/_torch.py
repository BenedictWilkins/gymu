#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.multiprocessing
from ..iterators._iterators import _tuple_or_mapping_each

__all__ = ("to_tensor_dataset", "to_tensor")

def to_tensor_dataset(dataset, num_workers : int = 0, show_progress : bool = False, lazy = False): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
    lazy_dataset = LazyTensorDataset(dataset, num_workers=num_workers, show_progress=show_progress)
    if lazy:
        return lazy_dataset
    else:
        return lazy_dataset.__prepare_data__()

def to_tensor(dataset):
    def _to_tensor(x):
        return torch.from_numpy(x) # TODO add options for list data etc
    return dataset.then(_tuple_or_mapping_each, fun=_to_tensor)

class LazyTensorDataset(TensorDataset):
    
    def __init__(self, dataset, num_workers : int = 0, show_progress : bool = False):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.show_progress = show_progress
        self.__getitem__ = self._getitem_first
       
    
    def __prepare_data__(self):
        torch.multiprocessing.set_sharing_strategy('file_system')                                  
        source = DataLoader(self.dataset, batch_size=512, shuffle=False, num_workers=self.num_workers, drop_last=False)   
        source = source if not self.show_progress else tqdm(source, desc="Loading Tensor Dataset")                                                                          
        source = iter(source)
        tensors = [[x] for x in next(source)]
        for z in tensors: # validate input... 
            if not torch.is_tensor(z[0]):
                raise ValueError(f"Expected torch.Tensor but found {type(z[0])}, choose a gymu.mode or convert to tuple before converting to a TensorDataset for example: 'dataset.decode().mode(gymu.sa).to_tensor_dataset()'.")
        for x in source:
            for z,t in zip(x, tensors):
                t.append(z)
        tensors = [torch.cat(z,dim=0) for z in tensors]  # TODO this uses double memory... perhaps we need to specify a max size?                                                     
        return TensorDataset(*tensors)
    
    def prepare_data(self): # for torch lightning (or otherwise)
        self.tensors = self.__prepare_data__().tensors

    def _getitem_first(self):
        assert len(self) == 0
        self.prepare_data()
        self.__getitem__ = super().__getitem__
    
    def __len__(self):
        if len(self.tensors) == 0:
            self.prepare_data()
        return self.tensors[0].size(0)