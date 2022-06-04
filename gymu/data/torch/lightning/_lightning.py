#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from dataclasses import dataclass, asdict
from attr import field
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from typing import Any, Iterable, Union

from ....utils import bind

__all__ = ("DataModule",)

@dataclass
class Dataset(TorchDataset):

    dataset : Union[Iterable, TorchDataset]
    batch_size : int = None
    num_workers : int = 0
    shuffle : bool = False
    prefetch_factor : int = 1
    persistent_workers : bool = False
    pin_memory : bool = False
    drop_last : bool = False

    def __post_init__(self):
        try:
            assert isinstance(len(self.dataset), int)
            setattr(self, "__len__", lambda : len(self.dataset))
        except:
            pass # this is an iterable dataset? probably...

    def __getitem__(self, indx):
        return self.dataset[indx]

    def __iter__(self):
        return iter(self.dataset)

    def dataloader(self, **kwargs):
        _kwargs = asdict(self)
        _kwargs.update(kwargs)
        if _kwargs.get('num_workers') == 0:
            _kwargs.pop('prefetch_factor', None) # prevents DataLoader error...
        return DataLoader(**_kwargs)

    def prepare_data(self, *args, **kwargs):
        if hasattr(self.dataset, "prepare_data"):
            self.dataset.prepare_data(*args, **kwargs)

class DataModule(pl.LightningDataModule):

    def __init__(self, 
                    train : Any = None, 
                    validate : Any = None, 
                    test : Any = None, **kwargs):
        """ A simple pytorch-lightning module that may be used with webdatasets.

        Args:
            train (Any, optional): training dataset. Defaults to None.
            validate (Any, optional): validation dataset. Defaults to None.
            test (Any, optional): test dataset. Defaults to None.
        """
        self.train = self._initialise(train, "train")
        self.validate = self._initialise(validate, "val")
        self.test = self._initialise(test, "test")
        super().__init__()

    def _initialise(self, dataset, name):
        if dataset is not None:
            if not hasattr(dataset, 'dataloader'):
                dataset = Dataset(dataset)
            setattr(self, f'{name}_dataloader', (lambda : getattr(self, name).dataloader()))
        return dataset

    def prepare_data_train(self, *args, **kwargs):
        if hasattr(self.train, "prepare_data"):
            self.train.prepare_data(*args, **kwargs)

    def prepare_data_validate(self, *args, **kwargs):
        if hasattr(self.validate, "prepare_data"):
            self.validate.prepare_data(*args, **kwargs) 

    def prepare_data_test(self, *args, **kwargs):
        if hasattr(self.test, "prepare_data"):
            self.test.prepare_data(*args, **kwargs) 

    def prepare_data(self, *args, **kwargs):
        self.prepare_data_train(*args, **kwargs)
        self.prepare_data_validate(*args, **kwargs)
        self.prepare_data_test(*args, **kwargs)
