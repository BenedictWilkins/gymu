#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from collections.abc import Iterable
from typing import List

import webdataset as wb
from torch.utils.data import IterableDataset

from ._tar import Composable
from ...utils import overload

__all__ = ("dataset",)

class _WebDatasetIterable(IterableDataset, wb.Composable, wb.Shorthands):
    def __init__(self, iterator):
        self.iterator = iterator
    def __iter__(self):
        return iter(self.iterator)


@overload
def dataset(*args, **kwargs):
    pass 

@dataset.args(List)
def dataset(path: List, **kwargs):
    return wb.WebDataset(path, **kwargs)

@dataset.args(str)
def dataset(path : str, **kwargs):
    return wb.WebDataset(path, **kwargs)

@dataset.args(Iterable)
def dataset(iterator : Iterable, **kwargs):
    return _WebDatasetIterable(iterator, **kwargs)