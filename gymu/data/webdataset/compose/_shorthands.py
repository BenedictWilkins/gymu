#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
from typing import List, Dict, Union, Any

from . import _compose
from . import _torch

__all__ = ("Shorthands",)

class Shorthands:

    # access like dataset.gymu.decode
    def decode(self, keep_meta=False):
        return _compose.decode(self, keep_meta=keep_meta)
    
    def mode(self, mode, ignore_last=True):
        return _compose.mode(self, mode, ignore_last=ignore_last)

    def keep(self, *keys):
        return _compose.keep(self, keys=keys)

    def discard(self, *keys):
        return _compose.discard(self, keys=keys)
    
    def window(self, window_size=2, **kwargs):
        return _compose.window(self, window_size=window_size, **kwargs)

    def unpack_info(self, *keys : List[str]):
        return _compose.unpack_info(self, *keys)
    
    def numpy(self):
        return _compose.numpy(self)

    def mask(self, **mask : Union[int, slice, np.ndarray]):
        return _compose.mask(self, mask)

    def to_dict(self, *keys : List[str]):
        return _compose.to_dict(self, *keys)

    # TODO if torch is not installs things will break...
    def to_tensor_dataset(self, num_workers : int = 0, show_progress : bool = False, order : List[str] = None): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
        return _torch.to_tensor_dataset(self, num_workers=num_workers, show_progress=show_progress, order=order)
