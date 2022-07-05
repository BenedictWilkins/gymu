#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import itertools
import numpy as np
from typing import Callable, List, Dict, Union, Any

from . import _compose
from . import _torch
from ...utils import logger, DEPRECATED

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
    
    @DEPRECATED("Shorthand 'numpy' is deprecated and will be removed in future versions, please use 'to_numpy' instead.")
    def numpy(self): # DEPRECATED
        return _compose.to_numpy(self)

    def to_numpy(self):
        return _compose.to_numpy(self)

    def mask(self, **mask : Union[int, slice, np.ndarray]):
        return _compose.mask(self, mask)

    def to_dict(self, *keys : List[str]):
        return _compose.to_dict(self, *keys)

    def map_each(self, fun : Callable):
        return _compose.map_each(self, fun)

    def skip(self, n):
        return _compose.skip(self, n)

    def to_tensor(self):
        return _torch.to_tensor(self)

    # TODO if torch is not installs things will break...
    def to_tensor_dataset(self, num_workers : int = 0, show_progress : bool = False): # WARNING YOU MIGHT RUN OUT OF MEMORY ;)
        return _torch.to_tensor_dataset(self, num_workers=num_workers, show_progress=show_progress)
