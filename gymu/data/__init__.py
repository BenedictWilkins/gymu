#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 03-02-2022 16:55:56

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from ._tar import write_episode

from . import compose
from . import iterators
from ._webdataset import *

__all__ = ("dataset", "compose", "iterators")

try:
    from . import config
    __all__ = __all__ + ("config",)
except ModuleNotFoundError as e:
    pass # hydra/omegaconf is not installed



try:
   from . import torch
   __all__ = __all__ + ("torch",)
except ModuleNotFoundError as e: # torch is missing
   pass 


