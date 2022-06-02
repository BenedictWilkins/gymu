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

try:
    from . import config
except ModuleNotFoundError as e:
    pass # hydra/omegaconf is not installed

try:
    from .webdataset import *
except ModuleNotFoundError as e:
    pass # webdataset is not installed ?

__all__ = ("config", "compose", "iterators")
