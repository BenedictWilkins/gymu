#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 17-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from ._torch import *
from ._tar import *

try:
    from ._webdataset import *
except ModuleNotFoundError as e:
    print(e)
    pass # webdataset is not installed ... thats ok
