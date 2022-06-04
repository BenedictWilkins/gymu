#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from ._compose import *
from ._shorthands import *
try:
    from ._torch import *
except ModuleNotFoundError as e:
    pass # torch is not installed?

