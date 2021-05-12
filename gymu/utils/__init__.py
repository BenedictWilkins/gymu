#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 23-04-2021 11:18:07

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

try: 
    import torch
except ImportError: # PyTorch was not found
    pass 
else: # Pytorch was found
    from . import data
    

