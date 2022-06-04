#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 23-04-2021 11:18:07

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from ._serialise import *

from functools import partial 

class bind(partial):
    """ 
        An improved version of functools 'partial' which accepts Ellipsis (...) as an 'args' placeholder. 
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)