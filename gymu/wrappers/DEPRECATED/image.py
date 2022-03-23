#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 17-02-2021 16:11:54

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

import numpy as np
import skimage.transform

from .np_wrap import NumpyWrapper, _wrap
from ..spaces import NumpyBox

class image(NumpyWrapper):

    def __init__(self, env):
        super(image, self).__init__(env)
        self._wrap_type = self.__class__.__name__.capitalize()

    @_wrap
    def grey(self, components=(0.299, 0.587, 0.114)):
        wrap = image(self)

        try:
            ci = wrap.observation_space.shape.index(3) # find channel index
        except:
            raise ValueError("States must be (3-channel) coloured images, incorrect state shape: {0}".format(self.observation_space.shape))
        
        components = np.array(components, dtype=np.float32)
        shape = [1] * len(wrap.observation_space.shape)
        shape[ci] = components.shape[0]
        components = components.reshape(shape)

        dtype = wrap.observation_space.dtype

        wrap._transform = lambda x: (x * components).sum(axis=ci, keepdims=True)
        wrap.observation_space = wrap._transform(wrap.observation_space)
        
        return wrap

    @_wrap
    def CHW(self): # assumes HWC format
        wrap = self.transpose((2,0,1))
        wrap.__class__ = image # hackz
        return wrap 

    @_wrap
    def HWC(self): # assumes CHW format
        wrap = self.transpose((1,2,0))
        wrap.__class__ = image # hackz
        return wrap 

    @_wrap
    def resize(self, *shape, interpolation=0):
        
        wrap = image(self)
        wrap._transform = lambda x: skimage.transform.resize(x, shape, order=interpolation, preserve_range=True)

        low = skimage.transform.resize(wrap.observation_space.low, shape, order=interpolation,  preserve_range=True).astype(np.float32)
        high = skimage.transform.resize(wrap.observation_space.high, shape, order=interpolation,  preserve_range=True).astype(np.float32)

        wrap.observation_space = NumpyBox(low, high)
        return wrap