#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-09-2020 11:03:23

    Wrappers from gym environments.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import copy
import gym
import numpy as np

from types import SimpleNamespace

try:
    import skimage.transform
except:
    pass 

def __is_channels__(axes):
    return axes == 1 or axes == 3 or axes == 4

def __image_format__(observation_space): # guess the shape, defaults to HWC
    shape = observation_space.shape
    if len(shape) == 3:
        if __is_channels__(shape[-1]):
            return shape, "HWC"
        elif __is_channels__(shape[0]):
            return shape, "CHW"
    else:
        raise ValueError("Invalid image format: {0} for observation space".format(shape, observation_space))

def __image_dtype__(observation_space):
    dtype = observation_space.dtype
    if np.issubdtype(dtype, np.integer):
        return dtype, (0,255)
    elif np.issubdtype(dtype, np.floating):
        return dtype, (0.,1.)
    else:
        raise ValueError("Invalid image  dtype: {0} for observation space".format(dtype, observation_space))


class Float(gym.Wrapper):

    def __init__(self, env):
        super(Float, self).__init__(env)
        self.observation_space = gym.spaces.Box(np.float32(0), np.float32(1), shape=env.observation_space.shape, dtype=np.float32)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.astype(np.float32) / 255., *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.astype(np.float32) / 255.

class Integer(gym.Wrapper):

    def __init__(self, env):
        super(Integer, self).__init__(env)
        self.observation_space = gym.spaces.Box(np.uint8(0), np.uint8(255), shape=env.observation_space.shape, dtype=np.uint8)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return ((observation * 255).astype(np.uint8), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return (observation * 255).astype(np.uint8)


class CHW(gym.Wrapper):

    def __init__(self, env):
        super(CHW, self).__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        h,w,c = self.observation_space.shape
        assert c == 1 or c == 3 or c == 4 # invalid channels
        self.observation_space.shape = (c,h,w)
        self.observation_space.low = self.observation_space.low.reshape(self.observation_space.shape)
        self.observation_space.high = self.observation_space.high.reshape(self.observation_space.shape)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.transpose((2,0,1)), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.transpose((2,0,1))
        
class HWC(gym.Wrapper):

    def __init__(self, env):
        super(HWC, self).__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        c,h,w = self.observation_space.shape
        assert c == 1 or c == 3 or c == 4 # invalid channels
        self.observation_space.shape = (h,w,c)
        self.observation_space.low = self.observation_space.low.reshape(self.observation_space.shape)
        self.observation_space.high = self.observation_space.high.reshape(self.observation_space.shape)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.transpose((1,2,0)), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.transpose((1,2,0))
    

class Grey(gym.Wrapper):

    def __init__(self, env, components=(0.299, 0.587, 0.114)):
        super(Grey, self).__init__(env)
        shape, fshape = __image_format__(env.observation_space)
        c = fshape.index("C") # default HWC
        if shape[c] != 3:
            raise ValueError("Observations must be colour images for grey-scale transform, received shape: {0}".format(shape))
        shape = list(shape)
        shape[c] = 1
        dtype, (_min,_max) = __image_dtype__(env.observation_space)
        self.observation_space = gym.spaces.Box(dtype(_min), dtype(_max), shape=tuple(shape), dtype=dtype)
        
        cshape = np.array([1,1,1])
        cshape[c] = 3
        self.__c = c
        self.components = np.array(components).reshape(cshape)
        self.components = self.components / self.components.sum()

    def step(self, action):
        observation, *rest = self.env.step(action)
        observation = np.multiply(observation, self.components, out=observation, casting="unsafe").sum(axis=self.__c, keepdims=True)
        return (observation, *rest)

    def reset(self):
        observation = self.env.reset()
        observation = np.multiply(observation, self.components, out=observation, casting="unsafe").sum(axis=self.__c, keepdims=True)
        return observation


class Scale(gym.Wrapper):

    interpolation = SimpleNamespace(nearest=0, bilinear=1, biquadratic=2, bicubic=3, biquartic=4, biquintic=5)

    """ 
        Scales observations (W,H) by a specified amount.
    """

    def __init__(self, env, scale, interpolation=interpolation.nearest, as_float=False):
        super(Scale, self).__init__(env)
        if isinstance(scale, (tuple, list)):
            assert len(scale) == 2
        elif isinstance(scale, (int, float)):
            scale = (scale, scale)
        self.scale = scale
        self.interpolation = interpolation

        self.__to_dtype = None

        if issubclass(env.observation_space.dtype.type, np.integer):
            self.__to_dtype = lambda x: (x * 255.).astype(np.uint8) 
        elif issubclass(env.observation_space.dtype.type, np.floating):
            self.__to_dtype = lambda x: x.astype(np.float32)
        else:
            raise TypeError("Invalid observation dtype: {0}".format(env.observation_space.dtype))

        shape, f = __image_format__(env.observation_space)
        
        ci, hi, wi = f.index("C"), f.index("H"), f.index("W")
        shape = list(shape)
        
        shape[hi] = int(shape[hi] * scale[1])
        shape[wi] = int(shape[wi] * scale[0])

        self.observation_space = gym.spaces.Box(env.observation_space.low.flat[0], env.observation_space.high.flat[0], 
                                                shape=shape, dtype=env.observation_space.dtype)
        
    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action, *args, **kwargs)
        observation = skimage.transform.resize(observation, self.observation_space.shape, order=self.interpolation,  preserve_range=True).astype(self.observation_space.dtype)
        #observation = self.__to_dtype(observation)
        return (observation, *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        observation =  skimage.transform.resize(observation, self.observation_space.shape, order=self.interpolation, preserve_range=True).astype(self.observation_space.dtype)
        #observation = self.__to_dtype(observation)
        return observation