#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Created on 23-04-2021 11:18:07

 [Description]
"""
__author__ ="Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ ="Development"

from . import _serialise

class overload:
    """
        Overload decorator for functions, example:
        '''
            @overload
            def f():
                pass

            @f.args(int, int)
            def f(x, y):
                print('two integers')

            @f.args(float)
            def f(x):
                print('one float')
        '''
        See https://stackoverflow.com/a/57726675/9704615 for details.
    """
    def __init__(self, f):
        self.cases = []

    def args(self, *args):
        def store_function(f):
            self.cases.append((tuple(args), f))
            return self
        return store_function

    def __call__(self, *args, **kwargs):
        for k,f in self.cases:
            if all([issubclass(type(x), k) for x in args]):
                return f(*args, **kwargs)
        return function(*args, **kwargs)

