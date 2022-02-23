#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 08-09-2020 16:21:47

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"



from setuptools import setup, find_packages

setup(name='gymu',
      version='0.0.1',
      description='Useful utilities for OpenAI gym.',
      url='https://github.com/BenedictWilkinsAI/gymu',
      author='Benedict Wilkins',
      author_email='brjw@hotmail.co.uk',
      license='GNU3',
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      install_requires=['gym', 
                        'numpy', 
                        'scikit-image',
                        'ray[tune]', 
                        'h5py',
                        'more_itertools'],
      include_package_data=True,
      packages=find_packages())
