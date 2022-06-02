#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 02-06-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pytorch_lightning as pl
from typing import Any

class DataModule(pl.LightningDataModule):

    def __init__(self, 
                    train : Any = None, 
                    validate : Any = None, 
                    test : Any = None, ):
        """ A simple pytorch-lightning module that may be used with webdatasets.

        Args:
            train (Any, optional): training dataset. Defaults to None.
            validate (Any, optional): validation dataset. Defaults to None.
            test (Any, optional): test dataset. Defaults to None.
        """
        super().__init__()
        self.train = train
        self.validate = validate
        self.test = test
        
    def prepare_data_train(self, *args, **kwargs):
        if hasattr(self.train, "prepare_data"):
            self.train.prepare_data(*args, **kwargs)

    def prepare_data_validate(self, *args, **kwargs):
        if hasattr(self.validate, "prepare_data"):
            self.validate.prepare_data(*args, **kwargs) 

    def prepare_data_test(self, *args, **kwargs):
        if hasattr(self.test, "prepare_data"):
            self.test.prepare_data(*args, **kwargs) 

    def prepare_data(self, *args, **kwargs):
        self.prepare_data_train(*args, **kwargs)
        self.prepare_data_validate(*args, **kwargs)
        self.prepare_data_test(*args, **kwargs)