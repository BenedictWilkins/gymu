#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-02-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import pathlib
import tarfile
from tqdm.auto import tqdm

__all__ = ("write_episode",)

def write_episode(iterator, path="./episode/", compression="gz"):
    """
        Write iterator data to a tar archive. May be used with the standard gymu iterator to create a dataset on disk. 

        Example:
        '''
            env = gymu.make("SpaceInvaders-v0")
            iterator = gymu.iterator(env, mode=gymu.mode.sar)
            gymu.data.write_episode(iterator)
        '''
    """
    if compression is None:
        ext, write = '.tar', 'w'
    else:
        ext, write = f".tar.{compression}", f"w:{compression}"

    path = pathlib.Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    path_tar = pathlib.Path(path, f"../{path.stem}{ext}").resolve()
    with tarfile.open(str(path_tar), write) as tar:
        for i, x in enumerate(tqdm(iterator)):
            file = pathlib.Path(path, str(i).zfill(8) + ".npz")
            for k, v in dict(x).items():        
                if v is None:
                    del x[k]
            np.savez(file, **x)
            tar.add(str(file), arcname=file.name)
            file.unlink() # delete file
    path.rmdir()



        






