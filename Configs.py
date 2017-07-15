#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: Configs.py.py
@time: 2017/7/15 11:33
@description: class definition of configuration parameters
"""

class Config:
    # Hyperparameters
    BATCH_SIZE = 50  ## number of proteins
    LR = 0.006  ## learning rate
    # Network Parameters
    MAX_STEPS = 200  ## number of residues
    INPUT_SIZE = 2   ## number of features for input, such as MSA information ...
    OUTPUT_SIZE = 2  ## number of labels for output, such as Ramachandran angles ...
    CELL_SIZE = 64   ## size of cell