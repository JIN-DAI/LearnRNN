#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: CreateTestData.py
@time: 2017/7/14 16:44
@description: function to create test sequence data for training and testing
"""

import numpy as np
import matplotlib.pyplot as plt

# function to create test input and output sequence with variable length
def get_batch(timeSteps, batchSize, batchSTART, plotSwitch=False):
    # xs shape (batchSize, timeSteps)
    xs = np.arange(batchSTART, batchSTART+timeSteps*batchSize).reshape((batchSize, timeSteps)) / 10 * np.pi
    xs = xs[:,:,np.newaxis]

    # input (batchSize, timeSteps, input_size)
    seq = np.append(np.sin(xs), np.cos(xs), axis=2)
    # output (batchSize, timeSteps, output_size)
    res = np.append(np.sin(3*xs)**2+3*np.cos(xs+np.pi/6.0)**3, np.tanh(3*np.sin(xs)+np.cos(xs+np.pi/3)), axis=2)

    # plot res and seq in the first batch
    if plotSwitch:
        plt.subplot(211)
        plt.plot(xs[0,:], seq[0,:,0], 'r', xs[0,:], seq[0,:,1], 'b--')
        plt.ylabel('input')
        plt.subplot(212)
        plt.plot(xs[0,:], res[0,:,0], 'r', xs[0,:], res[0,:,1], 'b--')
        plt.ylabel('output')
        plt.show()
    # return seq (batchSize, timeSteps, input_size)
    # return res (batchSize, timeSteps, output_size)
    # return xs  (batchSize, timeSteps)
    return [seq, res, xs]