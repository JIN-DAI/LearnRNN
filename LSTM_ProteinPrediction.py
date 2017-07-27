#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: LSTM_ProteinPrediction.py
@time: 2017/7/17 23:19
@description:
"""


#%% import modules
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from RNN_Models import LSTMRNN
from Configs import ProteinConfig


def main():
    # index of angles
    indexAngle = [0,1,2] # Ramachandran angles
    # indexAngle = [3,4] # Frenet angles

    # create configuration
    conf = ProteinConfig()

    # load data from folder "data"
    srcPathName = r"data"
    srcFileTrain = os.path.join(srcPathName, r"train.dat")
    srcFileTest = os.path.join(srcPathName, r"test.dat")
    with open(srcFileTrain, 'rb') as fr:
        featuresTrain = pickle.load(fr)
        anglesTrain = pickle.load(fr)
    with open(srcFileTest, 'rb') as fr:
        featuresTest = pickle.load(fr)
        anglesTest = pickle.load(fr)

    # padding zeros for training data
    for i in range(len(featuresTrain)):
        # features of training
        curShape = np.shape(featuresTrain[i])
        featuresTrain[i] = np.append(featuresTrain[i], np.zeros([conf.MAX_STEPS-curShape[0],curShape[1]]), axis=0)
        # angles of training
        curShape = np.shape(anglesTrain[i])
        anglesTrain[i] = np.append(anglesTrain[i], np.zeros([conf.MAX_STEPS-curShape[0],curShape[1]]), axis=0)
    # convert to np array
    TrainFeature = np.array(featuresTrain)
    TrainAngle = np.array(anglesTrain)

    # padding zeros for testing data
    for i in range(len(featuresTest)):
        # features of testing
        curShape = np.shape(featuresTest[i])
        featuresTest[i] = np.append(featuresTest[i], np.zeros([conf.MAX_STEPS-curShape[0],curShape[1]]), axis=0)
        # angles of testing
        curShape = np.shape(anglesTest[i])
        anglesTest[i] = np.append(anglesTest[i], np.zeros([conf.MAX_STEPS-curShape[0],curShape[1]]), axis=0)
    # convert to np array
    TestFeature = np.array(featuresTest)
    TestAngle = np.array(anglesTest)

    # variable to record start index of batch
    BATCH_START = 0

    # placeholder for input: (batch_size, max_steps, input_size)
    xs = tf.placeholder(tf.float32, [None, conf.MAX_STEPS, conf.INPUT_SIZE], name='xs')
    # placeholder for output: (batch_size, max_steps, output_size)
    ys = tf.placeholder(tf.float32, [None, conf.MAX_STEPS, conf.OUTPUT_SIZE], name='ys')

    # create an instance of LSTMRNN
    model = LSTMRNN(xs, ys, conf)

    # create a session
    sess = tf.Session()

    # for tensorboard
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs_LSTM", sess.graph)
    # to see the graph in command line window, then type:
    #   python -m tensorflow.tensorboard --logdir=logs_Regression

    # initialze all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # open figure to plot
    plt.ion()
    plt.show()
    lines_pdb = [None]*3
    lines_pre = [None]*3
    legd = [None]*3

    # total number of epoch
    num_epoch = 1
    # total number of run
    num_run = int(num_epoch*TrainFeature.shape[0]/conf.BATCH_SIZE)
    print("Total number of runs:", num_run)

    for i in range(num_run):
        # obtain one batch
        feature = TrainFeature[BATCH_START:BATCH_START+conf.BATCH_SIZE,:,:]
        angle = TrainAngle[BATCH_START:BATCH_START+conf.BATCH_SIZE,:,indexAngle]
        # increase the start of batch by conf.BATCH_SIZE
        BATCH_START += conf.BATCH_SIZE
        if BATCH_START >= TrainFeature.shape[0]:
            BATCH_START = 0

        # create the feed_dict
        feed_dict = {xs:feature, ys:angle}

        # run one step of training
        _, cost, pred = sess.run([model.optimizer, model.cost, model.prediction], feed_dict=feed_dict)

        # plotting
        # some plot index, removable
        t = np.arange(0, conf.MAX_STEPS)
        # subplot
        for iF in range(3):
            ax = plt.subplot(3,1,iF+1)
            try:
                ax.lines.remove(lines_pdb[iF][0])
                ax.lines.remove(lines_pre[iF][0])
                ax.lines.remove(legd[iF][0])
            except Exception:
                pass
            lines_pdb[iF] = plt.plot(t, angle[0, :, iF].flatten(), 'r', label='pdb')
            lines_pre[iF] = plt.plot(t, pred[0, :, iF].flatten(), 'b--', label='prediction')
            legd[iF] = plt.legend(loc='upper right')
            plt.ylabel('output_feature_%d'%(iF+1))
            plt.xlim(0, conf.MAX_STEPS)
            plt.ylim(-4, 4)
        plt.draw()
        plt.pause(0.3)
        # print and write to log
        if i % 20 == 0:
            ## test model and print
            accuracy = sess.run(model.accuracy,feed_dict={xs:TestFeature, ys:TestAngle[:,:,indexAngle]})
            print('cost: %g ; accuracy on tests set: %g'%(cost, accuracy))
            ## record result into summary
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)


#
if __name__ == '__main__':
    main()