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
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from RNN_Models import LSTMRNN, BiLSTMRNN
from Configs import RamachandranConfig, FrenetConfig
from Utils import Mod2Interval


def main():
    if True:
        # Ramachandran angles
        # index of angles
        indexAngle = [0,1,2]
        angleString = ["Phi","Psi","Omega"]
        # create configuration
        conf = RamachandranConfig()
    else:
        # Frenet angles
        indexAngle = [3,4]
        angleString = ["Kappa", "Tau"]
        # create configuration
        conf = FrenetConfig()

    # load data from folder "data"
    srcPathName = r"data"
    srcFileTrain = os.path.join(srcPathName, r"train.dat")
    srcFileTest = os.path.join(srcPathName, r"test.dat")
    with open(srcFileTrain, 'rb') as fr:
        featuresTrainAll = pickle.load(fr)
        anglesTrainAll = pickle.load(fr)
    with open(srcFileTest, 'rb') as fr:
        featuresTestAll = pickle.load(fr)
        anglesTestAll = pickle.load(fr)

    # screen data with MAX_STEPS
    # train data
    featuresTrain = []
    anglesTrain = []
    for idx in range(len(featuresTrainAll)):
        if len(featuresTrainAll[idx]) <= conf.MAX_STEPS and len(anglesTrainAll[idx]) <= conf.MAX_STEPS:
            featuresTrain.append(featuresTrainAll[idx])
            anglesTrain.append(anglesTrainAll[idx])
    print("Number of feature-angle pairs in train set: (%d, %d)" % (len(featuresTrain), len(anglesTrain)))
    # test data
    featuresTest = []
    anglesTest = []
    for idx in range(len(featuresTestAll)):
        if len(featuresTestAll[idx]) <= conf.MAX_STEPS and len(anglesTestAll[idx]) <= conf.MAX_STEPS:
            featuresTest.append(featuresTestAll[idx])
            anglesTest.append(anglesTestAll[idx])
    print("Number of feature-angle pairs in test set: (%d, %d)" % (len(featuresTest), len(anglesTest)))

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


    # placeholder for input: (batch_size, max_steps, input_size)
    xs = tf.placeholder(tf.float32, [None, conf.MAX_STEPS, conf.INPUT_SIZE], name='xs')
    # placeholder for output: (batch_size, max_steps, output_size)
    ys = tf.placeholder(tf.float32, [None, conf.MAX_STEPS, conf.OUTPUT_SIZE], name='ys')
    # placeholder for dropout
    dropout = tf.placeholder(tf.float32)

    if False:
        # create an instance of LSTMRNN
        model = LSTMRNN(xs, ys, dropout, conf)
    else:
        # create an instance of BiLSTMRNN
        model = BiLSTMRNN(xs, ys, dropout, conf)

    # create a session
    sess = tf.Session()

    # initialze all variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # create an instance of Saver
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(conf.checkpoint_dir)
    print("Checking %s ..." % (conf.checkpoint_dir))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Checkpoint found! Resume training ...")
    else:
        print("Checkpoint NOT found! Start training ...")
    #print(conf)


    # for tensorboard
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs_LSTM", sess.graph)
    # to see the graph in command line window, then type:
    #   python -m tensorflow.tensorboard --logdir=logs_Regression

    # open figure to plot
    plt.ion()
    plt.show()
    lines_pdb = [None]*len(indexAngle)
    lines_pre = [None]*len(indexAngle)
    legd = [None]*len(indexAngle)

    # variable to record start index of batch
    BATCH_START = 0

    # total number of epoch
    num_epoch = 2
    epoch_counter = 0
    # total number of run
    num_run = num_epoch*TrainFeature.shape[0]//conf.BATCH_SIZE
    print("Total number of runs: % 4d (%d epoches)" % (num_run, num_epoch))

    # run number to print
    num_print = 20
    # run number to save
    num_checkpoint = 100

    # dropout value
    dropout_train = 0.1
    dropout_test = 0.0

    for i in range(num_run):
        # obtain one batch
        feature = TrainFeature[BATCH_START:BATCH_START+conf.BATCH_SIZE,:,:]
        angle = TrainAngle[BATCH_START:BATCH_START+conf.BATCH_SIZE,:,indexAngle]

        # create the feed_dict
        feed_dict = {xs: feature, ys: angle, dropout: dropout_train}

        # print and write to log of initial step
        if i == 0:
            # validate model and print
            accTrain = sess.run(model.accuracy,
                                feed_dict={xs: TrainFeature, ys: TrainAngle[:, :, indexAngle], dropout: dropout_test})
            accTest = sess.run(model.accuracy,
                               feed_dict={xs: TestFeature, ys: TestAngle[:, :, indexAngle], dropout: dropout_test})
            print('Initial Step: accTrain = %.4f; accTest = %.4f' % (accTrain, accTest))
            # record result into summary
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

        # print number of epoch
        if BATCH_START == 0:
            epoch_counter += 1
            print('Epoch: %d' % epoch_counter)
        # increase the start of batch by conf.BATCH_SIZE
        BATCH_START += conf.BATCH_SIZE
        if BATCH_START >= TrainFeature.shape[0]:
            BATCH_START = 0

        # record start time
        start_time = time.time()
        # run one step of training
        _, cost, pred = sess.run([model.optimizer, model.cost, model.prediction], feed_dict=feed_dict)
        # calculate time duration of training step
        duration = time.time() - start_time

        # plotting
        # some plot index, removable
        t = np.arange(0, conf.MAX_STEPS)
        # subplot
        for iF in range(len(indexAngle)):
            ax = plt.subplot(len(indexAngle),1,iF+1)
            # remove previous curve
            try:
                ax.lines.remove(lines_pdb[iF][0])
                ax.lines.remove(lines_pre[iF][0])
                ax.lines.remove(legd[iF][0])
            except Exception:
                pass
            # current angle values
            y_pdb = angle[0, :, iF]
            y_pred = pred[0, :, iF].flatten()
            # modulate angles values into (-pi, pi]
            #for iV in range(len(y_pdb)):
            #    y_pdb[iV] = Mod2Interval(y_pdb[iV], np.pi)
            #    y_pred[iV] = Mod2Interval(y_pred[iV], np.pi)
            # plot angles curves
            lines_pdb[iF] = plt.plot(t, y_pdb, 'r', label='pdb')
            lines_pre[iF] = plt.plot(t, y_pred, 'b--', label='prediction')
            legd[iF] = plt.legend(loc='upper right')
            plt.ylabel(angleString[iF])
            plt.xlim(0, conf.MAX_STEPS)
            plt.ylim(-3.5, 3.5)
        # draw figure and keep for a while
        plt.draw()
        plt.pause(0.1)

        # print and write to log after training
        if (i+1) % num_print == 0:
            # validate model and print
            accTrain = sess.run(model.accuracy,
                                feed_dict={xs: TrainFeature, ys: TrainAngle[:, :, indexAngle], dropout: dropout_test})
            accTest = sess.run(model.accuracy,
                               feed_dict={xs: TestFeature, ys: TestAngle[:, :, indexAngle], dropout: dropout_test})
            print('Step% 4d(%.3f sec): cost = %.4f; accTrain = %.4f; accTest = %.4f'
                  % (i+1, duration, cost, accTrain, accTest))
            # record result into summary
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i+1)
        # save checkpoint
        if (i+1) % num_checkpoint == 0:
            save_path = saver.save(sess, conf.checkpoint_dir+r"model.ckpt")
            print("Checkpoint saved at step% 4d!" % (i+1))

    sess.close()


#
if __name__ == '__main__':
    main()