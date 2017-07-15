# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 23:01:05 2017

@author: JIN-DAI
"""


#%% import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CreateTestData import get_batch
from RNN_Models import LSTMRNN
from Configs import Config

BATCH_START = 0

def main():
    # create configuration
    conf = Config()
    # global variable decleration
    global BATCH_START

    # create an instance of LSTMRNN
    model = LSTMRNN(conf)

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

    # total number of runs
    num_run = 100
    # number of time steps in each run
    steps = np.random.randint(conf.MAX_STEPS // 3, conf.MAX_STEPS + 1, num_run)

    for i in range(num_run):
        # obtain one batch
        seq, res, xs = get_batch(steps[i], conf.BATCH_SIZE, BATCH_START)
        # increase the start of batch by timeSteps
        BATCH_START += steps[i]
        # padding to max_steps
        seq_padding = np.append(seq, np.zeros([conf.BATCH_SIZE, conf.MAX_STEPS - steps[i], conf.INPUT_SIZE]), axis=1)
        res_padding = np.append(res, np.zeros([conf.BATCH_SIZE, conf.MAX_STEPS - steps[i], conf.OUTPUT_SIZE]), axis=1)

        # create the feed_dict
        feed_dict = {
            model.xs: seq_padding,
            model.ys: res_padding,
            model.learning_rate: conf.LR
        }

        # run one step of training
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        # plotting
        plt.subplot(211)
        plt.plot(xs[0, :], res[0, :, 0].flatten(), 'r', xs[0, :], pred[:, 0].flatten()[:steps[i]], 'b--')
        plt.ylim((-4, 4))
        plt.ylabel('output_feature_1')
        plt.subplot(212)
        plt.plot(xs[0, :], res[0, :, 1].flatten(), 'r', xs[0, :], pred[:, 1].flatten()[:steps[i]], 'b--')
        plt.ylim((-2, 2))
        plt.ylabel('output_feature_2')
        plt.draw()
        plt.pause(0.3)
        # write to log
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

    ## test model
    test_seq, test_res, test_xs = get_batch(200, conf.BATCH_SIZE, BATCH_START)
    test_seq = test_seq[0:1, :]
    test_res = test_res[0:1, :]
    test_xs = test_xs[0, :]
    test_pred = sess.run(model.pred, feed_dict={model.xs: test_seq, model.ys: test_res})
    test_accuracy = np.mean(np.square(test_res[0, :, :] - test_pred), axis=0)
    print(test_accuracy)


#
if __name__ == '__main__':
    main()