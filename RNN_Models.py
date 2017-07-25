#!/usr/bin/env python
#coding=utf-8
"""
@version: 1.0
@author: jin.dai
@license: Apache Licence
@contact: daijing491@gmail.com
@software: PyCharm
@file: RNN_Models.py
@time: 2017/7/14 17:04
@description: LSTM RNN Model class definition
"""

import tensorflow as tf
from Utils import define_scope


# define class for LSTMRNN for variable length sequence
class LSTMRNN(object):
    # initializer
    def __init__(self, xs, ys, config):
        # model parameters
        self.max_steps = config.MAX_STEPS
        self.input_size = config.INPUT_SIZE
        self.output_size = config.OUTPUT_SIZE
        self.cell_size = config.CELL_SIZE
        self.learning_rate = config.LR
        # model interfaces for inputs
        with tf.name_scope('inputs'):
            # placeholder for input: (batch_size, max_steps, input_size)
            self.xs = xs
            # placeholder for output: (batch_size, max_steps, output_size)
            self.ys = ys
        # model behaviors
        self.prediction
        self.cost
        self.optimizer

    @define_scope
    def prediction(self):
        # ------------------------------------------------------
        # create basic LSTM cell --> http://arxiv.org/abs/1409.2329
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # initial zero state for LSTM
        cell_init_state = lstm_cell.zero_state(tf.shape(self.xs)[0], dtype=tf.float32)
        # creates a recurrent neural network specified by RNNCell --> https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.xs, initial_state=cell_init_state, time_major=False, sequence_length=LSTMRNN.length(self.xs))
        # ------------------------------------------------------
        # (batch_size, max_steps, cell_size) ==> (batch_size*max_steps, cell_size)
        l_out_x = tf.reshape(cell_outputs, [-1, self.cell_size], name='2_2D')
        # Ws (cell_size, out_size)
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        # bs (out_size,)
        bs_out = self._bias_variable([self.output_size, ])
        # (batch*max_steps, out_size)
        pred_2D = tf.matmul(l_out_x, Ws_out) + bs_out
        # reshape prediction value to (batch_size, max_steps, out_size)
        pred_3D = tf.reshape(pred_2D, [-1, self.max_steps, self.output_size], name='pred_3D')
        return pred_3D

    # cost function
    # compute cost function by the actual length
    @define_scope
    def cost(self):
        # compute losses for each step
        # loss on each step (batch_size, max_steps)
        losses = tf.reduce_sum(self.ms_error(pred_3D, self.ys), reduction_indices=2)
        # mask of padding part
        mask = tf.sign(tf.reduce_max(tf.abs(self.ys), reduction_indices=2))
        # mean losses of each sequence: average along steps
        # shape: (batch_size,)
        losses = tf.div(tf.reduce_sum(losses * mask, reduction_indices=1),
                        # total loss of each sequence by the actual length
                        tf.reduce_sum(mask, reduction_indices=1),  # divide with actual length
                        name='average_losses_per_seq')
        # average loses over batch
        cost_mean = tf.reduce_mean(losses, name='average_cost_per_batch')
        # record cost into summary
        tf.summary.scalar('cost_by_length', cost_mean)
        return cost_mean

    # train optimizer
    @define_scope
    def optimizer(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    # mean squared error
    def ms_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_target, y_pre))

    # weights: initialized with normal distribution
    def _weight_variable(self, shape, name='weights'):
        # initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        initializer = tf.contrib.layers.xavier_initializer()
        # details of get_variable() --> https://www.tensorflow.org/programmers_guide/variable_scope
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    # biases: initialized with small positive constant
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    # length of sequence in the same batch
    # assume that the sequences are padded with zero vectors to fill up the remaining time steps in the batch
    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))  # take sign of max abs along feature axis
        length = tf.reduce_sum(used, reduction_indices=1)  # sum along time axis
        length = tf.cast(length, tf.int32)  # data type -> int32
        return length