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


# length of sequence in the same batch
# assume that the sequences are padded with zero vectors to fill up the remaining time steps in the batch
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))  # take sign of max abs along feature axis
    length = tf.reduce_sum(used, reduction_indices=1)  # sum along time axis
    length = tf.cast(length, tf.int32)  # data type -> int32
    return length


# define class for LSTMRNN for variable length sequence
class LSTMRNN(object):
    # initializer
    def __init__(self, config):
        self.max_steps = config.MAX_STEPS
        self.input_size = config.INPUT_SIZE
        self.output_size = config.OUTPUT_SIZE
        self.cell_size = config.CELL_SIZE
        # placeholders for inputs
        with tf.name_scope('inputs'):
            # placeholder for input: (batch_size, max_steps, input_size)
            self.xs = tf.placeholder(tf.float32, [None, self.max_steps, self.input_size], name='xs')
            # placeholder for output: (batch_size, max_steps, output_size)
            self.ys = tf.placeholder(tf.float32, [None, self.max_steps, self.output_size], name='ys')
            # placeholder for learing rate
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # shape in placeholder?
        # variables for input hidden layer
        # tf.variable_scope('in_hidden'):
        #    self.add_input_layer()
        # variables for LSTM cell # NOTE: modified into multilayers!
        with tf.variable_scope('LSTM_layer'):
            self.add_lstm_layer()
        # variables for output hidden layer
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        # cost function
        with tf.name_scope('cost'):
            self.compute_cost_length()  # self.compute_cost()
        # train optimizer
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    # add one input hidden layer
    def add_input_layer(self):
        # (batch_size, max_steps, in_size) ==> (batch_size*max_steps, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size,)
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y (batch_size*max_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch_size, max_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.max_steps, self.cell_size], name='2_3D')

    # add one LSTM-RNN cell: (batch_size, max_steps, cell_size) -> (batch_size, max_steps, cell_size)
    def add_lstm_layer(self):
        # create basic LSTM cell --> http://arxiv.org/abs/1409.2329
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # create initial state as tuple:(batch_size, cell_size, ?)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(tf.shape(self.xs)[0], dtype=tf.float32)
        # creates a recurrent neural network specified by RNNCell --> https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        # self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
        #        lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False, sequence_length=length(self.xs))
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.xs, initial_state=self.cell_init_state, time_major=False, sequence_length=length(self.xs))

    # add one output hidden layer
    def add_output_layer(self):
        # (batch_size, max_steps, cell_size) ==> (batch_size*max_steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        # Ws (cell_size, out_size)
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        # bs (out_size,)
        bs_out = self._bias_variable([self.output_size, ])
        # (batch*max_steps, out_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    # compute cost function by the actual length
    def compute_cost_length(self):
        # reshape prediction value to (batch_size, max_steps, out_size)
        pred_3D = tf.reshape(self.pred, [-1, self.max_steps, self.output_size], name='pred_3D')
        # compute losses for each step
        with tf.name_scope('losses'):
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
        with tf.name_scope('average_cost'):
            self.cost = tf.reduce_mean(losses, name='average_cost_per_batch')
        # record cost into summary
        tf.summary.scalar('cost_by_length', self.cost)

    # compute cost function
    def compute_cost(self):
        # loss on each step (batch_size*max_steps)
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([tf.shape(self.xs)[0] * self.max_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses')
        # average loses over batch
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                tf.shape(self.xs)[0],
                name='average_cost')
        # record cost into summary
        tf.summary.scalar('cost', self.cost)

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

