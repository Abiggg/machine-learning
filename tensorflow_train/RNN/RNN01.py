import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

INPUT_NUM = 28
STEP_NUM = 28
CLASSIFY_NUM = 10
HIDDEN_LAYER_NUM = 128
BATCHSIZE_NUM = 100
LEARN_RATE = 0.001

x = tf.placeholder(tf.float32,[None, INPUT_NUM, STEP_NUM])
y = tf.placeholder(tf.float32, [None, CLASSIFY_NUM])

weight = {
    'in':tf.Variable(tf.random_normal([INPUT_NUM, HIDDEN_LAYER_NUM])),
    'out': tf.Variable(tf.random_normal([HIDDEN_LAYER_NUM, CLASSIFY_NUM]))
}

bias = {
    'in': tf.Variable(tf.random_normal([HIDDEN_LAYER_NUM,])),
    'out': tf.Variable(tf.random_normal([CLASSIFY_NUM,]))
}

def RNN(x_orign, weight, bias):
    x = tf.reshape(x_orign, [BATCHSIZE_NUM * STEP_NUM, INPUT_NUM])
    x_in = tf.matmul(x, weight['in']) + bias['in']
    x_in = tf.reshape(x_in, [-1, STEP_NUM, HIDDEN_LAYER_NUM])
    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_LAYER_NUM)
    _init_state = cell.zero_state(BATCHSIZE_NUM, tf.float32)
    output, states = tf.nn.dynamic_rnn(cell, x_in, initial_state = _init_state, time_major=False)
    result = tf.matmul(states[1], weight['out']) + bias['out']
    return result

pred = RNN(x, weight, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, pred))
optimizer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)

correct = tf.equal(tf.arg_max(y, 1), tf.arg_max(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * BATCHSIZE_NUM < 100000:
        batch_xs, batch_ys = mnist.train.next_batch(BATCHSIZE_NUM)
        batch_xs = batch_xs.reshape([BATCHSIZE_NUM, INPUT_NUM, STEP_NUM])
        sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step = step + 1

