import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import numpy as np

learnRate = 0.1

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1,2], -1, 1))

y = tf.matmul(w, x_data) + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(learnRate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for steps in range(0, 2000):
        sess.run(optimizer)
        if (steps % 20 == 0) :
            print(sess.run(w), sess.run(b))
