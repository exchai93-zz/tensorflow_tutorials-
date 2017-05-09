# mnist data hosted on Yann LeCun's website
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# placeholder is a value that we will input when we ask Tensorflow to run a computation
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros[10])

# one line to define model 
y = tf.nn.softmax(tf.matmul(x,W) + b)
