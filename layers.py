'''
Created on 29 paź 2017

@author: Miko
'''

import tensorflow as tf

def batch_norm(input_, name = "batch_norm"):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(inputs = input_,
                                            momentum = 0.9,
                                            epsilon = 1e-5,
                                            center = True,
                                            scale = True,
                                            training = True)
        
def conv(input_, channels_in, channels_out, name = "conv"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, channels_in, channels_out], initializer = tf.truncated_normal_initializer(stddev = 0.02))
        b = tf.get_variable('b', [channels_out], initializer = tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(input_, w, strides = [1, 1, 1, 1], padding = "SAME")
        conv = conv + b
        return conv
    
def lrelu(x, leak = 0.2, name = "lrelu"):
    return tf.maximum(x, leak + x)

def linear(input_, channels_in, channels_out, name = "linear"):
    with tf.variable_scope(name):
        matrix = tf.get_variable("Matrix", [channels_in, channels_out], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [channels_out], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, matrix) + bias
