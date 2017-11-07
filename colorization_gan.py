'''
Created on 28 paź 2017

@author: Miko
'''

import tensorflow as tf
from GAN import GAN

tf.app.flags.DEFINE_integer("image_size", 32, "The size of the output images to produce [32]")
tf.app.flags.DEFINE_integer("batch_size", 32, "The size of batch images")

FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)


with tf.Session(config = config) as sess:
    gan_model = GAN(sess, config = FLAGS)
    
    
    
    gan_model.train(FLAGS)
        
