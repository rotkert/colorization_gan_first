# -*- coding: utf-8 -*-

'''
Created on 28 paź 2017

@author: Miko
'''

import os
import tensorflow as tf
import datetime
from GAN import GAN

tf.app.flags.DEFINE_integer("image_size", 32, "The size of the output images to produce [32]")
tf.app.flags.DEFINE_integer("batch_size", 8, "The size of batch images")
tf.app.flags.DEFINE_string("result_dir", "/home/mkamins3/results", "Directory to save results")
tf.app.flags.DEFINE_integer("disc_step", 5, "Steps of discriminator in one iteration")
tf.app.flags.DEFINE_integer("gen_step", 1, "Steps of generator in one iteration")
tf.app.flags.DEFINE_integer("iterations", 2500, "Iterations of disc-gen steps")
tf.app.flags.DEFINE_integer("save_samples_interval", 20, "Interval of saving samples")
tf.app.flags.DEFINE_integer("save_model_interval", 50, "Interval of saving model")
tf.app.flags.DEFINE_string("run_dir", "", "run dir")

FLAGS = tf.app.flags.FLAGS

now = datetime.datetime.now()
FLAGS.run_dir = os.path.join(FLAGS.result_dir, "run-" + now.strftime("%Y-%m-%d_%H%M"))

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

if not os.path.exists(FLAGS.run_dir):
        os.makedirs(FLAGS.run_dir)


with tf.Session(config = config) as sess:
    gan_model = GAN(sess, config = FLAGS)
    gan_model.train(FLAGS)
        
