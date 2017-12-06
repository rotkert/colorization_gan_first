'''
Created on 28 paź 2017

@author: Miko
'''

import os
import tensorflow as tf
import layers
import numpy as np
import data_provider as dp
from layers import linear, batch_norm
import matplotlib.pyplot as plt
from skimage.color.colorconv import yuv2rgb
from scipy import misc

class GAN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.build_model(config)
        
    def build_model(self, config):
        self.z = tf.placeholder(tf.float32, [self.config.batch_size, 100], name = "z")
        self.images_YUV = tf.placeholder(tf.float32, shape = [None, config.image_size, config.image_size, 3], name = "real_images")
        
        self.images_Y, self.images_U, self.images_V = tf.split(self.images_YUV, 3, 3)
        
        # generator
        self.generated_images_UV = self.generator(self.z, self.images_Y, config)
        self.generated_images_YUV = tf.concat([self.images_Y, self.generated_images_UV], 3)
        
        # discriminator
        self.logits_real = self.discriminator(self.images_YUV, config = config)
        self.logits_generated = self.discriminator(self.generated_images_YUV, reuse = True, config = config) 
        
        self.d_loss = - tf.reduce_mean(self.logits_real - self.logits_generated)
        self.g_loss = -tf.reduce_mean(self.logits_generated)
        
        tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("g_loss", self.g_loss)
        
        self.total_loss = self.d_loss + self.g_loss
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()
        
    def generator(self, z, image_Y, config):
        with tf.variable_scope("generator"):
            h0 = linear(z, 100, config.image_size * config.image_size, name = "g_h0_lin")
            h0 = tf.reshape(h0, [-1, config.image_size, config.image_size, 1])
            h0 = tf.nn.relu(batch_norm(h0, name = "g_bn0"))
            
            h1 = tf.concat([image_Y, h0], 3)
            h1 = layers.conv(h1, 2, 128, name = "g_h1_conv")
            h1 = tf.nn.relu(batch_norm(h1, name = "g_bn1"))
            
            h2 = tf.concat([image_Y, h1], 3)
            h2 = layers.conv(h2, 129, 64, name = "g_h2_conv")
            h2 = tf.nn.relu(batch_norm(h2, name = "g_bn2"))
            
            h3 = tf.concat([image_Y, h2], 3)
            h3 = layers.conv(h3, 65, 64, name = "g_h3_conv")
            h3 = tf.nn.relu(batch_norm(h3, name = "g_bn3"))
            
            h4 = tf.concat([image_Y, h3], 3)
            h4 = layers.conv(h4, 65, 64, name = "g_h4_conv")
            h4 = tf.nn.relu(batch_norm(h4, name = "g_bn4"))
            
            h5 = tf.concat([image_Y, h4], 3)
            h5 = layers.conv(h5, 65, 32, name = "g_h5_conv")
            h5 = tf.nn.relu(batch_norm(h5, name = "g_bn5"))
            
            h6 = tf.concat([image_Y, h5], 3)
            h6 = layers.conv(h6, 33, 2, name = "g_h6_conv")
            return tf.nn.tanh(h6)
        
    def discriminator(self, image, reuse = False, config = None):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            h0 = layers.lrelu(layers.conv(image, 3, 64, name='d_h0_conv'))
            h1 = layers.lrelu(batch_norm(layers.conv(h0, 64, 128, name='d_h1_conv'), name='d_bn1'))
            h2 = layers.lrelu(batch_norm(layers.conv(h1, 128, 256, name='d_h2_conv'), name='d_bn2'))
            h3 = layers.lrelu(batch_norm(layers.conv(h2, 256, 512, name='d_h3_conv'), name='d_bn3'))

            h4 = linear(tf.reshape(h3, [config.batch_size, -1]), 524288, 64, name = "d_h4_lin")
            h5 = linear(h4, 64, 1, name = "d_h5_lin")
            return h5
        
    def save_images(self, images, size, image_path, color_space = "RGB"):
        merged_image = self.merge(images, size)
        return misc.imsave(image_path, merged_image)
 
    def merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img
    
    def save_model(self, config, step):
        model_name = "WGAN.model"
        model_dir = os.path.join(config.run_dir, "models")
 
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
 
        self.saver.save(self.sess, os.path.join(model_dir, model_name), global_step=step)
                
    def train(self, config):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(self.d_loss, var_list = self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.5).minimize(self.g_loss, var_list = self.g_vars)
        
        clip_ops = []
        for var in self.d_vars:
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)))
            
        clip_d_vars_op = tf.group(*clip_ops)
        
        tf.global_variables_initializer().run()
        
        data_provider = dp.DataProvider(config)
        
        sample_images = data_provider.load_sample()
        sample_z = np.random.uniform(-1, 1, size=(1, config.batch_size, 100))
        
        tf.summary.image("sample_org", yuv2rgb(sample_images), 8)
        self.save_images(yuv2rgb(sample_images), [1, config.batch_size], os.path.join(config.run_dir, "org.png"))
        
        writer = tf.summary.FileWriter(config.run_dir)
        writer.add_graph(self.sess.graph)
        
        counter = 0
        while counter < config.iterations:
            print(counter)
            for k_d in range(0, config.disc_step):
                print(k_d)
                batch_images = data_provider.load_data(config)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, 100]).astype(np.float32)
                _, _g_loss, _d_loss, _loss = self.sess.run([d_optim, self.g_loss, self.d_loss, self.total_loss],
                        feed_dict = {self.z: batch_z, self.images_YUV: batch_images})
                self.sess.run([clip_d_vars_op], feed_dict={})
                 
            for k_g in range(0, config.gen_step):
                batch_images = data_provider.load_data(config)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, 100]).astype(np.float32)
                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.images_YUV: batch_images})
                 
            if counter % config.save_samples_interval == 0:
                _generate_image, _g_loss, _d_loss, _loss = self.sess.run([self.generated_images_YUV, self.g_loss, self.d_loss, self.total_loss], feed_dict={self.z: sample_z[0], self.images_YUV: sample_images})
                _generate_image_rgb = yuv2rgb(_generate_image)
                tf.summary.image("sample_gen", _generate_image_rgb, 8)
                summ = tf.summary.merge_all()
                [s] = self.sess.run([summ], feed_dict={self.z: sample_z[0], self.images_YUV: sample_images})
                writer.add_summary(s, counter)
                self.save_images(_generate_image_rgb, [1, config.batch_size], os.path.join(config.run_dir, "step" + str(counter) + ".png"))
                
            if counter % config.save_model_interval == 0:
                self.save_model(config, counter)
                
            counter += 1
            
 
        