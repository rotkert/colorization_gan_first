'''
Created on 29 paź 2017

@author: Miko
'''

import numpy as np
import pickle
from skimage.color.colorconv import rgb2yuv

class DataProvider(object):
    
    def __init__(self, config):
        with open('F:\\magisterka\\datasets\\cifar-10-python\\cifar-10-batches-py\\data_batch_1', 'rb') as fo:
            raw = pickle.load(fo, encoding = 'bytes')
        
        raw_float = np.array(raw[b'data'], dtype = float) / 255.0
        self.data = raw_float.reshape([-1, 3, 32, 32])
        self.data = self.data.transpose([0, 2, 3, 1])
        self.data = rgb2yuv(self.data)
        
        self.len = len(self.data)
        self.batch_idxs = self.len / config.batch_size
        self.batch_idx = 0
        self.epoch_idx = 0
        
    def load_data(self, config, init = False):
        if init:
            np.random.shuffle(self.data)
            self.batch_idx = 0
            self.epoch_idx = 0
            
        batch_images = self.data[self.batch_idx * config.batch_size : (self.batch_idx + 1) * config.batch_size]
        self.batch_idx += 1
        
        if self.batch_idx >= self.batch_idxs:
            np.random.shuffle(self.data)
            self.batch_idx = 0
            self.epoch_idx += 1
            
        return batch_images
    
