'''
Created on 13 Aug 2020

@author: Tobias Pielok
'''

from .ts_util import *  
import numpy as np
from typing import List, Tuple

class ts_data(object):
    
    def __init__(self, ts: np.array, prop_train: float =0.75, has_time:bool = True, delta_t:float = 1.0):
        '''
        Utility object for time series data. 
        
        :param ts: PxN time series matrix of P timesteps consisting of N-1 features or PxN snapshot matrix of P timesteps consisting of N features
        :param prop_train: proportion of training data
        :param has_time: is ts a snapshot matrix
        :param delta_t: timestep to be applied if ts is a snapshot matrix
        '''
        if (has_time):
            self.ts = ts
        else:
            self.ts = add_uni_time(ts, delta_t)
            
        self._create_train_test(prop_train)
            
        self.train_ts_centered = None
        self.test_ts_centered = None 
        self.ts_centered = None
        
        self.train_ts_norm = None
        self.test_ts_norm = None
        self.ts_norm = None
         
        self.train_mean = None    
        self.train_std = None 
        self.train_inv_std = None
        
        self.x = None
        self.x_val = None
        self.x_all = None
        
        self.x_inp = None
        self.train_chunks = None
        self.x_val_inp = None
        self.x_val2_inp = None
        
    
    @property
    def num_features(self):
        return self.ts.shape[1] - 1
    
    @property
    def num_obs(self):
        return self.ts.shape[0] 
 
    @property
    def num_train(self):
        return self.train_ts.shape[0]
     
    @property
    def num_train_filtered(self):
        return self.x.shape[0]
    
    @property
    def num_test(self):
        return self.test_ts.shape[0] 
 
    @property
    def num_test_filtered(self):
        return self.x_val.shape[0] 
 
    def _create_train_test(self, prop_train=0.75):
        self.train_ts, self.test_ts = train_test_split_ts(self.ts, prop_train)
        
    def standardize(self):
        '''
        Standardize all training and evaluation set with the mean and the standard deviation matrix of the training set.
        '''
        self.train_ts_centered, self.train_mean = center_ts(self.train_ts)
        self.test_ts_centered = translate_ts(self.test_ts, -self.train_mean)
        self.ts_centered = translate_ts(self.ts, -self.train_mean)

        self.train_std = std_ts(self.train_ts_centered)
        self.train_inv_std = np.linalg.inv(self.train_std)

        self.train_ts_norm = scale_ts(self.train_ts_centered, self.train_inv_std)
        self.test_ts_norm = scale_ts(self.test_ts_centered, self.train_inv_std)
        self.ts_norm = scale_ts(self.ts_centered, self.train_inv_std)
        
    def generate_train_model_inputs(self, num_train_chunks:int =1, rate:float =0):
        '''
        'Hankelize' the training data and apply adaptive sampling rate to all data sets. 
        
        :param num_train_chunks: number of chunks
        :param rate: threshold value for sampling
        '''
        
        self.x = adapt_sampling_rate(self.train_ts_norm, rate)
        self.x_val = adapt_sampling_rate(self.test_ts_norm, rate)
        self.x_all = adapt_sampling_rate(self.ts_norm, rate)
        
        self.x_inp, self.train_chunks = prepare_train_model_data(self.x, num_train_chunks)
        self.x_val_inp, _ = prepare_train_model_data(self.x_val, 1)

        self.x_all[:,-1] = self.x_all[:,-1] - self.x_all[self.train_chunks[-1],-1]
        self.x_val2_inp, _ = prepare_train_model_data(self.x_all[self.train_chunks[-1]:], 1)
        
        
        
        