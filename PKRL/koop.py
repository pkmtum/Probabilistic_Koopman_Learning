'''
Created on 13 Aug 2020

@author: Tobias Pielok
'''

import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import h5py

from .models import *
from .svd_dmd import *
from .ts_data import *
from .dmd import *

from typing import List, Tuple

negloglik = lambda y, p_y: -p_y.log_prob(y)

class koop_model(object):
    
    def __init__(self, 
                 ts: ts_data, 
                 d: dmd, 
                 enc_nn_def: List[int], 
                 dec_nn_def: List[int],
                 prob:bool=False, 
                 res_enc_nn_def:List[int]=None, 
                 regu:[tfk.regularizers.Regularizer]=None, 
                 fix_inner_trafos:bool=False):
        '''
        Utility object for (RE-)(P)KM-model
        
        :param ts: time series data
        :param d: dmd object
        :param enc_nn_def: encoder NN definition
        :param dec_nn_def: decoder NN definition
        :param prob: flag whether probabilistic or deterministic model
        :param res_enc_nn_def: encoder NN definition for the residual network
        :param regu: regularizer to be applied (for deterministic model) 
        :param fix_inner_trafos: freeze inner linear transformations 
        '''
        
        self.ts_data = ts
        self.dmd = d
        self.enc_nn_def = enc_nn_def
        self.dec_nn_def = dec_nn_def
        self.prob = prob
        self.regu = regu
        self.fix_inner_trafos = fix_inner_trafos
          
        self.V_r = self.dmd.T
        self.num_lin_dim = self.V_r.shape[1]
          
        num_train = self.ts_data.num_train_filtered
          
        self.inp_enc, self.enc_nn_model, self.enc_added_model, self.enc_svd_model, self.enc_model = create_enc_models(enc_nn_def, self.ts_data.num_features, self.num_lin_dim, self.V_r, self.ts_data.train_std, prob, num_train, regu=regu, fix_inner_trafos=fix_inner_trafos)
        self.inp_exp, self.exp_model, self.inp_lin, self.lin_pred_model = create_exp_model(self.num_lin_dim, self.ts_data.num_features, self.dmd.K_vec, self.enc_model, prob, num_train, regu=regu)
        self.inp_dec, self.dec_nn_model, self.dec_svd_model, self.dec_model = create_dec_models(dec_nn_def, self.ts_data.num_features, self.num_lin_dim, self.V_r, self.ts_data.train_std, prob, num_train, regu=regu, fix_inner_trafos=fix_inner_trafos)
        
        if(res_enc_nn_def is not None):
            _, self.res_enc_nn_model = create_enc_nn_model(res_enc_nn_def, self.ts_data.num_features, self.num_lin_dim, prob, num_train, regu=regu)
            _, self.res_exp_model, _, self.res_lin_pred_model = create_exp_model(self.num_lin_dim, self.ts_data.num_features, np.zeros(self.dmd.K_vec.shape), self.res_enc_nn_model, prob, num_train, regu=regu)
        else:
            self.res_enc_nn_model = None 
            self.res_lin_pred_model = None 

          
        self.train_model, self.pred_model, self.std_model, self.pred_std_model = create_train_pred_model(self.ts_data.num_features, self.num_lin_dim, self.lin_pred_model, self.inp_lin, self.enc_model, self.dec_model, prob, num_train, self.res_lin_pred_model)
            
        self.generate_train_outputs()
            
    def generate_train_outputs(self):       
        self.train_output_size = self.train_model.output.shape[-1]
        self.y = np.zeros((self.ts_data.num_train_filtered, self.train_output_size))
        self.y_val = np.zeros((self.ts_data.num_test_filtered, self.train_output_size))          
            
    def compile_train_model_adam(self, learning_rate):
        if(not self.prob):
            self.compile_train_model(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=tfk.losses.MSE)
        else:
            self.compile_train_model(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)
                
              
    def compile_train_model(self, optimizer, loss):
        self.train_model.compile(optimizer=optimizer, loss=loss)
            
    def train(self, epochs, batch_dim, v=1):
        self.history = self.train_model.fit(self.ts_data.x_inp, self.y, epochs=epochs, verbose=v, batch_size=batch_dim,
                         use_multiprocessing=True, validation_data=[self.ts_data.x_val_inp, self.y_val])
            
    def pred_data(self):
        self.pred_train_data()
        self.pred_test_data()
        self.pred_test_data_forerun()
     
    def pred_train_data(self, num_preds=1):
        if(self.prob):
            self.train_mean = np.mean([self.pred_model(self.ts_data.x_inp[0]).mean() for j in range(num_preds)], 0)
            self.train_std = np.mean([self.pred_model(self.ts_data.x_inp[0]).stddev() for j in range(num_preds)], 0)
        else: 
            self.train_mean = self.pred_model(self.ts_data.x_inp[0])
        
    def pred_test_data(self, num_preds=1):
        if(self.prob):
            self.test_mean = np.mean([self.pred_model(self.ts_data.x_val_inp[0]).mean() for j in range(num_preds)], 0)
            self.test_std = np.mean([self.pred_model(self.ts_data.x_val_inp[0]).stddev() for j in range(num_preds)], 0)
        else:         
            self.test_mean = self.pred_model(self.ts_data.x_val_inp[0])
        
    def pred_test_data_forerun(self, num_preds=1):
        if(self.prob):
            self.test_mean_fr = np.mean([self.pred_model(self.ts_data.x_val2_inp[0]).mean() for j in range(num_preds)], 0)
            self.test_std_fr = np.mean([self.pred_model(self.ts_data.x_val2_inp[0]).stddev() for j in range(num_preds)], 0)
        else: 
            self.test_mean_fr = self.pred_model(self.ts_data.x_val2_inp[0])
        
    def plot_pred_train(self, features_to_show=None):
        if(self.prob):
            self.plot_prob_pred(self.ts_data.x, self.train_mean, self.train_std, self.ts_data.x_inp[1], features_to_show)
        else:
            self.plot_pred(self.ts_data.x, self.train_mean, self.ts_data.x_inp[1], features_to_show)
        
        
    def save_sample_pred(self, x_inp, x_rec, num_samples, name, time=None):
        preds = self.pred_model(x_inp).sample(num_samples).numpy()
        
        if(time is None):
            time = x_inp[:, -1].numpy()
                
        to_save = np.array([preds, time, x_rec.numpy()], dtype=object)     
        np.save(name, to_save)
                
    def save_sample_preds(self, num_samples, name=""):
            self.save_sample_pred(self.ts_data.x_inp[0], self.ts_data.x_inp[1], num_samples, name+"sampled_train", time=self.ts_data.x[:,-1])
            self.save_sample_pred(self.ts_data.x_val_inp[0], self.ts_data.x_val_inp[1], num_samples, name+"sampled_test")
            self.save_sample_pred(self.ts_data.x_val2_inp[0], self.ts_data.x_val2_inp[1], num_samples, name+"sampled_ftest")
        
    def plot_pred_test_forerun(self, features_to_show=None):
        if(self.prob):
            self.plot_prob_pred(self.ts_data.x_val2_inp[0], self.test_mean_fr, self.test_std_fr, self.ts_data.x_val2_inp[1], features_to_show)
        else:
            self.plot_pred(self.ts_data.x_val2_inp[0], self.test_mean_fr, self.ts_data.x_val2_inp[1], features_to_show)
        
    def plot_pred_test(self, features_to_show=None):
        if(self.prob):
            self.plot_prob_pred(self.ts_data.x_val_inp[0], self.test_mean, self.test_std, self.ts_data.x_val_inp[1], features_to_show)
        else:
            self.plot_pred(self.ts_data.x_val_inp[0], self.test_mean, self.ts_data.x_val_inp[1], features_to_show)
            
    def plot_prob_pred(self, x, sim_mean, sim_std, x_rec, features_to_show=None):
        if(features_to_show is None):
            features_to_show = range(self.ts_data.num_features)
        
        num_to_show = len(features_to_show)
        for i in features_to_show:
            plt.subplot(num_to_show, 2, i*2+1)
            data = np.vstack([sim_mean[:, i], sim_mean[:, i] + sim_std[:, i], 
                         sim_mean[:, i] - sim_std[:, i]])
            plt.plot(x[:,-1], np.transpose(data))
            plt.subplot(num_to_show, 2, i*2+2)
            data = np.vstack([np.transpose(x_rec[:, i]), data])
            plt.plot(x[:,-1], np.transpose(data))
            
    def plot_pred(self, x, sim_mean, x_rec, features_to_show=None):
        if(features_to_show is None):
            features_to_show = range(self.ts_data.num_features)
        
        num_to_show = len(features_to_show)
        for i in features_to_show:
            plt.subplot(num_to_show, 2, i*2+1)
            plt.plot(x[:,-1], sim_mean[:, i])
            plt.subplot(num_to_show, 2, i*2+2)
            data = np.vstack([np.transpose(x_rec[:, i]), sim_mean[:, i]])
            plt.plot(x[:,-1], np.transpose(data))
            
    def load_weights(self, fname):
        w = np.load(fname + ".npy", allow_pickle=True)
        self.enc_nn_model.set_weights(w[0])
        self.enc_added_model.set_weights(w[1])
        self.exp_model.set_weights(w[2])
        self.dec_nn_model.set_weights(w[3])
        self.enc_svd_model.set_weights(w[4])
        self.dec_svd_model.set_weights(w[5])
        
        c = 0
        if(self.prob):
            self.std_model.set_weights(w[6])
            self.pred_std_model.set_weights(w[7])
            c = 2
            
        if(self.res_enc_nn_model is not None and len(w)>6 + c):
            self.res_enc_nn_model.set_weights(w[6+c])
            self.res_exp_model.set_weights(w[7+c])            
        
    def save_weights(self, fname):
        if(self.res_enc_nn_model is None):
            if(self.prob):                
                weights = np.array([self.enc_nn_model.get_weights(), 
                                self.enc_added_model.get_weights(), 
                                self.exp_model.get_weights(), 
                                self.dec_nn_model.get_weights(),
                                self.enc_svd_model.get_weights(),
                                self.dec_svd_model.get_weights(),
                                self.std_model.get_weights(),
                                self.pred_std_model.get_weights()], dtype=object)
            else:
                 weights = np.array([self.enc_nn_model.get_weights(), 
                                self.enc_added_model.get_weights(), 
                                self.exp_model.get_weights(), 
                                self.dec_nn_model.get_weights(),
                                self.enc_svd_model.get_weights(),
                                self.dec_svd_model.get_weights()], dtype=object)               
        else:
            if(self.prob):
                weights = np.array([self.enc_nn_model.get_weights(), 
                                self.enc_added_model.get_weights(), 
                                self.exp_model.get_weights(), 
                                self.dec_nn_model.get_weights(),
                                self.enc_svd_model.get_weights(),
                                self.dec_svd_model.get_weights(),
                                self.std_model.get_weights(),
                                self.pred_std_model.get_weights(),
                                self.res_enc_nn_model.get_weights(),
                                self.res_exp_model.get_weights()], dtype=object)
            else:
                weights = np.array([self.enc_nn_model.get_weights(), 
                                self.enc_added_model.get_weights(), 
                                self.exp_model.get_weights(), 
                                self.dec_nn_model.get_weights(),
                                self.enc_svd_model.get_weights(),
                                self.dec_svd_model.get_weights(),
                                self.res_enc_nn_model.get_weights(),
                                self.res_exp_model.get_weights()], dtype=object)              
        np.save(fname, weights)
    
    def set_weights(self, model, weights, s=-10, shift=4, cshift=0):
        m_w = model.get_weights()

        for i in range(len(weights)):
            m_w[i*shift+cshift]   = weights[i]
            m_w[i*shift+1+cshift] = s * np.ones(m_w[i*shift+1+cshift].shape)
        model.set_weights(m_w)
  
    def set_exp_weights(self, exp_model, weights, s=-10, shift=0):
        m_w = exp_model.get_weights()
        m_w[0+shift] = np.hstack([K_to_vec(vec_to_K(weights[0]), use_softplus_inv=True), s*np.ones(len(weights[0]))])
        exp_model.set_weights(m_w)
    
    def init_with_mle_weights(self, fname:str, s:float=-10):
        '''
        Init (RE-)PKM-model with the weights of a (RE-)KM-model
        
        :param fname: filename of the weights of the (RE-)KM-model
        :param s: unscaled standard deviation parameters 
        '''
        
        
        weights = np.load(fname + ".npy", allow_pickle=True)
        
        self.set_weights(self.enc_nn_model, weights[0], s=s)
        self.set_weights(self.enc_nn_model, weights[0], cshift=2, s=s)

        if(not self.fix_inner_trafos):
            self.set_weights(self.enc_added_model, weights[1], s=s)
            self.set_weights(self.enc_added_model, weights[1], cshift=2, s=s)

        self.set_exp_weights(self.exp_model, weights[2], s=s)
        self.set_exp_weights(self.exp_model, weights[2], shift=1, s=s)

        if(self.fix_inner_trafos):
            self.set_weights(self.dec_nn_model, weights[3][1:], cshift=1, s=s)
            self.set_weights(self.dec_nn_model, weights[3][1:], cshift=3, s=s)            
        else:
            self.set_weights(self.dec_nn_model, weights[3], s=s)
            self.set_weights(self.dec_nn_model, weights[3], cshift=2, s=s)

        self.enc_svd_model.set_weights(weights[4])
        self.dec_svd_model.set_weights(weights[5])
        
        if(self.res_enc_nn_model is not None and len(weights)>6):
            self.set_weights(self.res_enc_nn_model, weights[6], s=s)
            self.set_weights(self.res_enc_nn_model, weights[6], cshift=2, s=s)
            
            self.set_exp_weights(self.res_exp_model, weights[7], s=s)
            self.set_exp_weights(self.res_exp_model, weights[7], shift=1, s=s)
    
    def compute_elbos(self, num_samples):   
        return [np.mean(self.train_model.loss(self.y, self.train_model(self.ts_data.x_inp))) for i in range(num_samples)]
    