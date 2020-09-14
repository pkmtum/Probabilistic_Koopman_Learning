'''
Created on 13 Aug 2020

@author: Tobias Pielok
'''

import numpy as np

from .svd_dmd import *
from .ts_data import ts_data

from typing import List, Tuple

class dmd(object):
    
    def __init__(self):
        '''
        Utility object for Dynamic-Mode-Decomposition
        '''
        self.Phi = None
        self.eigval = None
        self.Phi_r = None
        self.eigval_r = None
        self.K = None
        self.T = None
    
    def svd_dmd(self, ts: np.array, output_dim: int):
        '''
        Computes the SVD-DMD of ts. 
        
        :param ts: PxN time series matrix of P timesteps consisting of N-1 features
        :param output_dim: dimension of the low-rank space
        '''
        self._compute_timescale(ts) # assumes observations uniformely-spaced in time
        self.Phi, self.eigval = svd_dmd(ts, output_dim)
        self._compute_real_transforms()
  
    def random_search_stable_svd_dmd(self, ts: np.array, output_dims: int, timesteps: List[int], num_tries: List[int]):
        '''
        Tries to find the best SVD-DMD via random search.
        
        :param ts: PxN time series matrix of P timesteps consisting of N-1 features
        :param output_dims: search space of low-rank dimensions
        :param timesteps: search space of number of steps to be included in estimation step
        :param num_tries: number of evaluations done by random search 
        '''
    
        self._compute_timescale(ts)        
        self.Phi, self.eigval, _, _, _ = random_search_svd_dmd(ts, output_dims, timesteps, num_tries)
        self._compute_real_transforms() 
        
    def grid_search_stable_svd_dmd(self, ts: np.array, output_dims: int, timesteps: List[int]):
        '''
        Finds the best SVD-DMD via grid search.
        
        :param ts: PxN time series matrix of P timesteps consisting of N-1 features
        :param output_dims: search space of low-rank dimensions
        :param timesteps: search space of number of steps to be included in estimation step
        :param num_tries: number of evaluations done by random search 
        '''
        
        self._compute_timescale(ts)        
        self.Phi, self.eigval, _, _, _ = grid_search_svd_dmd(ts, output_dims, timesteps)
        self._compute_real_transforms()
        
    def _compute_timescale(self, ts: np.array):
        '''
        Computes timestep size
        
        :param ts: PxN time series matrix of P timesteps consisting of N-1 features
        '''
        self.timescale = ts[1, -1] - ts[0, -1] # Assumption: at least two observations and observations uniformely-spaced in time
        
    def _compute_real_transforms(self):
        self.Phi_r, self.eigval_r = dmd_real(self.Phi, self.eigval)

    def compute_K(self, extend:int=0):
        '''
        Computes the Koopman operator matrix and extend it with zero entries if necessary.
        :param extend: number of zero rows/columns to be added
        '''
        
        self.K, T_tmp = getK(self.eigval_r)
        self.T = self.Phi_r @ T_tmp
        if(extend > 0):
            self.T, self.K = extend_mats(self.T, self.K, extend)
    
    @property    
    def K_vec(self):
        return K_to_vec(self.K, self.timescale)
        
    def K_vec_prob(self, s=-10):
        return np.array(np.hstack([self.vec_K, s*np.ones(len(self.vec_K))]), dtype=np.float32)
        
    def predict(self, x0:np.array, num_preds:int, type:str='C'):
        '''
        Predict uniformely-spaced in time using the SVD-DMD matrices. 
        
        :param x0: Nx1 init value of the prediction
        :param num_preds: number of predictions
        :param type: 'K'oopman, 'R'eal, 'C'omplex (Just for testing since all types should in prinicipal predict the same.)
        '''
        
        if(type == 'K'):
            return dmd_predict(x0, self.T, expm(self.K), num_preds, self.timescale)
        elif(type == 'R'):
            return dmd_predict(x0, self.Phi_r, self.eigval_r, num_preds, self.timescale)
        elif(type == 'C'):
            return dmd_predict(x0, self.Phi, self.eigval, num_preds, self.timescale)
        else:
            return None
