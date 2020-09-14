'''
Created on 12 Aug 2020

@author: Tobias Pielok
'''

import numpy as np  
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple

def add_uni_time(snapshots: np.array, dt:float =1.0) -> np.array:
    '''
    Adds uniformely-spaced time dimension to snapshot matrix.
    
    :param snapshots: PxN snapshot matrix of P timesteps consisting of N features
    :param dt: timestep to be applied
    '''
    return np.hstack((snapshots, (np.arange(snapshots.shape[0]) * dt)[:, np.newaxis]))

def get_po_ts(timeseries: np.array, 
              num_obs: int, 
              features: List[int], 
              start_ind: int = 0, 
              num_step_size: int = 1) -> np.array:
    '''
    Returns partial observation time series
    
    :param timeseries: PxN time series matrix of P timesteps consisting of N-1 features
    :param num_obs: number of observations
    :param features: features to be selected
    :param start_ind: starting index
    :param num_step_size: step size
    '''
    
    return np.hstack((timeseries[start_ind:(start_ind+num_obs):num_step_size, features],
                      timeseries[start_ind:(start_ind+num_obs):num_step_size, -1][:, np.newaxis]))


def train_test_split_ts(ts: np.array, prop_train: float) -> Tuple[np.array]:
    '''
    Split time series into training and test set
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    :param prop_train: proportion of the training set
    '''
    num_train = int(ts.shape[0] * prop_train)
    ts_train = np.copy(ts[:num_train, :])
    ts_test = np.copy(ts[num_train:, :])
    ts_test[:, -1] = ts_test[:, -1] - ts_test[0, -1]
    return ts_train, ts_test

def mean_ts(ts: np.array) -> np.float:
    '''
    Returns the mean of the time series
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    '''
    return np.mean(ts[:, :-1], axis = 0)

def translate_ts(ts: np.array, v: np.array) -> np.array:
    '''
    Returns time series shifted in feature space 
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    :param v: Nx1 translation vector
    '''
    return ts + np.append(v, [0.])

def std_ts(ts: np.array) -> np.float:
    '''
    Returns the standard deviations of the time series
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    '''
    return np.diag(np.std(ts[:, :-1], axis = 0))
    
def scale_ts(ts: np.array, s_mat: np.array) -> np.array:
    '''
    Returns the time series scaled by a transformation matrix 
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    :param s_mat: N-1xN-1 transformation matrix
    '''
    
    return np.hstack((ts[:, :-1] @ s_mat, ts[:, -1][:, np.newaxis]))
    
def center_ts(ts: np.array) -> np.array: 
    '''
    Returns time series centered in feature space  
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    '''
    mean_x = mean_ts(ts)
    return translate_ts(ts, -mean_x), mean_x

def adapt_sampling_rate(ts: np.array, delta:float = 0.01) -> np.array:
    '''
    Returns time series with adapted sampling rated, s.t.
    ||z_m - z_m+1|| >= delta.
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    :param delta: threshold value
    '''
    
    cur_ind = 0
    indices = [0]

    num_obs = ts.shape[0]
    
    for ind in range(cur_ind + 1,num_obs-1):
        if (np.linalg.norm(ts[cur_ind, :-1] - ts[ind, :-1], ord = 2) > delta):
            cur_ind = ind
            indices.append(ind)
    indices.append(num_obs-1)
    return ts[indices, :]

def adapt_unif_sampling_rate(ts: np.array, stepsize: int = 40) -> np.array:
    '''
    Returns time series where every i'th step is taken
    
    :param ts:  PxN time series matrix of P timesteps consisting of N-1 features
    :param stepsize: i
    '''
    return ts[::stepsize]

def plot_ts(ts: np.array):
    '''
    Plots time series
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    '''
    plt.plot(ts[:, -1], ts[:, :-1])
    
def lin_ts(x: np.array, num_sub_ts: int) -> Tuple[np.array, List[int]]:
    '''
    Returns 'hankelized' time series and the associated split indices
    
    :param x: PxN time series matrix of P timesteps consisting of N-1 features
    :param num_sub_ts: number of splits
    '''
    
    n = x.shape[0] 
    chunk = int(n/num_sub_ts)
    
    first_chunk = n - (num_sub_ts-1)*chunk    
    chunks = [first_chunk + chunk*i for i in range(num_sub_ts)]

    x_lin = np.vstack([np.repeat(x[0, :-1][...,np.newaxis], first_chunk, 1),
              x[:first_chunk, -1]])
    for i in range(num_sub_ts-1):
        x_lin = np.hstack([x_lin, np.vstack([np.repeat(x[chunks[i], :-1][...,np.newaxis], chunk, 1),
              x[(chunks[i]):(chunks[i+1]), -1] -  x[(chunks[i]), -1]])])
    return x_lin.T, np.hstack([0,chunks[:-1]]).astype(int)

def prepare_train_model_data(x: np.array, num_sub_ts:int = 1, dtype=tf.float32) -> Tuple[List[tf.Tensor], List[int]]:
    '''
    Returns time series 'hankelized' and converted to tensor and also returns the associated split indices 
    
    :param x: PxN time series matrix of P timesteps consisting of N-1 features
    :param num_sub_ts: number of splits
    :param dtype: float type
    '''
    
    x_lin, chunks = lin_ts(x, num_sub_ts)
    x_rec = x[:, 0:-1]
    return [tf.convert_to_tensor(x_lin, dtype=dtype), tf.convert_to_tensor(x_rec, dtype=dtype)], chunks
