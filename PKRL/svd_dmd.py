'''
Created on 12 Aug 2020

@author: Tobias Pielok
'''

import numpy as np  
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import expm
from scipy.linalg import logm

from typing import List, Tuple

def svd_dmd(ts: np.array, r: int) -> Tuple[np.array, np.array]:
    '''
    Returns the SVD-DMD of ts.
    Implementation of algorithm 1 of Lu, H. and Tartakovsky, D. M. Predictive Accuracy of Dynamic Mode Decomposition. 2019. eprint: arXiv:1905.01587.
    
    :param ts: PxN time series matrix of P timesteps consisting of N-1 features
    :param r:  dimension of the low-rank space
    '''
    u, s, vh = np.linalg.svd(ts[:-1, :-1].transpose(), full_matrices=False)
    
    d = min(r, ts.shape[1])
    u = u[:, :d]
    vh = vh[:d, :]
    s = s[:d]
    
    S = u.transpose() @ ts[1:, :-1].transpose() @ \
            vh.transpose() @ np.diag(s ** -1)[:d, :d]
        
    eigval, eigvec = np.linalg.eig(S) 
    Phi = ts[1:,:-1].transpose() @ vh.transpose() @ np.diag(s ** -1) @ eigvec
    
    return Phi, np.diag(eigval)

def dmd_predict(x0: np.array, T: np.array, A:np.array, num_pred:int, timescale:float = 1) -> np.array:
    '''
    Predict uniformely-spaced in time via 
        x_i = T * A^i * T^-1 * x0 
    and returns prediction (num_pred)x(N+1) time series (with added time dimension)
    
    :param x0: Nx1 init value of the prediction
    :param T: Nxr projection matrix 
    :param A: rxr low-rank prediction matrix
    :param num_pred: number of predictions to be made 
    :param timescale: timestep size of the predictions
    '''
    
    b0 = np.linalg.pinv(T) @ x0

    pred_ts = np.zeros((num_pred, len(x0) + 1))
    pred_ts[:, -1] = np.array(range(num_pred)) * timescale

    for i in range(num_pred):
        pred_ts[i, :-1] =  np.real(T @ (np.linalg.matrix_power(A, i) @ b0))
    
    return pred_ts

def grid_search_svd_dmd(ts_centered: np.array, im_dims: List[int], steps: List[int]) -> Tuple[np.array, np.array, int, int, float]:
    ''' 
    Returns the projection and prediction matrices of the best SVD-DMD found via grid search on the centered training data.
    Also returns the hyperparameters of the best SVD-DMD and its evaluation value.         
    
    :param ts_centered: PxN time series matrix of P timesteps consisting of N-1 features
    :param im_dims: search space of low-rank dimensions
    :param steps: serach space of number of steps to be included in estimation step
    '''
    
    x0 = ts_centered[0, :-1]
    min_value = float('inf')
    min_Phi = None
    min_i = None
    min_eigval = None
    min_t = None

    for i in im_dims:
        for t in steps:
            ts = ts_centered[:t, :]
            num_preds = ts_centered.shape[0]            
            Phi, eigval = svd_dmd(ts, i)

            new_value = np.mean((ts_centered[:,:-1] - dmd_predict(x0, Phi, eigval, num_preds, 1)[:,:-1])**2)

            if (min_value > new_value and (np.real(np.log(np.diag(eigval))) < 0).all()):
                min_value = new_value
                min_i = i
                min_t = t
                min_Phi = Phi
                min_eigval = eigval
                print(np.round(min_value,3), min_i, min_t)
                
    return min_Phi, min_eigval, min_i, min_t, min_value


def random_search_svd_dmd(ts_centered: np.array, im_dims: List[int], steps: List[int], num_tries: int) -> Tuple[np.array, np.array, int, int, float]:
    ''' 
    Returns the projection and prediction matrices of the best SVD-DMD found via random search on the centered training data.
    Also returns the hyperparameters of the best SVD-DMD and its evaluation value.         
    
    :param ts_centered: PxN time series matrix of P timesteps consisting of N-1 features
    :param im_dims: search space of low-rank dimensions
    :param steps: search space of number of steps to be included in estimation step
    :param num_tries: number of evaluations done by random search 
    '''
    
    x0 = ts_centered[0, :-1]
    min_value = float('inf')
    min_Phi = None
    min_i = None
    min_eigval = None
    min_t = None


    for k in range(num_tries):
            i = np.random.choice(im_dims)
            t = np.random.choice(steps)
            ts = ts_centered[:t, :]
            num_preds = ts_centered.shape[0]            
            Phi, eigval = svd_dmd(ts, i)

            new_value = np.mean((ts_centered[:,:-1] - dmd_predict(x0, Phi, eigval, num_preds, 1)[:,:-1])**2)

            if (min_value > new_value and (np.real(np.log(np.diag(eigval))) < 0).all()):
                min_value = new_value
                min_i = i
                min_t = t
                min_Phi = Phi
                min_eigval = eigval
                print(np.round(min_value,3), min_i, min_t)
                
    return min_Phi, min_eigval, min_i, min_t, min_value

def dmd_real(Phi: np.array, eigval: np.array) -> Tuple[np.array, np.array]:
    '''
    Returns the real SVD-DMD representation as described in Tobias Pielok, Residual Enhanced Probabilistic Koopman-based Representation Learning, Master's thesis
    
    :param Phi: Nxr complex projection matrix 
    :param eigval: rxr complex low-rank prediction matrix
    '''
    
    D = eigval.shape[0]
    T_star = np.diag(np.repeat(1/np.sqrt(2) + 0j,D))

    is_cmpl = is_complex(np.diag(eigval))
    T_22_star = 1/np.sqrt(2) * np.array([[1, 1],[1j, -1j]])

    for i in range(D-1):
        if(is_cmpl[i]):
            is_cmpl[i+1] = False # already processed here
            T_star[i:(i+2), i:(i+2)] = T_22_star

    T_star = 1/np.sqrt(2) * T_star
    T = np.linalg.pinv(T_star)

    eigval_r = np.real(T_star @ eigval @ T) 
    Phi_r = np.real(Phi @ T)

    if (np.max(np.imag(Phi @ T)) > 10 ** -9):
        print("Err:", np.max(np.imag(Phi @ T)))
    
    return Phi_r, eigval_r

def softplus(x):
    return np.log(np.exp(x)+1)

def softplus_inverse(x):
    return np.log(np.exp(x)-0.9999999)

def vec_to_K(params: np.array, use_softplus:bool=False) -> np.array:
    '''
    Returns the prediction matrix of the associated parameter vector using the parametrization of 
    Pan, S. and Duraisamy, K. Physics-Informed Probabilistic Learning of Linear Embeddings of Non-linear Dynamics With Guaranteed Stability. 2019. eprint: arXiv:1906.03663.
    
    :param params: Mx1 parameter vector
    :param use_softplus: flag whether the sigma parameters must be softplus-transformed
    '''
    r = int((len(params)+1)/2)
    
    if use_softplus:
        sigma = -(softplus(params[0:r])**2)
    else:
        sigma = -(params[0:r]**2)
    ceta  = params[r:2*r-1]
    
    K = np.diag(sigma)
    for i in range(0,r-1):
        K[i, i + 1] = ceta[i]
        K[i + 1, i] = -ceta[i]    
    return K

def K_to_vec(K: np.array, timescale:float =1.0, use_softplus_inv:bool = False):
    '''
    Returns the parameter vector of the associated prediction matrix using the parametrization of 
    Pan, S. and Duraisamy, K. Physics-Informed Probabilistic Learning of Linear Embeddings of Non-linear Dynamics With Guaranteed Stability. 2019. eprint: arXiv:1906.03663.
    
    :param params: Mx1 parameter vector
    :param timescale: scale time 'dimension' by this value
    :param use_softplus_inv: flag whether the sigma parameters are softplus-transformed
    '''    
    
    if use_softplus_inv:
        return np.hstack([softplus_inverse(np.sqrt(-np.diag(K)/timescale)), np.diag(K, 1)/timescale])
    else:
        return np.hstack([np.sqrt(-np.diag(K)/timescale), np.diag(K, 1)/timescale])    

def is_complex(x):
    return np.abs(np.imag(x)) > 10 ** -9

def getK(A: np.array) -> Tuple[np.array, np.array]:
    '''
    Returns the transformation and prediction matrices [K, T] of a (complex) low-rank prediction matrix using the parametrization of 
    Pan, S. and Duraisamy, K. Physics-Informed Probabilistic Learning of Linear Embeddings of Non-linear Dynamics With Guaranteed Stability. 2019. eprint: arXiv:1906.03663, s.t.
        T * exp(K) * T^-1 = A.
    
    :param A: rxr (complex) low-rank prediction matrix
    '''
    
    A_eval, A_evec = np.linalg.eig(A)
    logA_eval = np.log(A_eval)
    sig = [np.sqrt(-np.real(logA_eval [i])) for i in range(len(logA_eval ))]
    cet = [np.abs(np.imag(logA_eval [i])) for i in range(len(logA_eval ))]

    set_zero = False
    for i in range(len(cet)):
        if (is_complex(logA_eval[i])):
            if (not set_zero):
                set_zero = True
            else:
                cet[i] = 0
                set_zero = False 
        else:
            cet[i] = 0
            
    cet = cet[:-1]
    
    K = vec_to_K(np.hstack([sig, cet]))
    
    expK_eval, expK_evec = np.linalg.eig(expm(K))
    
    return K, np.real(A_evec @ np.linalg.inv(expK_evec))

def extend_mats(T: np.array, K: np.array, diff: int) -> Tuple[np.array, np.array]:
    '''
    Returns the SVD-DMD matrices extended by 'diff' zero columns and rows [T_e, K_e], s.t.
        T_e in Nx(r + diff), K_e in (r + diff)x(r + diff).
    
    :param T: Nxr projection matrix 
    :param K: rxr low-rank prediction matrix
    :param diff: number of zero columns and rows to be added
    '''
    
    T_e = np.hstack([T, np.zeros((T.shape[0], diff))])
    K_e = np.pad(K, ((0, diff), (0, diff)), 'constant', constant_values=(0, 0))
    return T_e, K_e