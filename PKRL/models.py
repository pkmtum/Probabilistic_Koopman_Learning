'''
Created on 12 Aug 2020

@author: Tobias Pielok
'''

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import tensorflow_probability.python.distributions as tfd
import numpy as np

from . import layers

from typing import List, Tuple

kernel_divergence_fn_gen=lambda num_train: lambda q, p, _: tfd.kl_divergence(q, p) / (num_train * 1.0)
bias_divergence_fn_gen=lambda num_train: lambda q, p, _: tfd.kl_divergence(q, p) / (num_train * 1.0)

b_post  = tfp.layers.util.default_mean_field_normal_fn()
b_prior = tfp.layers.util.default_mean_field_normal_fn()
k_post = tfp.layers.util.default_mean_field_normal_fn()
k_prior = tfp.layers.util.default_mean_field_normal_fn()

def prior_exp(kernel_size: int, dtype=None, K_init: np.array=None):
    '''
    Mean-field-like prior definition for the exponential layer 
    
    :param kernel_size: number of parameters of K
    :param dtype: float type
    :param K_init: init parameter vector for K
    '''

    n = kernel_size 
    return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=tf.constant_initializer(
                K_init)),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(t[..., :n], tf.math.softplus(t[..., n:(2*n)])),
              reinterpreted_batch_ndims=1)),
    ])

def posterior_exp_mean_field(kernel_size: int, dtype=None, K_init: np.array=None):
    '''
    Mean-field posterior definition for the exponential layer 
    
    :param kernel_size: number of parameters of K
    :param dtype: float type
    :param K_init: init parameter vector for K
    '''
        
    n = kernel_size 
    return tf.keras.Sequential([
          tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=tf.constant_initializer(
                K_init)),
          tfp.layers.DistributionLambda(lambda t: tfd.Independent(
              tfd.Normal(t[..., :n], tf.math.softplus(t[..., n:(2*n)])),
              reinterpreted_batch_ndims=1)),
    ])

prior_exp_gen = lambda K_init: lambda kernel_size, dtype=None: prior_exp(kernel_size, dtype, K_init) 
post_exp_gen = lambda K_init: lambda kernel_size, dtype=None: posterior_exp_mean_field(kernel_size, dtype, K_init) 

def create_enc_nn_model(enc_nn_def: List[int], num_features: int, num_lin_dim: int, prob:bool=False, num_train:int=None, regu:[tfk.regularizers.Regularizer]=None) -> Tuple[tfk.Input, tfk.Model]:
    '''
    Returns the  NN encoder and its input layer 
    
    :param enc_nn_def: encoder NN definition
    :param num_features: number of features
    :param num_lin_dim: dimension of the linear space
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: number of training samples
    :param regu: regularizer to be applied (for deterministic model) 
    '''
    
    if(prob):
        if (num_train is None):
            print("num_train must be set when prob mode is used")
            return None, None, None, None, None
        else:
            kernel_divergence_fn = kernel_divergence_fn_gen(num_train)
            bias_divergence_fn = bias_divergence_fn_gen(num_train)
            
    inp_enc = tfk.Input(shape=(num_features))
    
    enc_nn_layers = [inp_enc]
    for e in enc_nn_def:
        if(prob):
            enc_nn_layers.append(tfp.layers.DenseReparameterization(e, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,    
                           kernel_posterior_fn=k_post,
                           kernel_prior_fn=k_prior,                           
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,
                           activation=tf.nn.swish))
        else:
            enc_nn_layers.append(tfk.layers.Dense(e, activation=tf.nn.swish,
                                 kernel_regularizer=regu,
                                 bias_regularizer=regu))
    
    if(prob):
        enc_nn_layers.append(tfp.layers.DenseReparameterization(num_lin_dim, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,    
                           kernel_posterior_fn=k_post,
                           kernel_prior_fn=k_prior,                            
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn)) 
    else:              
        enc_nn_layers.append(tfk.layers.Dense(num_lin_dim,
                                 kernel_regularizer=regu,
                                 bias_regularizer=regu))
        
    enc_nn_model = tfk.Sequential(enc_nn_layers)
    
    return inp_enc, enc_nn_model
    
def create_enc_models(enc_nn_def: List[int], 
                      num_features: int, 
                      num_lin_dim: int, 
                      V_r: np.array, 
                      train_std_x: np.array, 
                      prob:bool=False, 
                      num_train:int=None, 
                      regu:[tfk.regularizers.Regularizer]=None, 
                      fix_inner_trafos:bool=False) -> Tuple[tfk.Input, tfk.Model, tfk.Model, tfk.Model, tfk.Model]:
    '''
    Return encoder model (and all embeded models) and its input layer.
    
    :param enc_nn_def: encoder NN definition
    :param num_features: number of features
    :param num_lin_dim: dimension of the linear space
    :param V_r: projection matrix
    :param train_std_x: standard deviation matrix of the training data
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: number of training samples
    :param regu: regularizer to be applied (for deterministic model) 
    :param fix_inner_trafos: freeze inner linear transformations 
    '''
    if(prob):
        if (num_train is None):
            print("num_train must be set when prob mode is used")
            return None, None, None, None, None
        else:
            kernel_divergence_fn = kernel_divergence_fn_gen(num_train)
            bias_divergence_fn = bias_divergence_fn_gen(num_train)
            
    inp_enc, enc_nn_model = create_enc_nn_model(enc_nn_def, num_features, num_lin_dim, prob, num_train)    
    
    enc_svd_mat = train_std_x @ np.linalg.pinv(V_r).transpose()

    enc_svd = tfk.layers.Dense(num_lin_dim, use_bias=False, 
                               kernel_initializer=tf.constant_initializer(enc_svd_mat),
                               trainable=False
                               )(inp_enc)

    inp_added = tfk.Input(shape=(num_lin_dim))
    
    
    if(not fix_inner_trafos):
        if(prob):
            added = tfp.layers.DenseReparameterization(num_lin_dim, 
                               bias_posterior_fn=b_post,
                               bias_prior_fn=b_prior,    
                               kernel_posterior_fn=k_post,
                               kernel_prior_fn=k_prior,                            
                               kernel_divergence_fn=kernel_divergence_fn,
                               bias_divergence_fn=bias_divergence_fn)(inp_added)
        else:    
            added = tfk.layers.Dense(num_lin_dim,
                                     kernel_regularizer=regu,
                                     kernel_initializer=tf.keras.initializers.Identity(),
                                     bias_initializer=tf.zeros_initializer(),                                     
                                     bias_regularizer=regu)(inp_added)
    else:
        added = tfk.layers.Dense(num_lin_dim, use_bias=False, 
                               kernel_initializer=tf.keras.initializers.Identity(),
                               trainable=False)(inp_added)
        
    enc_added_model = tfk.Model(inputs = inp_added, outputs = added)

    enc_added = tfk.layers.add([enc_nn_model.output, enc_svd])
    enc_added = enc_added_model(enc_added)

    enc_model = tfk.Model(inputs = inp_enc, outputs = enc_added)
    enc_svd_model = tfk.Model(inputs = inp_enc, outputs = enc_svd)
    
    return inp_enc, enc_nn_model, enc_added_model, enc_svd_model, enc_model

def create_exp_model(num_lin_dim: int,
                     num_features: int, 
                     vec_K: np.array, 
                     enc_model: tfk.Model, 
                     prob:bool=False, 
                     num_train:int=None, 
                     s:float=-10, 
                     regu:[tfk.regularizers.Regularizer]=None) -> Tuple[tfk.Input, tfk.Model, tfk.Input, tfk.Model]:
    '''
    Returns the exponential model and the linear prediction model and their respective input layers.
    
    :param num_lin_dim: dimension of the linear space
    :param num_features: number of features
    :param vec_K: parameter vector of the prediction matrix K
    :param enc_model: Encoder model
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: flag whether probabilistic or deterministic model
    :param s: unscaled standard deviation parameters of the probablistic exp model 
    :param regu: regularizer to be applied (for deterministic model) 
    '''
    
    if(prob and num_train is None):
        print("num_train must be set when prob mode is used")
        return None, None, None, None
    
    inp_exp = tfk.Input(shape=(num_lin_dim+1))
    
    if(prob):
        k_init = np.array(np.hstack([vec_K, s*np.ones(len(vec_K))]), dtype=np.float32)
        exp_layer = layers.ExpVariational(post_exp_gen(k_init), prior_exp_gen(k_init), num_train)
    else:
        exp_layer = layers.ExpLayer(tf.constant_initializer(vec_K),
                                 param_regularizer=regu)
        
    pred_exp = layers.ExpDistributed(num_lin_dim, exp_layer)(inp_exp[tf.newaxis,...])
    exp_model = tfk.Model(inputs = inp_exp, outputs = pred_exp[0,...])

    inp_lin = tfk.Input(shape=(num_features+1))
    enc = enc_model(inp_lin[...,:num_features])
    combined_enc = tfk.layers.Concatenate()([enc, inp_lin[...,-1][..., tf.newaxis]])
    pred = exp_model(combined_enc)

    lin_pred_model = tfk.Model(inputs = inp_lin, outputs = pred)
    
    return inp_exp, exp_model, inp_lin, lin_pred_model

def create_dec_nn_model(dec_nn_def: List[int], num_features: int, num_lin_dim: int, prob:bool=False, num_train:int=None, regu:[tfk.regularizers.Regularizer]=None) -> Tuple[tfk.Input, tfk.Model]:
    '''
    Returns the  NN decoder and its input layer 
    
    :param dec_nn_def: decoder NN definition
    :param num_features: number of features
    :param num_lin_dim: dimension of the linear space
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: number of training samples
    :param regu: regularizer to be applied (for deterministic model) 
    '''
    
    if(prob):
        if (num_train is None):
            print("num_train must be set when prob mode is used")
            return None, None, None, None, None
        else:
            kernel_divergence_fn = kernel_divergence_fn_gen(num_train)
            bias_divergence_fn = bias_divergence_fn_gen(num_train)
    
    inp_nn_dec = tfk.Input(shape=(num_lin_dim))

    dec_nn_layers = [inp_nn_dec]
    for d in dec_nn_def:
        if(prob):
            dec_nn_layers.append(tfp.layers.DenseReparameterization(d, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,    
                           kernel_posterior_fn=k_post,
                           kernel_prior_fn=k_prior,                           
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,
                           activation=tf.nn.swish))
        else:
            dec_nn_layers.append(tfk.layers.Dense(d, activation=tf.nn.swish,
                                 kernel_regularizer=regu,
                                 bias_regularizer=regu))
    if(prob):
        dec_nn_layers.append(tfp.layers.DenseReparameterization(num_features, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,    
                           kernel_posterior_fn=k_post,
                           kernel_prior_fn=k_prior,                            
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn)) 
    else:              
        dec_nn_layers.append(tfk.layers.Dense(num_features,
                                 kernel_regularizer=regu,
                                 bias_regularizer=regu))
    
    dec_nn_inner_model = tfk.Sequential(dec_nn_layers)
    return dec_nn_inner_model  
    

def create_dec_models(dec_nn_def: List[int], 
                      num_features: int, 
                      num_lin_dim: int, 
                      V_r: np.array, 
                      train_std_x: np.array, 
                      prob:bool=False, 
                      num_train:int=None, 
                      regu:[tfk.regularizers.Regularizer]=None, 
                      fix_inner_trafos:bool=False) -> Tuple[tfk.Input, tfk.Model, tfk.Model, tfk.Model]:
    '''
    Return decoder model (and all embeded models) and its input layer.
    
    :param dec_nn_def: decoder NN definition
    :param num_features: number of features
    :param num_lin_dim: dimension of the linear space
    :param V_r: projection matrix
    :param train_std_x: standard deviation matrix of the training data
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: number of training samples
    :param regu: regularizer to be applied (for deterministic model) 
    :param fix_inner_trafos: freeze inner linear transformations 
    '''
    
    if(prob):
        if (num_train is None):
            print("num_train must be set when prob mode is used")
            return None, None, None, None
        else:
            kernel_divergence_fn = kernel_divergence_fn_gen(num_train)
            bias_divergence_fn = bias_divergence_fn_gen(num_train)
    
    dec_nn_inner_model = create_dec_nn_model(dec_nn_def, num_features, num_lin_dim, prob, num_train, regu)
    
    inp_dec = tfk.Input(shape=(num_lin_dim))
    
    if(not fix_inner_trafos):
        if(prob):
            dec_both = tfp.layers.DenseReparameterization(num_lin_dim, 
                               bias_posterior_fn=b_post,
                               bias_prior_fn=b_prior,    
                               kernel_posterior_fn=k_post,
                               kernel_prior_fn=k_prior,                            
                               kernel_divergence_fn=kernel_divergence_fn,
                               bias_divergence_fn=bias_divergence_fn)(inp_dec)
        else:
            dec_both = tfk.layers.Dense(num_lin_dim,
                                     kernel_regularizer=regu,
                                     kernel_initializer=tf.keras.initializers.Identity(),
                                     bias_initializer=tf.zeros_initializer(),                                     
                                     bias_regularizer=regu)(inp_dec)
    else:
        dec_both = tfk.layers.Dense(num_lin_dim,use_bias=False, 
                               kernel_initializer=tf.keras.initializers.Identity(),
                               trainable=False)(inp_dec)       
    
    dec_nn = dec_nn_inner_model(dec_both)
    
    dec_svd_mat = V_r.transpose() @ np.linalg.inv(train_std_x)
    inp_svd = tfk.Input(shape=(num_lin_dim))
    dec_svd = tfk.layers.Dense(num_features, use_bias=False, 
                               kernel_initializer=tf.constant_initializer(dec_svd_mat),
                               trainable=False,
                               )(inp_svd)
    dec_svd_model = tfk.Model(inputs = inp_svd, outputs = dec_svd)
    
    dec_nn_model = tfk.Model(inputs = inp_dec, outputs = dec_nn)
    dec_added = tfk.layers.add([dec_nn, dec_svd_model(dec_both)])
    dec_model = tfk.Model(inputs = inp_dec, outputs = dec_added)
    
    return inp_dec, dec_nn_model, dec_svd_model, dec_model

def create_train_pred_model(num_features: int, 
                            num_lin_dim: int, 
                            lin_pred_model: tfk.Model, 
                            inp_lin: tfk.Input, 
                            enc_model: tfk.Model, 
                            dec_model: tfk.Model, 
                            prob:bool=False, 
                            num_train:int=None, 
                            res_lin_pred_model:tfk.Model=None) -> Tuple[tfk.Model, tfk.Model, tfk.Model, tfk.Model]:
    '''
    Returns the training and the prediction model and their associated standard deviations models (if the probabilistic model is used)
    
    :param num_features: number of features
    :param num_lin_dim: dimension of the linear space
    :param lin_pred_model: linear prediction model
    :param inp_lin: input layer of the linear prediction model
    :param enc_model: encoder model
    :param dec_model: decoder model
    :param prob: flag whether probabilistic or deterministic model
    :param num_train: number of training samples
    :param res_lin_pred_model: linear prediction model of the residuals
    '''
    
    if(prob):
        if (num_train is None):
            print("num_train must be set when prob mode is used")
            return None, None
        else:
            kernel_divergence_fn = kernel_divergence_fn_gen(num_train)
            bias_divergence_fn = bias_divergence_fn_gen(num_train)
            
            
    train_output_size = num_features + num_lin_dim

    inp_rec = tfk.Input(shape=(num_features))
    
    train_mean_lin = lin_pred_model(inp_lin)
    train_mean_rec = enc_model(inp_rec)
    if(res_lin_pred_model is not None):
        train_mean_lin = train_mean_lin + res_lin_pred_model(inp_lin)
        
    subtracted_lin = tfk.layers.subtract([train_mean_lin, train_mean_rec])
    train_mean_lin_rec = dec_model(train_mean_lin)
    subtracted_rec = tfk.layers.subtract([train_mean_lin_rec, inp_rec])
    
    combined_sub_mean = tfk.layers.Concatenate()([subtracted_lin, subtracted_rec])

    std_lin_model = None
    pred_std_model = None

    if (prob):
        std_lin = tfk.layers.Dense(num_lin_dim, use_bias=False, kernel_initializer='zero', trainable=False)(inp_lin)
        std_lin = tfp.layers.DenseReparameterization(num_lin_dim, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn)(std_lin)
        std_rec = tfk.layers.Dense(num_lin_dim, use_bias=False, kernel_initializer='zero', trainable=False)(inp_lin)
        std_rec = tfp.layers.DenseReparameterization(num_features, 
                           bias_posterior_fn=b_post,
                           bias_prior_fn=b_prior,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn)(std_rec)
        combined_std = tfk.layers.Concatenate()([std_lin, std_rec])

        combined_train_model = tfk.layers.Concatenate()([combined_sub_mean, combined_std])
        z = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :train_output_size], 
                                                       scale= tf.math.softplus(t[..., train_output_size:])
                                                       ))(combined_train_model)

        train_model = tfk.Model(inputs = [inp_lin, inp_rec], outputs = z)
        std_lin_model = tfk.Model(inputs = inp_lin, outputs = std_lin)
        
        if(res_lin_pred_model is None):
            pred = lin_pred_model(inp_lin)
        else:
            pred = lin_pred_model(inp_lin) + res_lin_pred_model(inp_lin)
        pred = dec_model(pred) 

        pred_std_model = tfk.Model(inputs = inp_lin, outputs = std_rec)

        pred_std = pred_std_model(inp_lin)

        pred_comb = tfk.layers.Concatenate()([pred, pred_std])
        z_pred = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :num_features], 
                                                       scale= tf.math.softplus(t[..., num_features:])
                                                       ))(pred_comb)

        pred_model = tfk.Model(inputs = inp_lin, outputs = z_pred)
        
    else:
        train_model = tfk.Model(inputs = [inp_lin, inp_rec], outputs = combined_sub_mean)
        pred_model = tfk.Model(inputs = inp_lin, outputs = train_mean_lin_rec)
        
    
    return train_model, pred_model, std_lin_model, pred_std_model    