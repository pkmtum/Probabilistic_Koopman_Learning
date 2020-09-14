import PKRL as pk
import h5py
import numpy as np
from scipy.linalg import expm
import tensorflow.keras as tfk

# Generate data set

def g(x):
    return 10*np.array([(x[0] + x[1])**3, (-x[1])])

def g_inv(g):
    y = g/10
    return np.array([np.cbrt(y[0]) + y[0], (-y[1])])

ceta = -0.9

K = np.array([[-0.01, -ceta],[ceta, -0.01]])
x0 = np.array([0.5, 0.1])

num_obs = 2400
num_features = 2
t_scale = 0.025

x = np.zeros((num_obs, num_features+1))
x[0,:num_features] = x0
x[0,-1] = 0

np.random.seed(123)

for i in range(1,num_obs):
    x[i,:num_features] = x0 @ expm(K * i * t_scale) + g_inv(g(x0) @ expm(K * i * t_scale)) + np.random.randn(2)*0.05
    x[i, -1] = i*t_scale

# Generate training/test data & standardization
    
ts_data = pk.ts_data(x, has_time = True)
ts_data.standardize()

# DMD
dmd = pk.dmd()
#dmd.grid_search_stable_svd_dmd(ts_data.train_ts_centered, range(2,3), range(200, ts_data.num_train))

dmd.svd_dmd(ts_data.train_ts_centered[:3000,:], 2)
dmd.compute_K(extend = 0)
x0 = ts_data.train_ts_centered[0, :-1]
pk.ts_util.plot_ts(dmd.predict(x0, ts_data.num_train, type='K'))

# manual setting -> assure identifiability of the parameters
dmd.K = np.zeros((2,2))
dmd.T = np.identity(2)
pk.ts_util.plot_ts(dmd.predict(x0, ts_data.num_train, type='K'))

ts_data.generate_train_model_inputs(10, 0.44)
ts_data.num_train_filtered

np.save( "plot_data/ad_ts", ts_data.ts_norm)

# step 1
kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4, 6, 12, 10,8], dec_nn_def=[8, 10, 12, 6, 4],regu=None, fix_inner_trafos=True)
kmodel.compile_train_model_adam(learning_rate=0.05)

kmodel.train(epochs=100, batch_dim = ts_data.num_train_filtered)

kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/pre-np-ad2")

# step 2

ts_data.generate_train_model_inputs(3, 0.2)
ts_data.num_train_filtered

kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10,12, 6, 4],regu=None, fix_inner_trafos=True)
kmodel.compile_train_model_adam(learning_rate=0.005)

kmodel.load_weights("model/pre-np-ad2")
kmodel.train(epochs=200, batch_dim = 300)

kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/np-ad2")

# step 3 

pkmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10, 12, 6, 4], prob=True, fix_inner_trafos=True)
pkmodel.init_with_mle_weights("model/np-ad2")
pkmodel.compile_train_model_adam(learning_rate=0.01)

pkmodel.train(epochs=700, batch_dim =ts_data.num_train)

pkmodel.pred_data()
pkmodel.plot_pred_train()
pkmodel.plot_pred_test()
pkmodel.plot_pred_test_forerun()

pkmodel.save_weights("model/p-ad2")

# Compute standard deviations with uncertainties
pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3]) @ ts_data.train_std
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] - pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] + pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)

# Compute exp parameters with uncertainties
np.round(pkmodel.exp_model.get_weights()[0][2] ,4)
np.round(pkmodel.exp_model.get_weights()[0][2] + pk.svd_dmd.softplus(pkmodel.exp_model.get_weights()[0][5]),4)
np.round(pkmodel.exp_model.get_weights()[0][2] - pk.svd_dmd.softplus(pkmodel.exp_model.get_weights()[0][5]),4)

# Save predictions, ELBO training curve and ELBO distribution

pkmodel.save_sample_preds(2000, "AD2-")
np.save("ad2-hist", pkmodel.history.history['loss'])
elbos = [np.mean(pkmodel.train_model.loss(pkmodel.y, pkmodel.train_model(ts_data.x_inp))) for i in range(3)]