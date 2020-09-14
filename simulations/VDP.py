import PKRL as pk
import h5py
import numpy as np
from scipy.linalg import expm
import tensorflow.keras as tfk

# Generate data set

def comp_vdp_intern(delta_t, x, eps):
    return x + delta_t * np.array([x[1], -eps*(x[0]**2-1)*x[1]-x[0]])

def comp_vdp(delta_t, x0, eps, num_obs, std_err):
    timeseries = np.empty([len(x0), num_obs])
    
    timeseries[:, 0] = x0
    for i in range(1, num_obs):
        timeseries[:, i] = comp_vdp_intern(delta_t, timeseries[:, i-1], eps)
        timeseries[:, i-1] = timeseries[:, i-1] + np.random.randn(2)*std_err
    return timeseries

num_obs = 6*400*2
t_scale = 0.1/5/2
eps = 0.9
num_features = 2
std_err = 0.1

np.random.seed(123)

x = comp_vdp(t_scale, [2,-1], eps, num_obs, std_err)
x = np.vstack([x, t_scale*np.array(range(num_obs))[np.newaxis, ...]]).transpose()

pk.ts_util.plot_ts(x)

# Generate training/test data & standardization

ts_data = pk.ts_data(x, has_time = True)
ts_data.standardize()
pk.ts_util.plot_ts(ts_data.train_ts)
np.save("plot_data/vdp_ts", ts_data.ts_norm)

# DMD

dmd = pk.dmd()
dmd.random_search_stable_svd_dmd(ts_data.train_ts_centered, range(2,3), range(200, ts_data.num_train), 300)
dmd.svd_dmd(ts_data.train_ts_centered[:526,:], 2)
dmd.compute_K(extend = 0)
x0 = ts_data.train_ts_centered[0, :-1]
pk.ts_util.plot_ts(dmd.predict(x0, ts_data.num_train, type='K'))

# step 1
ts_data.generate_train_model_inputs(10, 0.22)
ts_data.num_train_filtered

kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4, 6, 12, 10,8], dec_nn_def=[8, 10, 12, 6, 4],regu=None)
kmodel.compile_train_model_adam(learning_rate=0.05)
kmodel.train(epochs=100, batch_dim = ts_data.num_train_filtered)

kmodel.pred_data()
kmodel.plot_pred_train()

# step 2

kmodel.save_weights("model/pre-np-vdp")
ts_data.generate_train_model_inputs(3, 0.16)
ts_data.num_train_filtered

kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10,12, 6, 4],regu=None)
kmodel.compile_train_model_adam(learning_rate=0.005)
kmodel.load_weights("model/pre-np-vdp")
kmodel.train(epochs=50, batch_dim = 300)

kmodel.pred_data()
kmodel.plot_pred_train()
kmodel.save_weights("model/np-vdp")

# step 3 

pkmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10, 12, 6, 4], prob=True)
pkmodel.init_with_mle_weights("model/np-vdp")
pkmodel.compile_train_model_adam(learning_rate=0.01)
pkmodel.train(epochs=1000, batch_dim =ts_data.num_train)

pkmodel.pred_data()
pkmodel.plot_pred_train()
pkmodel.plot_pred_test()
pkmodel.plot_pred_test_forerun()

pkmodel.save_weights("model/2p-vdp")

# Compute standard deviations with uncertainties
pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3]) @ ts_data.train_std
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] - pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] + pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)

# Save predictions, ELBO training curve and ELBO distribution

pkmodel.save_sample_preds(2000, "plot_data/2VDP-")
np.save("plot_data/2vdp-hist", pkmodel.history.history['loss'])
elbos = [np.mean(pkmodel.train_model.loss(pkmodel.y, pkmodel.train_model(ts_data.x_inp))) for i in range(50)]
np.save("plot_data/2vdp-elbos", elbos)