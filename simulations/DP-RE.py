import PKRL as pk
import h5py
import numpy as np
from scipy.linalg import expm
import tensorflow.keras as tfk

# Load data

with h5py.File("dp_sim.hdf5", "r") as f:
    timeseries = np.array(f['dp'])

num_obs = timeseries.shape[0]
features = range(2)
num_features = len(features)
x = pk.ts_util.get_po_ts(timeseries, num_obs, features)

# Generate training/test data & standardization

ts_data = pk.ts_data(x, has_time = True)
ts_data.standardize()
pk.ts_util.plot_ts(ts_data.train_ts)

# DMD

dmd = pk.dmd()
#dmd.random_search_stable_svd_dmd(ts_data.train_ts_centered, range(2,6), range(200, ts_data.num_train), 300)
dmd.svd_dmd(ts_data.train_ts_centered[:223,:], 2)
dmd.compute_K(extend = 5)
x0 = ts_data.train_ts_centered[0, :-1]
pk.ts_util.plot_ts(dmd.predict(x0, ts_data.num_train, type='K'))
ts_data.generate_train_model_inputs(10, 0.15)
ts_data.num_train_filtered

# step 1

kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4, 6, 12, 10, 8], dec_nn_def=[8, 10, 12, 6, 4],regu=None)
kmodel.compile_train_model_adam(learning_rate=0.01)
kmodel.train(epochs=500, batch_dim = 100)

kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/pre-np-dp-res")

# step 2

ts_data.generate_train_model_inputs(3, 0.07)
ts_data.num_train_filtered

kmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10,12, 6, 4],regu=None)
kmodel.compile_train_model_adam(learning_rate=0.005)
kmodel.load_weights("model/pre-np-dp-res")

kmodel.train(epochs=500, batch_dim = 300)
kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/np-dp-res")

# step 3 

pkmodel = pk.koop_model(ts_data, dmd, enc_nn_def=[4,6,12,10,8], dec_nn_def=[8, 10, 12, 6, 4], prob=True)
pkmodel.init_with_mle_weights("model/np-dp-res")
pkmodel.compile_train_model_adam(learning_rate=0.01)

pkmodel.train(epochs=1000, batch_dim =ts_data.num_train)

pkmodel.pred_data()
pkmodel.plot_pred_train()
pkmodel.plot_pred_test()
pkmodel.plot_pred_test_forerun()

pkmodel.save_weights("model/p-dp-res")

# Compute standard deviations with uncertainties

pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3]) @ ts_data.train_std
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] - pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)
np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] + pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)

# Save predictions, ELBO training curve and ELBO distribution

pkmodel.save_sample_preds(2000, "plot_data/DP-RES-")
np.save("plot_data/dp-hist-res", pkmodel.history.history['loss'])
elbos = [np.mean(pkmodel.train_model.loss(pkmodel.y, pkmodel.train_model(ts_data.x_inp))) for i in range(50)]
np.save("plot_data/dp-elbos-res", elbos)