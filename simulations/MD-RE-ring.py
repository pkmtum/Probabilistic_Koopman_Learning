import PKRL as pk
import h5py
import numpy as np
from scipy.linalg import expm
import tensorflow.keras as tfk

# Load data

with h5py.File("data/part_sim_ring_r.hdf5", "r") as f:
    timeseries = np.array(f['part'])

num_features = int(np.prod(timeseries.shape[0:2]))
num_obs = timeseries.shape[2]
timeseries = timeseries.reshape(num_features, num_obs).T

# Generate training/test data & standardization
    
ts_data = pk.ts_data(timeseries, has_time = False)
ts_data.standardize()
ts_data.generate_train_model_inputs(1)

# DMD

part_dmd = pk.dmd()
#part_dmd.random_search_stable_svd_dmd(ts_data.train_ts_centered, range(2,20), range(200,3000), num_tries=400)
#part_dmd.grid_search_stable_svd_dmd(ts_data.train_ts_centered, range(13,14), range(1500,1650))
part_dmd.svd_dmd(ts_data.train_ts_centered[:1519,:], 13)
part_dmd.compute_K(extend = 7)
x0 = ts_data.train_ts_centered[0, :-1]
pk.ts_util.plot_ts(part_dmd.predict(x0, ts_data.num_train, type='K'))

# step 1
ts_data.generate_train_model_inputs(10, 0.7)
ts_data.x.shape
pk.ts_util.plot_ts(ts_data.x)
kmodel = pk.koop_model(ts_data, part_dmd, enc_nn_def=[120,100,60,40], dec_nn_def=[40, 60, 100, 120])
kmodel.compile_train_model_adam(learning_rate=0.01)

kmodel.train(epochs=50, batch_dim = 100)

kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/np_md_weights_ring_pre_res")

# step 2

ts_data.generate_train_model_inputs(2, 0.31)
pk.ts_util.plot_ts(ts_data.x), ts_data.x.shape

kmodel = pk.koop_model(ts_data, part_dmd, enc_nn_def=[120,100,60,40], dec_nn_def=[40, 60, 100, 120])
kmodel.compile_train_model_adam(learning_rate=0.001)

kmodel.load_weights("model/np_md_weights_ring_pre_res")
kmodel.train(epochs=30, batch_dim = ts_data.num_train_filtered)

kmodel.pred_data()
kmodel.plot_pred_train()

kmodel.save_weights("model/np_md_weights_ring_res")

# step 3 

pkmodel = pk.koop_model(ts_data, part_dmd, enc_nn_def=[120,100,60,40], dec_nn_def=[40, 60, 100, 120], prob=True)
pkmodel.compile_train_model_adam(learning_rate=0.01)
pkmodel.init_with_mle_weights("model/np_md_weights_ring_res")

pkmodel.train(epochs=500, batch_dim = ts_data.num_train_filtered)

pkmodel.pred_data()
pkmodel.plot_pred_train()
pkmodel.plot_pred_test()
pkmodel.plot_pred_test_forerun()

pkmodel.save_weights("model/p_md_weights_ring_res")

# Compute standard deviations with uncertainties
m = np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3]) @ ts_data.train_std ,4) 
l = np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] - pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)
u = np.round(pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[3] + pk.svd_dmd.softplus(pkmodel.pred_std_model.get_weights()[4])) @ ts_data.train_std,4)
np.save("mlu/mlu-r-res",np.array([m,l,u]))

# Save predictions, ELBO training curve and ELBO distribution
pkmodel.save_sample_preds(2000, "plot_data/MD_res-ring-")
np.save("plot_data/md_res-ring-hist", pkmodel.history.history['loss'])
elbos = [np.mean(pkmodel.train_model.loss(pkmodel.y, pkmodel.train_model(ts_data.x_inp))) for i in range(50)]
np.save("plot_data/md_res-ring-elbos", elbos)