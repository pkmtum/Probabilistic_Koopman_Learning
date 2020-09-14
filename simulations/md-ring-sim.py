import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import h5py

m = 1

l_a = 1
l_r = 0.5

C_a = 1
C_r = 0.5

alpha = 1
beta = 0.5

N = 40
D = 2

num_steps = 4000

np.random.seed(3) 

# q_prime = f(q)
z = np.zeros([3*N, D, num_steps]) #[x,v,a] = [x, f] = [q, a]

x = z[:N, :, :]
v = z[N:2*N, :, :]
a = z[2*N:, :, :]

q = z[:2*N, :, :] # q = [x,v]
f = z[N:, :, :]   # f = [v,a]

id_all = np.arange(N)
idx_all_ex1 = [id_all != i for i in id_all]

def grad_morse(i, step = 0):
    dx = x[i, :, step] - x[idx_all_ex1[i], :, step]
    norm_dx = np.sqrt(np.sum((dx)**2, 1))

    gradU_r = -np.exp(-norm_dx/l_r) / (l_r * norm_dx) @ dx
    gradU_a = -np.exp(-norm_dx/l_a) / (l_a * norm_dx) @ dx

    return  C_r * gradU_r - C_a * gradU_a

def compute_a(step = 0):
    for i in range(N):
        a[i,:,step] = ((alpha - beta*np.sum(v[i,:,step]**2)) * v[i,:,step] - grad_morse(i, step))/m
        
h = 0.01

q[:,:, 0] = np.random.rand(2*N, D)

# Euler
compute_a()  
q[:,:, 1] =  q[:,:, 0] + h*f[:,:, 0]
# AB2
if(num_steps > 1):
    compute_a(1)
    q[:,:, 2] =  q[:,:, 1] + h*(3/2*f[:,:, 1] -1/2*f[:,:, 0])
# AB3
if(num_steps > 2):
    compute_a(2)
    q[:,:, 3] =  q[:,:, 2] + h*(23/12*f[:,:, 2] -16/12*f[:,:, 1] + 5/12*f[:,:, 0])
# AB4
for i in range(3, num_steps-1):
    compute_a(i)
    q[:,:, i+1] =  q[:,:, i] + h*(55/24*f[:,:, i] -59/24*f[:,:, i-1] + 37/24*f[:,:, i-2] -9/24*f[:,:, i-3])

with h5py.File("data/part_sim_ring_r.hdf5", "w") as f:
    dset = f.create_dataset("part", data = x)