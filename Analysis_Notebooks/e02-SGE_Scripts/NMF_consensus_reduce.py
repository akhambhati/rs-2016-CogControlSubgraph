#!/bin/zsh

#
# Make sure that output files arrive in the working directory
#$ -cwd
#
# Join the outputs together
#$ -j y
#
# Reserve 10 GB of memory
#$ -l h_vmem=80.5G,s_vmem=80.0G
#
#Specify which nodes to use
#$ -q himem.q
#

source activate echobase
ipython -c "

######
from __future__ import division

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import glob
import numpy as np

sys.path.append('$1')
import Echobase
Subgraph = Echobase.Network.Partitioning.Subgraph

path_InpData = '$2'
path_ExpData = '$3'
path_Output = '{}/NMF_Consensus.Param.All.npz'.format(path_ExpData)

# Check if the output already exists (can be commented if overwrite is needed)
if os.path.exists(path_Output):
    raise Exception('Output {} already exists'.format(path_Output))

# Gather all NMF_consensus seeds
seed_paths = glob.glob('{}/NMF_Consensus.Param.*.npz'.format(path_ExpData))

# Aggregate the estimated subgraphs of each seed
fac_subnet_seeds = []
for ii, path in enumerate(seed_paths):
    data = np.load(path, mmap_mode='r')
    fac_subnet = data['fac_subnet'][:, :]
    data.close()

    n_fac = fac_subnet.shape[0]
    n_conn = fac_subnet.shape[1]

    for iy in xrange(fac_subnet.shape[0]):
        fac_subnet_seeds.append(fac_subnet[iy, :])
fac_subnet_seeds = np.array(fac_subnet_seeds)

n_obs = fac_subnet_seeds.shape[0]
n_conn = fac_subnet_seeds.shape[1]

sys.stdout.write('\nAggregated {} seed subgraphs...'.format(n_obs))

# Consensus Subgraphs
sys.stdout.write('\nFinding consensus subgraphs...')
fac_cons_subnet, fac_cons_seeds, err = Subgraph.nmf.snmf_bcd(
    fac_subnet_seeds,
    alpha=0.0,
    beta=0.0,
    fac_subnet_init=np.random.uniform(low=0.0, high=1.0, size=(n_fac, n_conn)),
    fac_coef_init=np.random.uniform(low=0.0, high=1.0, size=(n_fac, n_obs)),
    max_iter=100, verbose=False)


# Consensus Coefficients
sys.stdout.write('\nCalculating subgraph expression coefficients...')
cfg_data_path = '{}/Population.Configuration_Matrix.Norm.npz'.format(path_InpData)
data_cfg = np.load(cfg_data_path, mmap_mode='r')
n_win = data_cfg['cfg_matr'].shape[0]
fac_cons_subnet_2, fac_cons_coef_2, err = Subgraph.nmf.snmf_bcd(
    data_cfg['cfg_matr'],
    alpha=0.0,
    beta=0.0,
    fac_subnet_init=fac_cons_subnet,
    fac_coef_init=np.random.uniform(low=0.0, high=1.0, size=(n_fac, n_win)),
    max_iter=100, verbose=False)

# Cache the Consensus NMF result
np.savez(path_Output,
         fac_subnet=fac_cons_subnet_2,
         fac_coef=fac_cons_coef_2,
         err=err)
######

"
