#!/bin/zsh

#
# Make sure that output files arrive in the working directory
#$ -cwd
#
# Join the outputs together
#$ -j y
#
# Reserve 10 GB of memory
#$ -l h_vmem=10.5G,s_vmem=10.0G
#
#Specify which nodes to use
#$ -q all.q,basic.q
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

sge_task_id = int(os.environ['SGE_TASK_ID'])-1

sys.path.append('$1')
import Echobase
Subgraph = Echobase.Network.Partitioning.Subgraph

path_InpData = '$2'
path_ExpData = '$3'
path_Output = '{}/NMF_Surrogate.Param.{}.npz'.format(path_ExpData, sge_task_id)

# Check if the output already exists (can be commented if overwrite is needed)
if os.path.exists(path_Output):
    raise Exception('Output {} already exists'.format(path_Output))

# Load the configuration matrix
cfg_data = np.load('{}/Population.Configuration_Matrix.Norm.npz'.format(path_InpData))
cfg_matr = cfg_data['cfg_matr']

# Load the NMF Consensus subgraphs
df_cons = np.load('{}/NMF_Consensus.Param.All.npz'.format(path_ExpData))
fac_subnet_true = df_cons['fac_subnet'].copy()
rank = fac_subnet_true.shape[0]

# Generate a surrogate subgraph set
fac_subnet_surr = np.dot(np.random.rand(rank, rank), fac_subnet_true)
fac_subnet_surr = (fac_subnet_surr.T / np.linalg.norm(fac_subnet_surr, axis=1)).T

# Randomly initialize the coefficients before NNLS
fac_coef = np.random.uniform(low=0, high=1.0,
                             size=(rank,
                                   cfg_matr.shape[0]))

# Run NMF Algorithm
_, fac_coef, err = Subgraph.nmf.snmf_bcd(
    cfg_matr,
    alpha=0.0,
    beta=0.0,
    fac_subnet_init=fac_subnet_surr,
    fac_coef_init=fac_coef,
    max_iter=1, verbose=False)

# Cache the NMF result
np.savez(path_Output,
         fac_subnet=fac_subnet_surr,
         fac_coef=fac_coef,
         err=err)
######

"
