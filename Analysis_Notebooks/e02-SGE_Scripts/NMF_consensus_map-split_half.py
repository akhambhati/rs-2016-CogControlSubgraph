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

# Load the configuration matrix and optimal parameter set
cfg_data = np.load('{}/Population.Configuration_Matrix.Norm.npz'.format(path_InpData))
cfg_matr_full = cfg_data['cfg_matr']
cfg_obs_lut = cfg_data['cfg_obs_lut']
proc_item = np.load('{}/NMF_CrossValidation.Optimal_Param.npz'.format(path_ExpData))['opt_param'][()]

# Divide the blocks in half
split_grp = {'A': np.array(cfg_obs_lut[..., :3].reshape(-1), dtype=int),
             'B': np.array(cfg_obs_lut[..., 3:].reshape(-1), dtype=int)}

for grp_id in split_grp.keys():
    path_Output = '{}/NMF_Consensus.Param.{}.{}.npz'.format(path_ExpData, grp_id, sge_task_id)
    # Check if the output already exists (can be commented if overwrite is needed)
    if os.path.exists(path_Output):
        raise Exception('Output {} already exists'.format(path_Output))
    
    cfg_matr = cfg_matr_full[split_grp[grp_id], :]

    # Grab the task ID of the current job (and the associated parameter dictionary)
    fac_subnet = np.random.uniform(low=0, high=1.0,
                                   size=(proc_item['rank'],
                                         cfg_matr.shape[1]))
    fac_coef = np.random.uniform(low=0, high=1.0,
                                 size=(proc_item['rank'],
                                       cfg_matr.shape[0]))

    # Run NMF Algorithm
    fac_subnet, fac_coef, err = Subgraph.nmf.snmf_bcd(
        cfg_matr,
        alpha=proc_item['alpha'],
        beta=proc_item['beta'],
        fac_subnet_init=fac_subnet,
        fac_coef_init=fac_coef,
        max_iter=100, verbose=False)

    # Cache the NMF result
    np.savez(path_Output,
             fac_subnet=fac_subnet,
             fac_coef=fac_coef,
             err=err)
######

"
