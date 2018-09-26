#!/usr/bin/env python3

import copy
import json

import LinearResponseVariationalBayes as vb
import logistic_glmm_lib as logit_glmm
import LinearResponseVariationalBayes.SparseObjectives as obj_lib


import numpy as np
import os
import pickle

from scikits.sparse.cholmod import cholesky
import scipy as sp
from scipy.sparse.linalg import LinearOperator

import subprocess

#analysis_name = 'simulated_data_small'
analysis_name = 'criteo_subsampled'

git_dir_cmd = subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
assert git_dir_cmd.returncode == 0
git_dir = git_dir_cmd.stdout.decode("utf-8").strip()

data_dir = os.path.join(git_dir, 'code/data')

pickle_output_filename = os.path.join(
    data_dir, '%s_python_vb_results.pkl' % analysis_name)

pkl_file = open(pickle_output_filename, 'rb')
vb_results = pickle.load(pkl_file)


# In[ ]:

json_filename = os.path.join(data_dir, '%s_stan_dat.json' % analysis_name)
y_g_vec, y_vec, x_mat, glmm_par, prior_par = logit_glmm.load_json_data(json_filename)

K = x_mat.shape[1]
NG = np.max(y_g_vec) + 1

# Define moment parameters
moment_wrapper = logit_glmm.MomentWrapper(glmm_par)
glmm_par = logit_glmm.get_glmm_parameters(K=K, NG=NG)


# In[ ]:

mle_par = logit_glmm.get_mle_parameters(K=K, NG=NG)
glmm_par = logit_glmm.get_glmm_parameters(K=K, NG=NG)
glmm_par.set_free(vb_results['glmm_par_free'])

# Set from VB
mle_par['beta'].set(glmm_par['beta'].e())
mle_par['mu'].set(glmm_par['mu'].e())
mle_par['tau'].set(glmm_par['tau'].e())
mle_par['u'].set(glmm_par['u'].e())

model = logit_glmm.LogisticGLMMMaximumLikelihood(mle_par, prior_par, x_mat, y_vec, y_g_vec)
objective = Objective(fun=model.get_log_loss, par=model.mle_par)

u_free_init = np.random.random(model.mle_par.free_size())
objective.logger.initialize()
objective.logger.print_every = 20
map_time = time.time()
mle_opt = optimize.minimize(
    lambda par: objective.fun_free(par, verbose=True),
    x0=u_free_init,
    method='trust-ncg',
    jac=objective.fun_free_grad,
    hessp=objective.fun_free_hvp,
    tol=1e-6, options={'maxiter': 200, 'disp': True, 'gtol': 1e-6 })
map_time = time.time() - map_time
free_opt = mle_opt.x
model.mle_par.set_free(free_opt)


# In[ ]:
model.mle_par.set_free(free_opt)
mle_moment_par = logit_glmm.set_moment_par_from_mle(moment_wrapper.moment_par, model.mle_par)
mle_moment_par_vector = copy.deepcopy(mle_moment_par.get_vector())

# Write the result to a JSON file for use in R.
pickle_output_filename = os.path.join(data_dir, '%s_python_vb_map_results.pkl' % analysis_name)
pickle_output = open(pickle_output_filename, 'wb')

# Unlike with JSON, numpy arrays can be pickled.
model.mle_par.set_free(free_opt)
pickle_result_dict = {  'pickle_output_filename': pickle_output_filename,
                        'map_time': map_time,
                        'mle_moment_par_vector': mle_moment_par_vector,
                        'mle_par_dictval': model.mle_par.dictval(),
                        'mle_par_vector': model.mle_par.get_vector(),
                        'mle_par_free': model.mle_par.get_free()
                     }

# Pickle dictionary using protocol 0.
pickle.dump(pickle_result_dict, pickle_output)
pickle_output.close()

print(pickle_output_filename)

print('\n\nDone.')
