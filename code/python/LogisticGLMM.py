
# coding: utf-8

# In[1]:

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


# In[2]:

#analysis_name = 'simulated_data_small'
analysis_name = 'criteo_subsampled'

git_dir_cmd = subprocess.run(
    ['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
assert git_dir_cmd.returncode == 0
git_dir = git_dir_cmd.stdout.decode("utf-8").strip()

data_dir = os.path.join(git_dir, 'code/data')
json_filename = os.path.join(data_dir, '%s_stan_dat.json' % analysis_name)
y_g_vec, y_vec, x_mat, glmm_par, prior_par = logit_glmm.load_json_data(json_filename)

num_gh_points = 4

timer = obj_lib.Timer()


# In[ ]:

# Initialize.

# Slightly smarter inits would probably improve fit time, but as of now it doesn't
# seem worth explaining in the paper.

logit_glmm.initialize_glmm_pars(glmm_par)
free_par_vec = glmm_par.get_free()
init_par_vec = copy.deepcopy(free_par_vec)
prior_par_vec = prior_par.get_vector()

model = logit_glmm.LogisticGLMM(
    glmm_par, prior_par, x_mat, y_vec, y_g_vec, num_gh_points=num_gh_points)


# In[ ]:

# Optimize.

print('Running Newton Trust Region.')
timer.tic()
vb_opt = model.tr_optimize(init_par_vec, num_gh_points, gtol=1e-6, maxiter=500)
opt_x = vb_opt.x
timer.toc('vb_time')

print('Done.')


# In[ ]:

# Get the Hessians at the number of draws used for optimization
model.glmm_par.set_free(opt_x)

timer.tic()
print('KL Hessian...\n')
kl_hess = model.get_sparse_free_hessian(opt_x, print_every_n=100)

print('Log prior Hessian...\n')
log_prior_hess = model.get_prior_hess(prior_par.get_vector(), opt_x)

moment_jac = model.moment_wrapper.get_moment_jacobian(opt_x)
timer.toc('hess_time')


# In[ ]:

print('Solving systems...\n')
timer.tic()
from scikits.sparse.cholmod import cholesky
kl_hess_chol = cholesky(kl_hess)
kl_inv_moment_jac = kl_hess_chol.solve_A(moment_jac.T)
lrvb_cov = np.matmul(moment_jac, kl_inv_moment_jac)
vb_prior_sens = np.matmul(log_prior_hess, kl_inv_moment_jac).T
timer.toc('inverse_time')
print('Done\n')


# In[ ]:

# Time using conjugate gradient to get a single row of the moment sensitivity.
class OptimumHVP(object):
    def __init__(self, glmm_par, opt_x, moment_jac):
        self.verbose = False
        self.print_every = 10
        self.reset_iter()
        self.opt_x = opt_x
        self.moment_jac = moment_jac
        self.lo = LinearOperator(
            (glmm_par.free_size(), glmm_par.free_size()), self.hvp)
        
    def reset_iter(self):
        self.iter = 0
    
    def hvp(self, vec):
        self.iter += 1
        if self.verbose and self.iter % self.print_every == 0:
            print('Iter ', self.iter)
        return model.objective.fun_free_hvp(self.opt_x, vec)
    
    def get_moment_sensitivity_row(self, moment_row):
        self.reset_iter()
        moment_jac_vec = moment_jac[moment_row, :].flatten()
        cg_res, info = sp.sparse.linalg.cg(self.lo, moment_jac_vec)
        return cg_res, info

moment_row = 0
optimum_hvp = OptimumHVP(glmm_par, opt_x, moment_jac)
optimum_hvp.verbose = True
optimum_hvp.print_every = 20
timer.tic()
cg_res, info = optimum_hvp.get_moment_sensitivity_row(0)
timer.toc('cg_row_time')

num_cg_iterations = optimum_hvp.iter
print('Number of iterations: ', optimum_hvp.iter)

print(np.max(np.abs(cg_res - kl_inv_moment_jac[:, moment_row].flatten())))


# In[ ]:

print(pickle_result_dict)


# In[ ]:

# Write the result to a pickle file for use in subsequent analysis.
model.glmm_par.set_free(opt_x)
model.prior_par.set_vector(prior_par_vec)

run_name = 'production'

pickle_output_filename = os.path.join(data_dir, '%s_python_vb_results.pkl' % analysis_name)
pickle_output = open(pickle_output_filename, 'wb')

# Unlike with JSON, numpy arrays can be pickled directly.
# Note that it does not seem that you can pickle a sparse Cholesky decomposition.
pickle_result_dict = logit_glmm.get_pickle_dictionary(model, kl_hess, moment_jac)
pickle_result_dict.update(
                     { 'run_name': run_name,
                       'vb_time': timer.time_dict['vb_time'],
                       'hess_time': timer.time_dict['hess_time'],
                       'inverse_time': timer.time_dict['inverse_time'],
                       'cg_row_time': timer.time_dict['cg_row_time'],
                       'num_cg_iterations': num_cg_iterations,
                       'lrvb_cov': np.squeeze(lrvb_cov),
                       'kl_inv_moment_jac': kl_inv_moment_jac,
                       'vb_prior_sens': np.squeeze(vb_prior_sens),
                       'log_prior_hess': np.squeeze(log_prior_hess) })

# Pickle dictionary.
pickle.dump(pickle_result_dict, pickle_output)
pickle_output.close()

print(pickle_output_filename)

print('Done.')

