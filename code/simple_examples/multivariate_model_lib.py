import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
from autograd import numpy as np

import pandas as pd
import scipy as sp
import numpy as onp

def get_moment_params(dim):
    moment_par = vb.ModelParamsDict('moments')
    moment_par.push_param(vb.VectorParam('x', size=dim))
    #moment_par.push_param(vb.VectorParam('expx', size=dim))
    return moment_par

def get_normal_variance(norm_mean, norm_var, fun, sims=1000000):
    draws = sp.stats.norm.rvs(loc=norm_mean, scale=np.sqrt(norm_var), size=sims)
    return(np.var(fun(draws)))

assert(np.abs(get_normal_variance(0, 2, lambda x: x) - 2) < 1e-2)
assert(np.abs(get_normal_variance(1, 2, lambda x: x) - 2) < 1e-2)

def get_moment_parameter_df(
    moment_par, margin=None, varname='x', metric='mean', method='none'):

    if margin is None:
        parval = moment_par[varname].get()
        dims = np.arange(0, len(parval))
    else:
        parval = [ to_scalar(moment_par[varname].get()[margin]) ]
        dims = [ margin ]

    return pd.DataFrame(data={
        'value': parval,
        'metric': np.full(len(parval), metric),
        'method': np.full(len(parval), method),
        'dims': dims })


def get_expx_variance(norm_mean, norm_var):
    return (np.exp(norm_var) - 1) * np.exp(2 * norm_mean + norm_var)


def get_marginal_vb_parameters(vb_par, margin):
    v = vb_par.values
    marginal_vb_par = get_vb_parameters(dim=1)
    marginal_vb_par['mean'].set(v['mean'][margin])
    marginal_vb_par['log_var'].set(v['log_var'][margin])
    return marginal_vb_par



def get_vb_parameters(dim):
    par = vb.ModelParamsDict('par')
    par.push_param(vb.VectorParam('mean', size=dim))
    par.push_param(vb.VectorParam('log_var', size=dim))
    return par


class FactorizingNormalApproximation(object):
    def __init__(self, dim, num_draws=100000):
        self.dim = dim

        self.par = get_vb_parameters(dim)
        self.moment_par = get_moment_params(dim)
        self.lrvb_var = get_moment_params(dim)
        self.mfvb_var = get_moment_params(dim)
        self.set_moments()

        self.moment_converter = obj_lib.ParameterConverter(
            self.par, self.moment_par, self.set_moments)

        self.set_draws(num_draws)

    def set_draws(self, num_sims):
        # Keep a different set of draws for each dimension.
        self.draws = sp.stats.norm.rvs(loc=0, scale=1, size=(num_sims, self.dim))

    def estimate_expectation(self, fun):
        # Estimate an intractable expectation with Monte Carlo.
        v = self.par.values
        scale = np.expand_dims(np.exp(0.5 * v['log_var']), 0)
        draws = self.draws * scale + np.expand_dims(v['mean'], 0)
        fun_draws = fun(draws)
        return np.mean(fun(draws)), np.std(fun(draws)) / np.sqrt(self.draws.shape[0])

    def get_log_prob(self, x):
        log_var = self.par['log_var'].get()
        mean = self.par['mean'].get()
        lp_mat = (-0.5 / np.exp(log_var)) * (x - mean) ** 2 - 0.5 * log_var
        return np.sum(lp_mat, axis=1)

    def get_entropy(self):
        # Up to a constant.
        return 0.5 * np.sum(self.par['log_var'].get())

    def get_variance(self):
        return np.exp(self.par['log_var'].get())

    def set_moments(self):
        mean = self.par['mean'].get()
        variance = self.get_variance()
        self.moment_par['x'].set(mean)
        #self.moment_par['expx'].set(np.exp(mean + 0.5 * variance))

    # The objective hessian should be the KL -- i.e., it should be positive definite.
    def get_lrvb_cov(self, obj_hess):
        moment_jac = self.moment_converter.free_to_vec_jacobian(self.par.get_free())
        return np.matmul(moment_jac, np.linalg.solve(obj_hess, np.transpose(moment_jac)))

    def set_moment_variances(self, obj_hess):
        # LRVB variances
        lrvb_cov = self.get_lrvb_cov(obj_hess)
        self.lrvb_var.set_vector(np.diag(lrvb_cov))

        # MFVB variances
        mean = self.par['mean'].get()
        variance = self.get_variance()
        self.mfvb_var['x'].set(variance)
        #self.mfvb_var['expx'].set(get_expx_variance(mean, variance))

    def get_moment_df(self):
        return pd.concat([
            get_moment_parameter_df(self.moment_par, method='mfvb', metric='mean'),
            get_moment_parameter_df(self.lrvb_var, method='lrvb', metric='var'),
            get_moment_parameter_df(self.mfvb_var, method='mfvb', metric='var')
        ])


class MultivariateMAPApproximation(object):
    def __init__(self, dim):
        self.dim = dim
        self.map_par = vb.VectorParam('mode', size=dim)

        # It helps to avoid bad function approximations to not start right at zero.
        self.map_par.set_free(np.full(self.dim, 0.01))

        self.moment_par = get_moment_params(dim)
        self.map_var = get_moment_params(dim)

    # The objective is a loss (i.e. something we minimize) so we expect the Hessian
    # to be positive definite at the optimum.
    def get_hessian(self, x, objective):
        vec_par = self.map_par.get_vector()
        grad = objective.fun_vector_grad(vec_par)
        if np.linalg.norm(grad) > 1e-6:
            print(
             'Warning: The gradient is not zero at the MAP parameters: %f' % \
                np.linalg.norm(grad))
        return objective.fun_vector_hessian(vec_par)

    # Returns the MAP approximation at x to the log probability at the current optimum.
    def get_map_approx(self, x, objective):
        loc = self.map_par.get()
        hess = self.get_hessian(x, objective)
        x_centered = (x - np.transpose(loc))
        return -0.5 * np.einsum('ni,ij,nj->n', x_centered, hess, x_centered)

    def get_marginal_map_approx(self, x, objective, margin):
        loc = self.map_par.get()[margin]
        hess = self.get_hessian(x, objective)
        map_var = np.linalg.inv(hess)[margin, margin]

        x_centered = (x - loc)
        return -0.5 * (x_centered ** 2) / map_var

    def get_variance(self, objective):
        loc = self.map_par.get()
        hess = self.get_hessian(loc, objective)
        return np.linalg.inv(hess)

    def set_moments(self, objective):
        loc = self.map_par.get()
        var = np.diag(self.get_variance(objective))
        self.moment_par['x'].set(loc)
        self.map_var['x'].set(var)

    def get_moment_df(self):
        return pd.concat([
            get_moment_parameter_df(self.moment_par, method='map', metric='mean'),
            get_moment_parameter_df(self.map_var, method='map', metric='var')
        ])

        # self.moment_par['expx'].set(np.exp(loc + 0.5 * var))
        # self.map_var['expx'].set(get_expx_variance(loc, var))
