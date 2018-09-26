import LinearResponseVariationalBayes as vb
import LinearResponseVariationalBayes.SparseObjectives as obj_lib
from autograd import numpy as np
import pandas as pd

import scipy as sp
import numpy as onp

import multivariate_model_lib as mml

def to_scalar(x):
    if hasattr(x, "__len__"):
        assert len(x) == 1
        return x[0]
    else:
        return x

assert to_scalar(5.0) == 5
assert to_scalar([5.0]) == 5
assert to_scalar(np.array([5.0])) == 5


def get_mixture_parameters(dim, num_components):
    mix_par = vb.ModelParamsDict()
    mix_par.push_param(
        vb.ArrayParam('loc', shape=(num_components, dim)))
    mix_par.push_param(vb.PosDefMatrixParamArray(
        'info', array_shape=(num_components, ), matrix_size=dim))
    mix_par.push_param(vb.SimplexParam('w', shape=(1, num_components)))
    return mix_par


def get_marginal_mixture_parameters(mix_par, margin):
    v = mix_par.values
    num_components = v['loc'].shape[0]
    marginal_mix_par = get_mixture_parameters(
        dim=1, num_components=num_components)

    mix_cov = [ np.linalg.inv(v['info'][k, :, :]) \
        for k in range(num_components) ]
    marginal_info = np.array([ [[1 / mix_cov[k][margin, margin]]] \
        for k in range(num_components) ])
    marginal_mix_par['loc'].set(np.expand_dims(v['loc'][:, margin], 1))
    marginal_mix_par['info'].set(marginal_info)
    marginal_mix_par['w'].set(v['w'])

    return marginal_mix_par


class MixtureModel(object):
    def __init__(self, num_gh_points=50, dim=2,
                 num_components=3, num_draws=10000):

        self.mix_par = get_mixture_parameters(
            dim=dim, num_components=num_components)
        self.moment_par = mml.get_moment_params(dim)
        self.true_var = mml.get_moment_params(dim)
        self.dim = dim
        self.num_components = num_components

        self.set_monte_carlo_draws(num_draws)
        self.zero_tilt()

        self.q = mml.FactorizingNormalApproximation(dim=dim, num_draws=1000)
        self.map_approx = mml.MultivariateMAPApproximation(dim=dim)

        self.kl_objective = obj_lib.Objective(self.q.par, self.get_kl)
        self.map_objective = obj_lib.Objective(
            self.map_approx.map_par, self.map_loss_function)

    def zero_tilt(self):
        # Always run set_moments_par after changing the tilting.
        self.t_x = np.zeros((self.dim, ))
        self.t_expx = np.zeros((self.dim, ))
        self.set_moments_par()

    # Log probability at x, which should be n x p.
    def get_log_prob_nk(self, x):
        # Up to a constant.
        v = self.mix_par.values
        x_centered = np.expand_dims(x, 2) - np.expand_dims(np.transpose(v['loc']), 0)
        log_det_k = np.array(
            [ np.linalg.slogdet(v['info'][k,:,:])[1] \
            for k in range(self.num_components) ])
        lp_nk = \
            -0.5 * np.einsum('nik,kij,njk->nk',
                             x_centered, v['info'], x_centered) + \
            0.5 * np.expand_dims(log_det_k, 0)
        return lp_nk

    def get_log_prob_n(self, x):
        lp_nk = self.get_log_prob_nk(x)

        # Prevent underflows in the weights by doing logsumexp ourself.
        v = self.mix_par.values
        lp_n_max = np.amax(lp_nk, axis=1)
        lp_n = np.log(np.einsum(
            'nk,k->n',
            np.exp(lp_nk - np.expand_dims(lp_n_max, 1)),
            v['w'][0,:])) + lp_n_max
        return lp_n

    def get_log_prob(self, x, use_tilt=True):
        lp_n = self.get_log_prob_n(x)
        if use_tilt:
            tilt = np.einsum('np,p', x, self.t_x) + \
                   np.einsum('np,p', np.exp(x), self.t_expx)
        else:
            tilt = 0
        return lp_n + tilt

    def map_loss_function(self):
        mode = np.expand_dims(self.map_approx.map_par.get(), 0)
        return -1 * self.get_log_prob(mode)[0]

    def get_kl(self):
        return -1 * (self.q.estimate_expectation(self.get_log_prob)[0] + \
                     self.q.get_entropy())

    def get_untilted_draws(self, num_draws):
        v = self.mix_par.values
        z = sp.stats.multinomial.rvs(n=num_draws, p=v['w'][0, :], size=1)[0]
        covs = [ np.linalg.inv(v['info'][k, :, :]) \
                 for k in range(self.num_components)]
        means = [ v['loc'][k, :] for k in range(self.num_components) ]

        draws = [ np.array(sp.stats.multivariate_normal.rvs(
                      means[k], cov=covs[k], size=z[k])) \
                  for k in range(self.num_components)]

        # Oh, numpy.  :(
        if self.dim == 1:
            draws = [ np.expand_dims(d, 1) for d in draws]

        return np.vstack(draws)

    def set_monte_carlo_draws(self, num_draws):
        self.draws = self.get_untilted_draws(num_draws)

    def get_function_mean_and_var(self, fun):
        lp_base = self.get_log_prob(self.draws, use_tilt=False)
        lp = self.get_log_prob(self.draws)

        log_weights = lp - lp_base
        log_weights = log_weights - sp.special.logsumexp(log_weights)

        weights = np.expand_dims(np.exp(log_weights), 1)
        draw_f = fun(self.draws)
        e_f = np.sum(weights * draw_f, axis=0)
        e_f2 = np.sum(weights * (draw_f ** 2), axis=0)

        return e_f, e_f2 - e_f ** 2, log_weights

    def set_moments_par(self):
        e_x, var_x, log_weights = \
            self.get_function_mean_and_var(lambda x: x)
        eff_num_samples_x = 1 / np.sum(np.exp(2 * log_weights))
        e_expx, var_expx, log_weights = \
            self.get_function_mean_and_var(lambda x: np.exp(x))
        eff_num_samples_expx = 1 / np.sum(np.exp(2 * log_weights))

        self.moment_par['x'].set(e_x)
        self.true_var['x'].set(var_x)

        #self.true_var['expx'].set(var_expx)
        #self.moment_par['expx'].set(e_expx)

        return eff_num_samples_x, eff_num_samples_expx

    def optimize_q(self):
        vb_opt_res = sp.optimize.minimize(
            fun=self.kl_objective.fun_free,
            x0=self.q.par.get_free(),
            jac=self.kl_objective.fun_free_grad,
            hess=self.kl_objective.fun_free_hessian,
            method='trust-exact')
        self.q.par.set_free(vb_opt_res.x)
        kl_hess = self.kl_objective.fun_free_hessian(vb_opt_res.x)
        self.q.set_moments()
        self.q.set_moment_variances(kl_hess)
        return vb_opt_res

    def optimize_map(self):
        map_opt_res = sp.optimize.minimize(
            fun=self.map_objective.fun_free,
            x0=self.map_approx.map_par.get_free(),
            jac=self.map_objective.fun_free_grad,
            hess=self.map_objective.fun_free_hessian,
            method='trust-exact')
        self.map_approx.map_par.set_free(map_opt_res.x)
        self.map_approx.set_moments(self.map_objective)
        return map_opt_res

    def get_moment_df(self):
        return pd.concat([
            mml.get_moment_parameter_df(self.moment_par, method='truth', metric='mean'),
            mml.get_moment_parameter_df(self.true_var, method='truth', metric='var')
        ])

    def get_combined_moment_df(self):
        df = pd.concat([
            self.get_moment_df(),
            self.q.get_moment_df(),
            self.map_approx.get_moment_df()
        ])
        # Necessary to convert to an R dataframe.
        df.index = range(df.shape[0])
        return df

    def get_map_log_prob(self, x):
        return self.map_approx.get_map_approx(x, self.map_objective)

    def get_map_marginal_log_prob(self, x, margin):
        return self.map_approx.get_marginal_map_approx(
            x, self.map_objective, margin)



################
# Set up some interesting problems


def get_rotation_mat(theta):
    return(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))

def rotate_mat(mat, theta):
    rot_mat = get_rotation_mat(theta)
    return np.matmul(rot_mat, np.matmul(mat, np.linalg.inv(rot_mat)))

def set_mix_par_3comp(model, delta, sd_scale, sd_narrowness, w_diff):
    m = np.zeros((3, 2))
    m[0, 0:2] = np.array([-delta, -delta])
    m[1, 0:2] = np.array([0, 0])
    m[2, 0:2] = np.array([delta, delta])

    base_cov_mat = rotate_mat(
        np.diag([1, sd_narrowness ** 2]) * (sd_scale * delta) ** 2, np.pi / 4)
    print(base_cov_mat)
    base_info_mat = np.linalg.inv(base_cov_mat)

    w_vec = np.array([[(1/3) - 0.5 * w_diff, 1/3 + w_diff, (1/3) - 0.5 * w_diff]])

    model.mix_par['loc'].set(m)
    model.mix_par['info'].set(np.array([ base_info_mat for k in range(3)]))
    model.mix_par['w'].set(w_vec)
    model.set_moments_par()
