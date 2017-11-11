# Helper functions for analyzing the Logit LRVB results in R.

library(reticulate)
library(purrr)
library(dplyr)

RecursiveUnpackParameter <- function(par, level=0, id_name="par") {
  if (is.numeric(par)) {
    if (length(par) == 1) {
      return(tibble(val=par))
    } else {
      return(tibble(val=par, component=1:length(par)))
    }
  } else if (is.list(par)) {
    next_level <- map(par, RecursiveUnpackParameter, level + 1)
    return(bind_rows(next_level, .id=paste(id_name, level + 1, sep="_")))
  }
}


# Useful for constructing readable python commands using R variables.
`%_%` <- function(x, y) { paste(x, y, sep="")}

InitializePython <- function(git_repo_loc=Sys.getenv("GIT_REPO_LOC")) {
  py_run_string("import sys")
  py_run_string("import pickle")
  for (py_lib in c("LinearResponseVariationalBayes.py",
                   "VariationalBayesPythonWorkbench/Models/LogisticGLMM/",
                   "autograd")) {
    py_run_string("sys.path.append('" %_% file.path(git_repo_loc, py_lib) %_% "')")
  }
  py_run_string("import LinearResponseVariationalBayes as vb")
  py_run_string("import LogisticGLMM_lib as logit_glmm")
}


ConvertPythonMomentVectorToDF <- function(moment_vector, glmm_par) {
  local_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  local_moment_par$moment_par$set_vector(array(moment_vector))
  return(
    RecursiveUnpackParameter(local_moment_par$moment_par$dictval()) %>%
      rename(par=par_1))
}

ConvertStanVectorToDF <- function(
    stan_vec, param_names, glmm_par, py_main=reticulate::import_main()) {

  k <- glmm_par$param_dict$beta$size()
  ng <- glmm_par$param_dict$u$size()

  beta_colnames <- sprintf("beta[%d]", 1:k)
  u_colnames <- sprintf("u[%d]", 1:ng)

  beta <- stan_vec[beta_colnames]
  mu <- stan_vec["mu"]
  tau <- stan_vec["tau"]
  log_tau <- stan_vec["log_tau"]
  u <- stan_vec[u_colnames]

  mcmc_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  mcmc_param_dict <- mcmc_moment_par$moment_par$param_dict

  mcmc_param_dict$e_beta$set(array(beta))
  mcmc_param_dict$e_mu$set(mu)
  mcmc_param_dict$e_tau$set(tau)
  mcmc_param_dict$e_log_tau$set(log_tau)
  mcmc_param_dict$e_u$set(array(u))

  return(ConvertPythonMomentVectorToDF(mcmc_moment_par$moment_par$get_vector(), glmm_par))
}

ConvertGlmerMeanResultToDF <- function(
  glmer_list, glmm_par, py_main=reticulate::import_main()) {

  k <- length(glmer_list$beta_mean)
  ng <- length(glmer_list$u_map)

  beta <- glmer_list$beta_mean
  mu <- glmer_list$mu_mean
  tau <- glmer_list$tau_mean
  # Transforms of MLEs are the MLEs of transforms.
  log_tau <- log(tau)
  u <- glmer_list$u_map

  glmer_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  glmer_param_dict <- glmer_moment_par$moment_par$param_dict

  glmer_param_dict$e_beta$set(array(beta))
  glmer_param_dict$e_mu$set(mu)
  glmer_param_dict$e_tau$set(tau)
  glmer_param_dict$e_log_tau$set(log_tau)
  glmer_param_dict$e_u$set(array(u))

  return(ConvertPythonMomentVectorToDF(
      glmer_moment_par$moment_par$get_vector(), glmm_par))
}

ConvertGlmerSDResultToDF <- function(
  glmer_list, glmm_par, py_main=reticulate::import_main()) {

  k <- length(glmer_list$beta_mean)
  ng <- length(glmer_list$u_map)

  beta <- glmer_list$beta_sd
  mu <- glmer_list$mu_sd

  # Technically, this is an estimate of the posterior standard deviation
  # of u conditional on mu, beta, and tau.  Applying the law of total variance
  # would amount to doing something much like LRVB.
  u <- glmer_list$u_cond_sd

  glmer_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  glmer_param_dict <- glmer_moment_par$moment_par$param_dict

  glmer_param_dict$e_beta$set(array(beta))
  glmer_param_dict$e_mu$set(mu)
  glmer_param_dict$e_u$set(array(u))

  # As far as I know, glmer doesn't report the standard errors of the
  # random effect variance.
  glmer_param_dict$e_tau$set(0)
  glmer_param_dict$e_log_tau$set(0)

  glmer_sd_df <- ConvertPythonMomentVectorToDF(
      glmer_moment_par$moment_par$get_vector(), glmm_par)

  # Make the tau sds NA
  glmer_sd_df[grepl("tau", glmer_sd_df$par), "val"] <- NA
  return(glmer_sd_df)
}

GetMFVBCovVector <- function(glmm_par) {
  cov_moment_par <- py_main$logit_glmm$MomentWrapper(glmm_par)
  cov_param_dict <- cov_moment_par$moment_par$param_dict
  cov_moment_par$moment_par$set_vector(array(Inf, cov_moment_par$moment_par$vector_size()))

  beta_cov <- 1 / glmm_par$param_dict$beta$param_dict$info$get()
  cov_param_dict$e_beta$set(array(beta_cov))

  cov_param_dict$e_mu$set(1.0 / glmm_par$param_dict$mu$param_dict$info$get())

  tau_alpha <- glmm_par$param_dict$tau$param_dict$shape$get()
  tau_beta <- glmm_par$param_dict$tau$param_dict$rate$get()
  tau_var <- tau_alpha / (tau_beta ^ 2)
  cov_param_dict$e_tau$set(tau_var)

  log_tau_var <- trigamma(tau_alpha)
  cov_param_dict$e_log_tau$set(log_tau_var)

  u_var <- 1.0 / glmm_par$param_dict$u$param_dict$info$get()
  cov_param_dict$e_u$set(array(u_var))

  return(cov_moment_par$moment_par$get_vector())
}
