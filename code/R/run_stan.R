library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(mvtnorm)
library(boot) # for inv.logit
library(rstansensitivity)

library(lme4)
library(rstan)


########################
# Run stan

rstan_options(auto_write=FALSE)

project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "VariationalBayesPythonWorkbench/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"

true_params <- list()
if (analysis_name == "simulated_data_small") {
  n_obs_per_group <- 10
  k_reg <- 5
  n_groups <- 100
  n_obs <- n_groups * n_obs_per_group
  
  set.seed(42)
  true_params <- list()
  true_params$n_obs <- n_obs
  true_params$k_reg <- k_reg
  true_params$n_groups <- n_groups
  true_params$tau <- 1
  true_params$mu <- -3.5
  true_params$beta <- 1:k_reg

  true_params$u <- list()
  for (g in 1:n_groups) {
    true_params$u[[g]] <- rnorm(1, true_params$mu, 1 / sqrt(true_params$tau))
  }
  
  # Select correlated regressors to induce posterior correlation in beta.
  x_cov <- (matrix(0.5, k_reg, k_reg) + diag(k_reg)) / 2.5
  x <- rmvnorm(n_obs, sigma=x_cov)
  
  # y_g is expected to be zero-indexed.
  y_g <- as.integer(rep(1:n_groups, each=n_obs_per_group) - 1)
  true_offsets <- x %*% true_params$beta
  for (n in 1:n_obs) {
    # C++ is zero indexed but R is one indexed
    true_offsets[n] <- true_offsets[n] + true_params$u[[y_g[n] + 1]]
  }
  true_probs <- inv.logit(true_offsets)
  print(summary(true_probs))
  y <- rbinom(n=n_obs, size=1, prob=true_probs)
  
  iters <- 3000 # We actually need more than this -- use this for debugging.
} else if (analysis_name == "criteo_subsampled") { 
  load(file.path(data_directory, "criteo_data_for_paper.Rdata"))
  iters <- 10000
} else {
  stop("Unknown analysis name.")
}


k_reg <- ncol(x)

stan_dat <- list(NG = max(y_g) + 1,
                 N = length(y),
                 K = ncol(x),
                 y_group = y_g,
                 y = y,
                 x = x,

                 # Priors
                 beta_prior_mean = rep(0, k_reg),
                 beta_prior_info = 0.1 * diag(k_reg),
                 mu_prior_mean = 0.0,
                 mu_prior_info = 0.01,
                 tau_prior_alpha = 3.0,
                 tau_prior_beta = 3.0)


##############
# Export the data for fitting in Python.

json_filename <- file.path(
    data_directory, paste(analysis_name, "_stan_dat.json", sep=""))
json_file <- file(json_filename, "w")
json_list <- toJSON(list(stan_dat=stan_dat))
write(json_list, file=json_file)
close(json_file)


##############
# MCMC


stan_directory <- file.path(project_directory, "stan")
stan_model_name <- "logit_glmm"
model_file <- file.path(
    stan_directory, paste(stan_model_name, "stan", sep="."))
model_file_rdata <- file.path(
    stan_directory, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  # Run this to force re-compilation of the model.
  print("Compiling Stan model.")
  # In the stan directory run
  # $GIT_REPO_LOC/StanSensitivity/python/generate_models.py --base_model=logit_glmm.stan
  model_file <- file.path(stan_directory, paste(stan_model_name, "_generated.stan", sep=""))
  model <- stan_model(model_file)
  stan_sensitivity_model <- GetStanSensitivityModel(
    file.path(stan_directory, "logit_glmm"), stan_dat)
  save(model, stan_sensitivity_model, file=model_file_rdata)
}


# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
seed <- 42
chains <- 1
cores <- 1 # Use one core for the sensitivity analysis.


# MCMC draws.
mcmc_time <- Sys.time()
stan_dat$mu_prior_epsilon <- 0
stan_sim <- sampling(
  model, data=stan_dat, seed=seed, iter=iters, chains=chains, cores=cores)
mcmc_time <- Sys.time() - mcmc_time

# ADVI.
advi_time <- Sys.time()
stan_advi <- vb(model, data=stan_dat,  algorithm="meanfield",
                output_samples=iters)
advi_time <- Sys.time() - advi_time

# Get a MAP estimate.
bfgs_map_time <- Sys.time()
stan_map_bfgs <- optimizing(
    model, data=stan_dat, algorithm="BFGS", hessian=TRUE,
    init=get_inits(stan_sim)[[1]], verbose=TRUE,
    tol_obj=1e-12, tol_grad=1e-12, tol_param=1e-12)
bfgs_map_time <- Sys.time() - bfgs_map_time

stan_map <- stan_map_bfgs
map_time <- bfgs_map_time


# Get the sensitivity results.
stopifnot(cores == 1) # rstansensitivity only supports one core for now.
draws_mat <- rstan::extract(stan_sim, permute=FALSE)[,1,]
mcmc_sens_time <- Sys.time()
sens_result <- GetStanSensitivityFromModelFit(stan_sim, draws_mat, stan_sensitivity_model)
mcmc_sens_time <- Sys.time() - mcmc_sens_time


# Save the results to an RData file for further post-processing.
stan_draws_file <- file.path(
    data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat, true_params,
     sens_result, stan_sensitivity_model, mcmc_sens_time,
     stan_advi, advi_time,
     stan_map, map_time,
     chains, cores,
     file=stan_draws_file)
