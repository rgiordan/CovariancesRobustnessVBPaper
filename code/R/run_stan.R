#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(rstan)
library(rstansensitivity)


rstan_options(auto_write=FALSE)

git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
data_dir <- file.path(git_repo, "code/data")

analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"
json_filename <- file.path(
  data_dir, paste(analysis_name, "_stan_dat.json", sep=""))

json_dat <- fromJSON(readLines(json_filename))
stan_dat <- json_dat$stan_dat


##############
# MCMC

stan_dir <- file.path(git_repo, "code/R/stan/")
stan_model_name <- "logit_glmm"
model_file_rdata <- file.path(data_dir, paste(stan_model_name, "Rdata", sep="."))
if (file.exists(model_file_rdata)) {
  print("Loading pre-compiled Stan model.")
  load(model_file_rdata)
} else {
  # Run this to force re-compilation of the model.
  print("Compiling Stan model.")
  # In the stan directory run
  # StanSensitivity/python/generate_models.py --base_model=logit_glmm.stan
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
    data_dir, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))
save(stan_sim, mcmc_time, stan_dat, true_params,
     sens_result, stan_sensitivity_model, mcmc_sens_time,
     stan_advi, advi_time,
     stan_map, map_time,
     chains, cores,
     file=stan_draws_file)
