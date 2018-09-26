#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(rstan)
library(rstansensitivity)

rstan_options(auto_write=FALSE)


##############
# Load the data.

git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
data_dir <- file.path(git_repo, "code/data")

analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"

# Input file:
json_filename <- file.path(
  data_dir, paste(analysis_name, "_stan_dat.json", sep=""))

# Output file:
stan_draws_file <- file.path(
  data_dir, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

json_dat <- fromJSON(readLines(json_filename))
stan_dat <- json_dat$stan_dat

stan_dir <- file.path(git_repo, "code/R/stan/")
stan_model_name <- "logit_glmm"

# Complie the Stan model.
print("Compiling Stan model.")

# To make the sensitivity scripts, run the following command in the stan directory:
# StanSensitivity/python/generate_models.py --base_model=logit_glmm.stan
model_file <- file.path(stan_dir, paste(stan_model_name, "_generated.stan", sep=""))
model <- stan_model(model_file)
stan_sensitivity_model <- GetStanSensitivityModel(file.path(stan_dir, "logit_glmm"), stan_dat)


###################
# Run MCMC.

# Some knobs we can tweak.  Note that we need many iterations to accurately assess
# the prior sensitivity in the MCMC noise.
iters <- json_dat$iters
seed <- 42
chains <- 1
cores <- 1 # Note: the senstivity analysis currently only supports one core.

# MCMC draws.
mcmc_time <- Sys.time()
stan_dat$mu_prior_epsilon <- 0
stan_sim <- sampling(
  model, data=stan_dat, seed=seed, iter=iters, chains=chains, cores=cores)
mcmc_time <- Sys.time() - mcmc_time

# Get the sensitivity results.
stopifnot(cores == 1) # rstansensitivity only supports one core for now.
draws_mat <- rstan::extract(stan_sim, permute=FALSE)[,1,]
mcmc_sens_time <- Sys.time()
sens_result <- GetStanSensitivityFromModelFit(stan_sim, draws_mat, stan_sensitivity_model)
mcmc_sens_time <- Sys.time() - mcmc_sens_time

# Save the results to an RData file for further post-processing.
save(stan_sim, mcmc_time, stan_dat,
     sens_result, stan_sensitivity_model, mcmc_sens_time,
     chains, cores,
     file=stan_draws_file)
