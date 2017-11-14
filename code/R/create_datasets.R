#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(boot) # for inv.logit

git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
data_directory <- file.path(project_directory, "code/data/")
input_rdata_filename <- file.path(data_directory, "criteo/criteo_data_for_paper.Rdata")
json_filename <- file.path(
  data_directory, paste(analysis_name, "_stan_dat.json", sep=""))

cat("Loading data from", input_rdata_filename, "\n")

analysis_name <- "criteo_subsampled"

# If you set the analysis name to this value, it will simulate a small dataset
# instead of loading the criteo data.  This allows for faster iteration and
# experimentation.
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
  true_params <- list()
  load(input_rdata_filename)
  iters <- 10000
} else {
  stop("Unknown analysis name.")
}


# Save all the data in a format readable by Stan.
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


# Export the data in JSON so it can be read by both Stan and Python.
json_file <- file(json_filename, "w")
json_list <- toJSON(list(stan_dat=stan_dat, true_params=true_params))
write(json_list, file=json_file)
close(json_file)
