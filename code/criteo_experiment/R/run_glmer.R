library(rstan)
library(lme4)

library(dplyr)
library(reshape2)
library(trust)
library(tidyr)

library(LRVBUtils)

library(mvtnorm)
library(gridExtra)

git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
project_directory  <- file.path(git_repo, "code/criteo_experiment")
# project_directory <- file.path(
#   Sys.getenv("GIT_REPO_LOC"),
#   "VariationalBayesPythonWorkbench/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

source(file.path(project_directory, "R/logit_glmm_lib.R"))
source(file.path(project_directory, "R/densities_lib.R"))

analysis_name <- "criteo_subsampled"
#analysis_name <- "simulated_data_small"

# If true, save the results to a file readable by knitr.
save_results <- TRUE
glmer_results_file <-
  file.path(data_directory,
            paste(analysis_name, "glmer_results.Rdata", sep="_"))

stan_draws_file <- file.path(
  data_directory, paste(analysis_name, "_mcmc_draws.Rdata", sep=""))

stan_results <- LoadIntoEnvironment(stan_draws_file)


##############
# frequentist glmm
glmm_df <- tibble(y=stan_results$stan_dat$y, y_g=as.character(stan_results$stan_dat$y_g))
for (col in 1:stan_results$stan_dat$K) {
  glmm_df[paste("x", col, sep=".")] <- stan_results$stan_dat$x[, col]
}
regressors <- paste("x", 1:5, sep=".")
glmm_formula_string <- sprintf("y ~ %s + (1|y_g)", paste(regressors, collapse=" + "))

glmer_time <- Sys.time()
glmm_res <- glmer(formula(glmm_formula_string),
                  data=glmm_df, family="binomial", verbose=FALSE,
                  nAGQ=4)
glmer_time <- Sys.time() - glmer_time

glmm_summary <- summary(glmm_res)
# u <- data.frame(ranef(glmm_res)$y_g)
# names(u) <- "u"
# u$y_g <- rownames(u)

# u_df <- inner_join(u, glmm_df[, "y_g"], by="y_g")
# mean(u_df$u)
# mean(u$u[y_g + 1])

glmm_list <- list()
glmm_list$beta_mean <- glmm_summary$coefficients[regressors, "Estimate"]
glmm_list$beta_sd <- glmm_summary$coefficients[regressors, "Std. Error"]
glmm_list$beta_par <- rownames(glmm_summary$coefficients[regressors, ])

glmm_list$mu_mean <- glmm_summary$coefficients["(Intercept)", "Estimate"]
glmm_list$mu_sd <- glmm_summary$coefficients["(Intercept)", "Std. Error"]

glmm_list$tau_mean <- 1 / attr(glmm_summary$varcor$y_g, "stddev") ^ 2

# Glmer uses a non-centered model, where stan and vb use a centered model.
glmm_ranef <- ranef(glmm_res, condVar=TRUE)
y_g_numerical_perm <- order(as.numeric(rownames(glmm_ranef$y_g)))
glmm_list$u_map <- as.numeric(
  ranef(glmm_res)$y_g[y_g_numerical_perm, "(Intercept)"]) + glmm_list$mu_mean
u_post_sd <- sqrt(array(attr(glmm_ranef$y_g, "postVar")[y_g_numerical_perm]))
glmm_list$u_cond_sd <- u_post_sd

glmm_list$glmm_time <- as.numeric(glmer_time, units="secs")
glmm_list$glmm_res <- glmm_res

# Debugging
# plot(stan_results$true_params$u, glmm_list$u_map); abline(0, 1)

if (save_results) {
  print(sprintf("Saving to %s", glmer_results_file))
  save(glmm_list, file=glmer_results_file)
  
  # Save in JSON for reading in Python
  json_filename <- file.path(
    data_directory, paste(analysis_name, "_glmer_results.json", sep=""))
  json_file <- file(json_filename, "w")
  glmm_json_list <- glmm_list
  glmm_json_list$glmm_res <- NULL # Cannot be saved as JSON
  json_list <- toJSON(glmm_json_list)
  write(json_list, file=json_file)
  close(json_file)
}

