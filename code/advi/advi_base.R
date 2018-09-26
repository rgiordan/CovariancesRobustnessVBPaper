
library(rstan)
options(mc.cores=4)

library(numDeriv)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)
library(mvtnorm)
library(Matrix)

rstan_options(auto_write=TRUE)

# Set this to be the appropriate location of the repo on your computer.
# Run from anywhere in the StanSensitivity repository.
setwd("/home/rgiordan/Documents/git_repos/CovariancesRobustnessVBPaper/code/advi")
git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
example_dir <- file.path(git_repo, "examples/advi")
source(file.path(example_dir, "advi_lib.R"))
source(file.path(example_dir, "py_opt_lib.R"))

py_main <- InitializePython()

##################################
# Uncomment to compile and run the base model.

model_names_for_paper <- c(
  "models/sesame_street1",
  "models/election88",
  "models/cjs_cov_raneff",
  "models/radon_vary_intercept_floor"
)

for (model_name in model_names_for_paper) {


####################################
# Begin analysis

# A list for keeping numbers that we'll want to put in the paper.
model_metadata <- list()

stan_data <- new.env()
source(paste(example_dir,
             paste(model_name, "data.R", sep="."), sep="/"), local=stan_data)
stan_data <- as.list(stan_data)
model_metadata$data_kb <- format(object.size(stan_data), units="Kb")
cat("Data size:  ", model_metadata$data_kb)

model <- stan_model(file.path(example_dir, paste(model_name, "stan", sep=".")))
modelfit <- sampling(model, data=stan_data, chains=1, iter=1)
draw_list <- get_inits(modelfit, iter=1)[[1]]
model_metadata$num_upars <- modelfit@.MISC$stan_fit_instance$num_pars_unconstrained()
cat("Number of upars: ", model_metadata$num_upars, "\n")

# Get samples:

num_mcmc_samples <- 8000
chains <- 2
model_metadata$num_mcmc_samples <- num_mcmc_samples
model_metadata$chains <- chains
sampling_file <- file.path(example_dir, paste(model_name, "_sampling.Rdata", sep=""))
# Not every saved file has the times.  Make sure you don't accidentally look at the wrong time in this case.
mcmc_time <- NULL
advi_time <- NULL
if (!file.exists(sampling_file)) {
  print("Running sampler.")
  set.seed(42)
  mcmc_time <- Sys.time()
  sampling_result <- sampling(model, data=stan_data, chains=chains, iter=num_mcmc_samples)
  mcmc_time <- Sys.time() - mcmc_time
  advi_time <- Sys.time()
  #advi_result <- vb(model, data=stan_data, iter=num_mcmc_samples, algorithm="meanfield")
  advi_result <- vb(model, data=stan_data, iter=num_mcmc_samples)
  advi_time <- Sys.time() - advi_time
  fr_advi_time <- Sys.time()
  fr_advi_result <- vb(model, data=stan_data, iter=num_mcmc_samples, algorithm="fullrank")
  fr_advi_time <- Sys.time() - fr_advi_time
  save(advi_result, fr_advi_result, sampling_result,
       advi_time, fr_advi_time, mcmc_time,
       file=sampling_file)
} else {
  print(sprintf("Loading cached samples from %s", sampling_file))
  load(sampling_file)
}
#print(summary(sampling_result))
set.seed(43)

model_metadata$mcmc_time <- mcmc_time
model_metadata$advi_time <- advi_time
model_metadata$fr_advi_time <- fr_advi_time
cat("mcmc time: ", as.numeric(mcmc_time, units="secs"))
cat("advi time: ", as.numeric(advi_time, units="secs"))
cat("fr advi time: ", as.numeric(fr_advi_time, units="secs"))

sampling_params <- data.frame(do.call(
  rbind, get_sampler_params(
    sampling_result, inc_warmup = FALSE)))
head(sampling_params)
model_metadata$num_divergent <- sum(sampling_params$divergent__)
cat("Number of divergent transitions: ", model_metadata$num_divergent, "\n")
cat("Proportion of divergent transitions: ",
    model_metadata$num_divergent / nrow(sampling_params), "\n")


#############
# ADVI objectives

draw_list <- get_inits(modelfit, iter=1)[[1]]

# Get inits from the true posterior or the ADVI result
advi_list <- GetADVIParametersFromSamples(advi_result, modelfit, draw_list, chain=1)
fr_advi_list <- GetADVIParametersFromSamples(fr_advi_result, modelfit, draw_list, chain=1)
mcmc_advi_list <- GetADVIParametersFromSamples(sampling_result, modelfit, draw_list, chain=1)

cat("Model contains ", advi_list$upar_length, " parameters.\n")

# Optimize
# Evaluated at a bad optimum, this is unreliable.
#choose_num_draws <- ChooseNumSamples(advi_list, modelfit, 0.5, num_draws=20)
#num_draws <- max(c(choose_num_draws, 10))
num_draws <- 10
model_metadata$num_draws <- num_draws
cat("Using ", num_draws, " draws.\n")
advi_list_init <- advi_list

kl_funcs <- GetKLFunctions(advi_list, modelfit, num_draws=num_draws)
advi_par_init <- WrapADVIParams(advi_list)

# Optimize in R
r_opt_time <- NA
advi_list_r_opt <- NA
if (TRUE) {
  kl_funcs <- GetKLFunctions(advi_list, modelfit, num_draws=num_draws)
  r_opt_time <- Sys.time()
  # Compare with CG?
  advi_par_init <- WrapADVIParams(advi_list)
  r_advi_opt <- optim(
    par=advi_par_init,
    fn=kl_funcs$KLForOpt,
    gr=kl_funcs$KLGradForOpt,
    method="CG"
  )
  advi_list_r_opt <- UnwrapADVIParams(r_advi_opt$par)
  r_opt_time <- Sys.time() - r_opt_time
  print(r_advi_opt$message)
}
model_metadata$r_opt_time <- r_opt_time


# Optimize in Python
kl_funcs <- GetKLFunctions(advi_list, modelfit, num_draws=num_draws)
DefinePythonOptimizationFunctions(
  "kl_opt_obj",
  kl_funcs$KLForOpt, kl_funcs$KLGradForOpt,
  kl_funcs$KLHessianForOpt,
  kl_funcs$KLHVPForOpt)

py_opt_time <- Sys.time()
py_opt_list <- PythonOptimizeTrustNCG(init_x=WrapADVIParams(advi_list_init), "kl_opt_obj")
py_opt_time <- Sys.time() - py_opt_time
model_metadata$py_opt_time <- py_opt_time
advi_list_opt <- UnwrapADVIParams(py_opt_list$x)


#############################
# MAP

map_funcs <- GetMAPObjectives(modelfit)
DefinePythonOptimizationFunctions(
  "map_opt_obj",
  map_funcs$f,
  map_funcs$grad,
  map_funcs$hessian,
  map_funcs$hvp)

map_time <- Sys.time()
#map_init_x <- WrapADVIParams(advi_list_init)[1:advi_list_init$upar_length]
#map_init_x <- rep(0, advi_list_init$upar_length)
#map_init_x <- runif(advi_list_init$upar_length)
map_init_opt <- optim(
  par=rep(0, advi_list_init$upar_length),
  fn=map_funcs$f,
  gr=map_funcs$grad,
  method="BFGS")
map_init_x <- map_init_opt$par

map_opt_list <- PythonOptimizeTrustNCG(
  init_x=map_init_x,
  "map_opt_obj")
map_info <- map_funcs$hessian(map_opt_list$x)
map_cov <- 0 * diag(nrow(map_info))
model_metadata$bad_map <- FALSE
tryCatch(map_cov <- solve(map_info),
         error=function(e) { print(e); print("Using default zero-cov mat."); model_metadata$bad_map <<- TRUE })
map_time <- Sys.time() - map_time
model_metadata$map_time <- map_time

map_eigenvalues <- eigen(map_info)$values
#stopifnot(min(map_eigenvalues) > -1e-8)
summary(map_eigenvalues)
#map_cov <- matrix(NA, nrow(map_info), ncol(map_info) )

data.frame(
  name=modelfit@.MISC$stan_fit_instance$unconstrained_param_names(0, 0),
  info_diag=diag(map_info),
  est=map_opt_list$x)

map_opt <- list(
  mean=map_opt_list$x,
  info=map_info,
  cov=map_cov,
  sd=sqrt(diag(map_cov)))

##############################
# Inspect optimization results

advi_par_opt <- WrapADVIParams(advi_list_opt)

model_metadata$our_kl <- kl_funcs$KLForOpt(WrapADVIParams(advi_list_opt), verbose=FALSE)
model_metadata$stan_kl <- kl_funcs$KLForOpt(WrapADVIParams(advi_list), verbose=FALSE)

print("KL comparison:")
cat("Our KL: ", model_metadata$our_kl, "\n")
cat("Stan KL: ", model_metadata$stan_kl, "\n")

kl_hess_time <- Sys.time()
kl_hess <- kl_funcs$KLHessianForOpt(advi_par_opt)
kl_hess_time <- Sys.time() - kl_hess_time
model_metadata$kl_hess_time <- kl_hess_time
cat("KL Hessian time: ", kl_hess_time, "\n")
cat("KL Hessian dimension: ", dim(kl_hess), "\n")

kl_grad <- kl_funcs$KLGradForOpt(advi_par_opt)
stan_advi_kl_hess <- kl_funcs$KLHessianForOpt(WrapADVIParams(advi_list))
stan_advi_kl_grad <- kl_funcs$KLGradForOpt(WrapADVIParams(advi_list))
r_kl_hess <- kl_funcs$KLHessianForOpt(WrapADVIParams(advi_list_r_opt))
r_kl_grad <- kl_funcs$KLGradForOpt(WrapADVIParams(advi_list_r_opt))

inv_time <- Sys.time()
lrvb_cov <- solve(kl_hess)
inv_time <- Sys.time() - inv_time
model_metadata$inv_time <- inv_time

lrvb_ev <- eigen(kl_hess)$values
stopifnot(min(lrvb_ev) > -1e-8)
max(lrvb_ev) / min(lrvb_ev)
#stan_lrvb_ev <- eigen(stan_advi_kl_hess)$values
#stopifnot(min(stan_lrvb_ev) > -1e-8)
stopifnot(sum(is.nan(stan_advi_kl_hess)) == 0)
stopifnot(sum(is.nan(kl_hess)) == 0)

######################################
# What's the sampling variance from the stochastic approximation?

par_cov <- EstimateSamplingCovariance(advi_list_opt, kl_hess, kl_funcs)
sample_error_df <- GetSampleErrorDataframe(advi_list_opt, par_cov, lrvb_cov)
accuracy <- 0.5
target_quantile <- 1.0
req_num_samples <- ChooseNumSamples(advi_list_opt, kl_hess, kl_funcs, lrvb_cov,
                                    accuracy=accuracy, target_quantile=target_quantile)

cat("Number of samples required: ", req_num_samples, "\n")
cat("Number of samples used: ", nrow(kl_funcs$norm_draws_opt), "\n")
if (req_num_samples > num_draws) {
  warn("Detected that more samples were required than were used.")
}

model_metadata$accuracy <- accuracy
model_metadata$target_quantile <- target_quantile
model_metadata$req_num_samples <- req_num_samples

if (FALSE) {
  ggplot(melt(sample_error_df)) +
    geom_histogram(aes(x=value)) +
    facet_grid(variable ~ .)
  summary(sample_error_df$mu_sd / sample_error_df$vb_sd)
  summary(sample_error_df$mu_sd)
}

#######################
# Posterior means via draws.

num_post_draws <- 5000
model_metadata$num_post_draws <- num_post_draws
norm_draws <- GetNormDraws(advi_list_opt, num_draws=num_post_draws)


upar_names <- modelfit@.MISC$stan_fit_instance$unconstrained_param_names(FALSE, FALSE)
mcmc_draws <-
  do.call(rbind,
          lapply(1:chains,
          function(chain) { GetUnconstrainedParameterMatrix(sampling_result, modelfit, draw_list, chain) }))
colnames(mcmc_draws) <- upar_names
mcmc_res <- GetConstrainedParameterLists(mcmc_draws, modelfit, draw_list)

fr_advi_draws <-
  GetUnconstrainedParameterMatrix(fr_advi_result, modelfit, draw_list, 1)
colnames(fr_advi_draws) <- upar_names
fr_advi_res <- GetConstrainedParameterLists(fr_advi_draws, modelfit, draw_list)



mcmc_result_df <-
  SummarizeParameterList(mcmc_res) %>%
  mutate(method="mcmc")
stan_advi_result_df <-
  GetADVIConstrainedDataframe(
    advi_list, modelfit, draw_list, norm_draws=norm_draws) %>%
  mutate(method="stan_advi")
fr_advi_result_df <-
  SummarizeParameterList(fr_advi_res) %>%
  mutate(method="fr_advi")

map_draws <- rmvnorm(n=num_post_draws, mean=map_opt$mean, sigma=map_opt$cov)
map_res <- GetConstrainedParameterLists(map_draws, modelfit, draw_list)
map_result_df <-
  SummarizeParameterList(map_res) %>%
  mutate(method="map")

opt_advi_result_df <-
  GetADVIConstrainedDataframe(
    advi_list_opt, modelfit, draw_list, norm_draws=norm_draws) %>%
  mutate(method="opt_advi")

lrvb_result_df <-
  GetADVIConstrainedDataframe(
    advi_list_opt, modelfit, draw_list, num_post_draws=num_post_draws,
    lrvb_cov=lrvb_cov) %>%
  mutate(method="lrvb_advi")

# For the bad optima, make the covariance positive definite if they are not.
stan_advi_lrvb_cov <- MakeMatrixPD(solve(stan_advi_kl_hess))
stan_lrvb_advi_result_df <-
  GetADVIConstrainedDataframe(
    advi_list, modelfit, draw_list, num_post_draws=num_post_draws,
    lrvb_cov=stan_advi_lrvb_cov) %>%
  mutate(method="stan_lrvb_advi")

r_lrvb_advi_cov <- MakeMatrixPD(solve(r_kl_hess))
r_lrvb_advi_df <-
  GetADVIConstrainedDataframe(
    advi_list_r_opt, modelfit, draw_list, num_post_draws=num_post_draws,
    lrvb_cov=r_lrvb_advi_cov) %>%
  mutate(method="r_lrvb_advi")

result_df <- bind_rows(
  mcmc_result_df,
  stan_advi_result_df,
  map_result_df,
  opt_advi_result_df,
  lrvb_result_df,
  stan_lrvb_advi_result_df,
  r_lrvb_advi_df,
  fr_advi_result_df)

varcols <- names(result_df)[ grepl("Var[0-9]+", names(result_df))]
varcols_mat <- as.matrix(result_df[varcols])
index_col <- apply(as.matrix(result_df[varcols]), MARGIN=1,
                   FUN=function(x) { paste(x, collapse="_") })
result_df["Index"] <- index_col
head(result_df)

param_dims <-
  modelfit@.MISC$stan_fit_instance$param_dims() %>%
  lapply(function(x) { ifelse(length(x) == 0, 1, x) }) %>%
  unlist() %>%
  as.data.frame() %>%
  rename(dim = ".")
param_dims$param <- rownames(param_dims)
rownames(param_dims) <- NULL

max_dim <- Inf
graph_result_cast_df <-
  dcast(result_df, param + Index + metric ~ method) %>%
  inner_join(param_dims, by="param") %>%
  filter(dim < max_dim)

grid.arrange(
  ggplot(filter(graph_result_cast_df, metric == "mean")) +
    geom_point(aes(x=mcmc, y=opt_advi, color=param)) +
    geom_abline(aes(slope=1, intercept=0)) +
    expand_limits(x=0, y=0) +
    ggtitle("ADVI mean")
,
  ggplot(filter(graph_result_cast_df, metric == "mean")) +
    geom_point(aes(x=mcmc, y=lrvb_advi, color=param)) +
    geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("LRVB ADVI mean")
,
ggplot(filter(graph_result_cast_df, metric == "mean")) +
  geom_point(aes(x=mcmc, y=stan_lrvb_advi, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("Stan LRVB ADVI mean")
,
ggplot(filter(graph_result_cast_df, metric == "mean")) +
  geom_point(aes(x=mcmc, y=fr_advi, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("Full rank advi")
,
ggplot(filter(graph_result_cast_df, metric == "mean")) +
  geom_point(aes(x=mcmc, y=map, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("MAP mean")
,

###############
# Standard deviations

ggplot(filter(graph_result_cast_df, metric == "sd")) +
    geom_point(aes(x=mcmc, y=opt_advi, color=param)) +
    geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("ADVI sd")
  ,
  ggplot(filter(graph_result_cast_df, metric == "sd")) +
    geom_point(aes(x=mcmc, y=lrvb_advi, color=param)) +
    geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("LRVB ADVI sd")
,
ggplot(filter(graph_result_cast_df, metric == "sd")) +
  geom_point(aes(x=mcmc, y=stan_lrvb_advi, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("Stan LRVB ADVI sd")
,
ggplot(filter(graph_result_cast_df, metric == "sd")) +
  geom_point(aes(x=mcmc, y=fr_advi, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("Full rank advi")
,
ggplot(filter(graph_result_cast_df, metric == "sd")) +
  geom_point(aes(x=mcmc, y=map, color=param)) +
  geom_abline(aes(slope=1, intercept=0)) +
  expand_limits(x=0, y=0) +
  ggtitle("MAP sd")
,
  ncol=5
)


group_by(graph_result_cast_df, metric) %>%
  summarize(lrvb_vs_advi=sum(abs(lrvb_advi - opt_advi)),
            advi_error=sum(abs(mcmc - opt_advi)),
            stan_lrvb_error=sum(abs(mcmc - stan_lrvb_advi)),
            lrvb_error=sum(abs(mcmc - lrvb_advi)),
            map_error=sum(abs(mcmc - map))
  )


##################################################
# Look at differences relative to MCMC and posterior stdev

result_cast_df <- dcast(result_df, Index + param + method ~ metric)
result_vs_mcmc_df <-
  inner_join(filter(result_cast_df, method != "mcmc"),
             filter(result_cast_df, method == "mcmc"),
             by=c("Index", "param"), suffix=c("", "_mcmc")) %>%
  mutate(diff_z=(mean - mean_mcmc) / sd_mcmc,
         sd_error=sd - sd_mcmc,
         mean_error=mean - mean_mcmc)
if (FALSE) {
  ggplot(result_vs_mcmc_df) +
    geom_histogram(aes(x=diff_z, fill=method)) +
    geom_vline(aes(xintercept=-2)) +
    geom_vline(aes(xintercept=2)) +
    facet_grid(method ~ .)
  ggplot(result_vs_mcmc_df) +
    geom_histogram(aes(x=sd_error, fill=method)) +
    facet_grid(method ~ .)

}


#######################
# Times
total_advi_time <- advi_time + py_opt_time
total_lrvb_time <- total_advi_time + kl_hess_time + inv_time

cat("ADVI time fraction: ",
    as.numeric(total_advi_time, units="secs") / as.numeric(mcmc_time, units="secs"), "\n")
cat("LRVB time fraction: ",
    as.numeric(total_lrvb_time, units="secs") / as.numeric(mcmc_time, units="secs"), "\n")
cat("R LRVB time fraction: ",
    as.numeric(advi_time + r_opt_time + kl_hess_time + inv_time, units="secs") /
      as.numeric(mcmc_time, units="secs"), "\n")
cat("FR ADVI time fraction: ",
    as.numeric(fr_advi_time, units="secs") / as.numeric(mcmc_time, units="secs"), "\n")
cat("MAP time fraction: ",
    as.numeric(map_time, units="secs") / as.numeric(mcmc_time, units="secs"), "\n")


####################################
# Save results for paper

model_metadata$model_name <- model_name
result_filename <-
  file.path(example_dir, "results", sprintf("%s_lrvb_advi_results.Rdata", sub("models/", "", model_name)))
save(graph_result_cast_df, model_metadata, result_vs_mcmc_df, file=result_filename)


} # For model_name in model_names_for_paper
