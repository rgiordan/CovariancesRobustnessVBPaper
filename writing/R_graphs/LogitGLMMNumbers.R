
# Define LaTeX macros that will let us automatically refer
# to simulation and model parameters.

DefineMacro <- function(macro_name, value, digits=3) {
  sprintf_code <- paste("%0.", digits, "f", sep="")
  cat("\\newcommand{\\", macro_name, "}{", sprintf(sprintf_code, value), "}\n", sep="")
}

# From the parametric sensitivy environment
if (exists("glmm_env")) {
  DefineMacro("glmmDimension", glmm_env$beta_dim, digits=0)
  DefineMacro("glmmNumGroups", glmm_env$num_groups, digits=0)
  DefineMacro("glmmNumObs", glmm_env$num_obs, digits=0)
  DefineMacro("glmmHessDim", glmm_env$hess_dim, digits=0)
  
  DefineMacro("glmmInverseTime", glmm_env$inverse_time, digits=0)
  DefineMacro("glmmHessianTime", glmm_env$hess_time, digits=0)
  DefineMacro("glmmVBTime", glmm_env$vb_time, digits=0)
  DefineMacro("glmmMCMCTime", glmm_env$mcmc_time, digits=0)
  DefineMacro("glmmMCMCTimeMinutes", glmm_env$mcmc_time / 60., digits=0)
  DefineMacro("glmmCGRowTime", glmm_env$cg_row_time, digits=1)
  DefineMacro("glmmCGRowIters", glmm_env$num_cg_iterations, digits=0)
  DefineMacro("glmmCGBetaTime", glmm_env$cg_row_time * glmm_env$beta_dim, digits=1)
  DefineMacro("glmmMAPTime", glmm_env$map_time, digits=0)
  DefineMacro("glmmGLMERTime", glmm_env$glmer_time, digits=0)
  DefineMacro("glmmVBRefitTime", glmm_env$vb_refit_time, digits=1)
  
  DefineMacro("glmmSpeedup", glmm_env$mcmc_time / glmm_env$vb_time, digits=0)
  DefineMacro("glmmNumMCMCDraws", glmm_env$num_mcmc_draws, digits=0)
  DefineMacro("glmmNumGHPoints", glmm_env$num_gh_points, digits=0)

  DefineMacro("glmmBetaInfoDiag", glmm_env$pp$beta_prior_info[1, 1], digits=3)
  DefineMacro("glmmBetaLoc", glmm_env$pp$beta_prior_mean[1], digits=3)
  DefineMacro("glmmMuLoc", glmm_env$pp$mu_prior_mean, digits=3)
  DefineMacro("glmmMuInfo", glmm_env$pp$mu_prior_info, digits=3)
  DefineMacro("glmmTauAlpha", glmm_env$pp$tau_prior_alpha, digits=3)
  DefineMacro("glmmTauBeta", glmm_env$pp$tau_prior_beta, digits=3)
}
