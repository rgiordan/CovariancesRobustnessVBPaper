library(abind)


GetBetaLogDensity <- function(beta, vp_opt, draw, pp, unconstrained, calculate_gradient) {
  draw$beta_e_vec <- beta
  draw$beta_e2_vec <- beta %*% t(beta)
  opt$calculate_gradient <- calculate_gradient
  opt$calculate_hessian <- FALSE
  q_derivs <- GetLogVariationalDensityDerivatives(draw, vp_opt, opt, global_only=TRUE,
                                                  include_beta=TRUE, include_mu=FALSE, include_tau=FALSE)
  return(q_derivs)
}


# Get a function that converts a a draw from mu_k and a standard mvn into a draw from (mu_c | mu_k)
# k is the conditioned component, c is the "complement", i.e. the rest
GetConditionalMVNFunction <- function(k_ind, mvn_mean, mvn_info) {
  mvn_sigma <- solve(mvn_info)
  c_ind <- setdiff(1:length(mvn_mean), k_ind)
  
  # The scale in front of the mu_k for the mean of (mu_c | mu_k)
  # mu_cc_sigma <- mu_sigma[c_ind, c_ind, drop=FALSE]
  mu_kk_sigma <- mvn_sigma[k_ind, k_ind, drop=FALSE]
  mu_ck_sigma <- mvn_sigma[c_ind, k_ind, drop=FALSE]
  sig_cc_corr <- mu_ck_sigma %*% solve(mu_kk_sigma)
  
  # What to multiply by to get Cov(mu_c | mu_k)
  mu_c_cov <- solve(mvn_info[c_ind, c_ind])
  mu_c_scale <- t(chol(mu_c_cov))
  
  # Given u and a draws mu_c_std ~ Standard normal, convert mu_c_std to a draw from MVN( . | mu_k).
  # If there are multiple mu_c_std, each draw should be in its own column.
  GetConditionalDraw <- function(mu_k, mu_c_std) {
    mu_c_mean <- mvn_mean[c_ind] + sig_cc_corr %*% (mu_k - mvn_mean[k_ind, drop=FALSE])
    mu_c_scale %*% mu_c_std + matrix(rep(mu_c_mean, ncol(mu_c_std)), ncol=ncol(mu_c_std))
  }
}


# Beta draws:
GetBetaImportanceFunctions <- function(beta_comp, vp_opt, pp, lrvb_results) {
  mp_opt <- GetMomentParametersFromNaturalParameters(vp_opt)
  
  beta_cov <- solve(vp_opt$beta_info)
  u_mean <- vp_opt$beta_loc[beta_comp]
  # Increase the variance for sampling.  How much is enough?
  u_cov <- (1.5 ^ 2) * beta_cov[beta_comp, beta_comp]
  GetULogDensity <- function(u) {
    dnorm(u, mean=u_mean, sd=sqrt(u_cov), log=TRUE)
  }
  
  DrawU <- function(n_samples) {
    rnorm(n_samples, mean=u_mean, sd=sqrt(u_cov))
  }
  
  prior_cov <- solve(pp$beta_info)
  GetLogPrior <- function(u) {
    dnorm(u, mean=pp$beta_loc[beta_comp], sd=sqrt(prior_cov[beta_comp, beta_comp]), log=TRUE)
  }

  # This is the marginal density of the beta_comp component.
  GetLogVariationalDensity <- function(u) {
    return(dnorm(u, mean=vp_opt$beta_loc[beta_comp], sd=sqrt(beta_cov[beta_comp, beta_comp]), log=TRUE))
  }
  
  # The remainder is used to calculate the gbar term.
  
  # This is the density and derivatives of the full beta density.  
  global_mask <- rep(FALSE, vp_opt$encoded_size)
  global_indices <- unique(c(vp_indices$beta_loc, as.numeric(vp_indices$beta_info[]),
                             vp_indices$mu_loc, vp_indices$mu_info,
                             vp_indices$tau_alpha, vp_indices$tau_beta))
  global_mask[global_indices] <- TRUE
  GetFullBetaLogVariationalDensity <- function(u) {
    mp_draw <- mp_opt
    log_q_grad <- rep(0, vp_indices$encoded_size)
    beta_q_derivs <- GetBetaLogDensity(u, vp_opt, mp_draw, pp, TRUE, TRUE)
    log_q_grad[global_mask] <- beta_q_derivs$grad
    list(val=beta_q_derivs$val, grad=log_q_grad)
  }
  
  GetFullLogQGradTerm <- function(beta) {
    beta_log_q_derivs <- GetFullBetaLogVariationalDensity(beta)
    return(as.numeric(beta_log_q_derivs$grad))
  }
  
  lrvb_pre_factor <- -1 * lrvb_results$jac %*% solve(lrvb_results$elbo_hess)
  DrawConditionalBeta <- GetConditionalMVNFunction(beta_comp, vp_opt$beta_loc, vp_opt$beta_info)
  GetLogQGradResults <- function(u_draws, num_mc_draws, normalize=TRUE) {
    c_ind <- setdiff(1:vp_opt$k_reg, beta_comp)
    beta_std_draws <- rmvnorm(num_mc_draws, mean=rep(0, vp_opt$k - 1))
    
    # Draws from the rest of beta (beta "complement") given u_draws.
    DrawBetaGivenU <- function(u) {
      beta_draw <- matrix(NaN, vp_opt$k_reg, num_mc_draws)
      beta_draw[c_ind, ] <- DrawConditionalBeta(u, t(beta_std_draws))
      beta_draw[beta_comp, ] <- u
      return(beta_draw)
    }
    beta_u_draws_list <- lapply(u_draws, DrawBetaGivenU)

    # The dimensions of beta_u_draws are (component, mc draw, u draw)
    beta_u_draws <- abind(beta_u_draws_list, along=3)

    # The dimensions of lrvb_term_draws work out to be c(moment index, conditional beta draw, u draw)
    lrvb_term_draws <- apply(beta_u_draws, MARGIN=c(2, 3), FUN=GetFullLogQGradTerm)
    lrvb_term_e <- apply(lrvb_term_draws, MARGIN=c(1, 3), FUN=mean)

    if (normalize) {
      imp_ratio <- exp(GetLogVariationalDensity(u_draws) - GetULogDensity(u_draws))
      lrvb_term_e_means <- colSums(imp_ratio * t(lrvb_term_e)) / sum(imp_ratio)
      lrvb_term_e <- lrvb_term_e - lrvb_term_e_means
    }
    
    
    lrvb_terms <- lrvb_pre_factor %*% lrvb_term_e

    return(list(lrvb_terms=lrvb_terms, lrvb_term_e=lrvb_term_e, lrvb_term_draws=lrvb_term_draws, beta_u_draws=beta_u_draws))
  }

  GetLogQGradTerms <- function(u_draws, num_mc_draws, normalize=TRUE) {
    as.matrix(GetLogQGradResults(u_draws, num_mc_draws, normalize=normalize)$lrvb_terms)
  }
  
  return(list(GetULogDensity=GetULogDensity,
              DrawU=DrawU,
              GetLogPrior=GetLogPrior,
              GetLogVariationalDensity=GetLogVariationalDensity,
              GetLogQGradTerms=GetLogQGradTerms,
              GetLogQGradResults=GetLogQGradResults))
}

