library(rstan)
library(purrr)

########################################
# Functions for manipulating stan output

ClearDrawList <- function(draw_list) {
  for (varname in names(draw_list)) {
    varshape <- dim(draw_list[[varname]])
    if (is.null(varshape)) {
      varlen <- length(draw_list[[varname]])
      draw_list[[varname]] <- rep(NA, varlen)
      stopifnot(length(draw_list[[varname]]) == varlen)
    } else {
      draw_list[[varname]] <- array(NA, varshape)
      stopifnot(varshape == dim(draw_list[[varname]]))
    }
  }
  return(draw_list)
}


# A replacement for get_inits, which is broken for vb results
GetParameterList <- function(result_extract, chain, iter, modelfit, draw_list) {
  new_draw_list <- ClearDrawList(draw_list)
  for (varname in names(draw_list)) {
    vardims <- modelfit@par_dims[[varname]]
    if (length(vardims) == 0) {
      new_draw_list[[varname]] <- result_extract[[varname]][iter, chain, ]
    } else {
      new_draw_list[[varname]] <-
        array(result_extract[[varname]][iter, chain, ], vardims)
    }
  }
  return(new_draw_list)
}


ExtractStanToList <- function(sampling_result, draw_list) {
    result_extract <- list()
    for (varname in names(draw_list)) {
      result_extract[[varname]] <- extract(sampling_result, permuted=FALSE, varname)
    }
    return(result_extract)
}


GetUnconstrainedParameterMatrix <- function(sampling_result, modelfit, draw_list, chain) {
  # Get the results in a list.
  result_extract <- ExtractStanToList(sampling_result, draw_list)

  # Copy each element to a single list, unconstrain, and add a row to upar_matrix.
  num_iters <- sampling_result@sim$iter - sampling_result@sim$warmup
  upar_matrix <- matrix(NA, nrow=num_iters, ncol=get_num_upars(modelfit))
  pb <- txtProgressBar(min=0, max=nrow(upar_matrix), style=3)
  for (iter in 1:nrow(upar_matrix)) {
    setTxtProgressBar(pb, iter)
    new_draw_list <- GetParameterList(result_extract, chain, iter, modelfit, draw_list)
    upar_matrix[iter, ] <- unconstrain_pars(modelfit, new_draw_list)
  }
  close(pb)
  return(upar_matrix)
}



GetConstrainedParameterLists <- function(advi_draws, modelfit, draw_list) {
  par_list <-lapply(1:nrow(advi_draws),
                    function(iter) constrain_pars(modelfit, advi_draws[iter, ]))
  mean_list <- draw_list
  sd_list <- draw_list
  for (varname in names(draw_list)) {
    var_list <- lapply(par_list, function(x) { x[[varname]] })
    var2_list <- lapply(par_list, function(x) { x[[varname]] ^ 2 })
    var_sum <- reduce(var_list, `+`)
    var2_sum <- reduce(var2_list, `+`)
    var_mean <- var_sum / length(var_list)
    mean_list[[varname]] <- var_mean
    sd_list[[varname]] <- sqrt(var2_sum / length(var_list) - var_mean ^ 2)
  }
  return(list(par=par_list, mean=mean_list, sd=sd_list))
}


GetConstrainedParameterDataFrame <- function(par_list, draw_list) {
  df_list <- list()
  for (varname in names(par_list)) {
    var_df <- melt(par_list[[varname]]) %>% mutate(param=varname)
    df_list[[varname]] <- var_df
  }
  df <- do.call(bind_rows, df_list)
  return(df)
}

SummarizeParameterList <- function(par_list) {
  rbind(GetConstrainedParameterDataFrame(par_list$mean) %>% mutate(metric="mean"),
        GetConstrainedParameterDataFrame(par_list$sd) %>% mutate(metric="sd"))
}

GetLRVBDraws <- function(advi_list, lrvb_cov, num_draws) {
    mean_inds <- 1:advi_list$upar_length
    lrvb_z_cov <- lrvb_cov[mean_inds, mean_inds]
    lrvb_draws <- rmvnorm(n=num_draws, mean=advi_list$mean, sigma=lrvb_z_cov)
    return(lrvb_draws)
}

GetADVIConstrainedDataframe <- function(
    advi_list, modelfit, draw_list, num_post_draws=NULL, norm_draws=NULL, lrvb_cov=NULL) {

  if (is.null(lrvb_cov)) {
    if (is.null(norm_draws)) {
      norm_draws <- GetNormDraws(advi_list, num_draws=num_post_draws)
    }
    advi_draws <- GetADVIDraws(advi_list, norm_draws)

  } else {
    stopifnot(is.null(norm_draws))
    advi_draws <- GetLRVBDraws(advi_list, lrvb_cov, num_post_draws)
  }

  return(SummarizeParameterList(
      GetConstrainedParameterLists(advi_draws, modelfit, draw_list)))
}


# # Test:
# advi_draw_list <- GetParameterList(draw_list, advi_result, chain=1, iter=iter)
# free_pars <- unconstrain_pars(modelfit, advi_draw_list)
# advi_pars <- constrain_pars(modelfit, free_pars)
#
# for (varname in names(empty_draw_list)) {
#   if (max(abs(advi_draw_list[[varname]] - advi_pars[[varname]])) > 1e-8) {
#     print(varname)
#     print(advi_draw_list[[varname]])
#     print(advi_pars[[varname]])
#     stop("Parameter mismatch in constraining / unconstraining")
#   }
# }


########################################
# Functions for evaluating ADVI objectives


GetADVIParametersFromSamples <- function(
    sampling_result, modelfit, draw_list, chain) {

  upar_matrix <- GetUnconstrainedParameterMatrix(
      sampling_result, modelfit, draw_list, chain)
  advi_list <- list(
    upar_length=ncol(upar_matrix),
    mean=apply(upar_matrix, MARGIN=2, FUN=mean),
    log_sigma=log(apply(upar_matrix, MARGIN=2, FUN=sd))
  )
  return(advi_list)
}

# Each row is a draw, each column a parameters
GetNormDraws <- function(advi_list, num_draws) {
  return(matrix(rnorm(num_draws * advi_list$upar_length), nrow=num_draws))
}


GetADVIDraws <- function(advi_list, norm_draws) {
  # TODO: stack the normal draws the other way.
  return(norm_draws * rep(exp(advi_list$log_sigma), each=nrow(norm_draws)) +
         rep(advi_list$mean, each=nrow(norm_draws)))
}


#######################
# Applying stan gradients to a set of draws.

GetLogLikDraws <- function(advi_draws, modelfit) {
  return(sapply(1:nrow(advi_draws),
                function(i) {
                  log_prob(modelfit, advi_draws[i, ])
                }))
}

# Returns draws x parameter dim matrix of gradients.
GetLogLikGradDraws <- function(advi_draws, modelfit) {
  grad_draws <- sapply(1:nrow(advi_draws),
                function(i) {
                  grad_log_prob(modelfit, advi_draws[i, ])
                }, simplify="array")
  return(grad_draws)
}

GetLogLikHessianDraws <- function(advi_draws, modelfit) {
  hessian_draws <- sapply(1:nrow(advi_draws),
                          function(i) {
                            hessian_log_prob(modelfit, advi_draws[i, ])
                          }, simplify="array")
  return(hessian_draws)
}


GetLogLikHVPDraws <- function(advi_draws, modelfit, vec) {
  if (!is.null(dim(vec))) {
      if (length(dim(vec)) == 1) {
          vec <- as.numeric(vec)
      }
  }
  if (is.null(dim(vec))) {
      hvp_draws <- sapply(
          1:nrow(advi_draws),
          function(i) {
            hessian_times_vector_log_prob(modelfit, advi_draws[i, ], vec)
          }, simplify="array")

  } else {
      stopifnot(length(dim(vec)) == 2)
      hvp_draws <- sapply(
          1:nrow(advi_draws),
              function(i) {
                hessian_times_vector_log_prob(modelfit, advi_draws[i, ], vec[i, ])
              }, simplify="array")
  }
  return(hvp_draws)
}


####################
# ADVI objective

# Expected log likelihood gradients with respect to the draws.  I think
# maybe we don't actually need these.

GetExpectedLogLik <- function(advi_draws, modelfit) {
  return(mean(GetLogLikDraws(advi_draws, modelfit)))
}

# GetExpectedLogLikGrad <- function(advi_draws, modelfit) {
#   return(apply(GetLogLikGradDraws(advi_draws, modelfit), MARGIN=1, mean))
# }
#
# GetExpectedLogLikHessian <- function(advi_draws, modelfit) {
#   return(apply(GetLogLikHessianDraws(advi_draws, modelfit), MARGIN=c(1, 2), mean))
# }
#
# GetExpectedLogLikHVP <- function(advi_draws, modelfit, vec) {
#   return(apply(GetLogLikHVPDraws(advi_draws, modelfit, vec), MARGIN=1, mean))
# }


# Entropy gradients with respect to the unconstrained parameters.

GetADVIEntropy <- function(advi_list) {
  return(sum(advi_list$log_sigma))
}

GetADVIEntropyGrad <- function(advi_list) {
  return(c(rep(0, advi_list$upar_length), rep(1, advi_list$upar_length)))
}

# GetADVIEntropyHessian <- function(advi_list) {
#   hess_dim <- 2 * advi_list$upar_length
#   return(matrix(0, nrow=hess_dim, ncol=hess_dim))
# }
#
# GetADVIEntropyHVP <- function(advi_list) {
#   hess_dim <- 2 * advi_list$upar_length
#   return(rep(0, hess_dim))
# }


##############################
# Wrappers for the reparameterization trick

GetKL <- function(advi_list, advi_draws, modelfit) {
    return(-1 * (GetExpectedLogLik(advi_draws, modelfit) +
                 GetADVIEntropy(advi_list)))
}

# Grad wrt the unconstrained parameters through the reparameterization trick.
# You may want the draws to estimate the sample variance of the optimum.
GetKLGradDraws <- function(advi_list, advi_draws, modelfit) {
    num_draws <- nrow(advi_draws)

    dtheta_deta <- advi_draws - rep(advi_list$mean, each=num_draws)
    log_lik_grad_draws <- GetLogLikGradDraws(advi_draws, modelfit)
    log_lik_eta_grad_draws <-
      sapply(1:num_draws,
             function(i) { log_lik_grad_draws[,i] *  dtheta_deta[i, ] },
             simplify="array")
    return(rbind(log_lik_grad_draws, log_lik_eta_grad_draws))
}

# Grad wrt the unconstrained parameters through the reparameterization trick.
GetKLGrad <- function(advi_list, advi_draws, modelfit) {
    num_draws <- nrow(advi_draws)

    dtheta_deta <- advi_draws - rep(advi_list$mean, each=num_draws)
    grad_draws <- GetKLGradDraws(advi_list, advi_draws, modelfit)
    log_lik_grad <- apply(grad_draws, MARGIN=1, mean)
    kl_grad <- -1 * (log_lik_grad + GetADVIEntropyGrad(advi_list))
    return(kl_grad)
}

GetKLHessian <- function(advi_list, advi_draws, modelfit) {
    num_draws <- nrow(advi_draws)

    log_lik_grad_draws <- GetLogLikGradDraws(advi_draws, modelfit)
    log_lik_hessian_draws <- GetLogLikHessianDraws(advi_draws, modelfit)
    dtheta_deta <- advi_draws - rep(advi_list$mean, each=num_draws)

    log_lik_mumu_hessian <- apply(log_lik_hessian_draws, MARGIN=c(1, 2), mean)

    # Derivatives wrt eta is in the rows.
    log_lik_etamu_hessian_draws <-
      sapply(1:num_draws,
             function(i) { log_lik_hessian_draws[,,i] * dtheta_deta[i, ] },
             simplify="array")
    log_lik_etamu_hessian <-
        apply(log_lik_etamu_hessian_draws, MARGIN=c(1, 2), mean)

    log_lik_etaeta_hessian_draws <-
      sapply(1:num_draws,
             function(i) {
                log_lik_hessian_draws[,,i] *
                    (dtheta_deta[i, ] %*% t(dtheta_deta[i, ])) +
                diag(dtheta_deta[i, ] * log_lik_grad_draws[, i])  },
             simplify="array")
    log_lik_etaeta_hessian <-
        apply(log_lik_etaeta_hessian_draws, MARGIN=c(1, 2), mean)

    loglik_hessian <- -1 *
      cbind(rbind(log_lik_mumu_hessian,     log_lik_etamu_hessian),
            rbind(t(log_lik_etamu_hessian), log_lik_etaeta_hessian))

    return(loglik_hessian)
}


GetKLHessianVectorProduct <- function(vec, advi_list, advi_draws, modelfit) {
    num_draws <- nrow(advi_draws)
    vec_mu <- vec[1:advi_list$upar_length]
    vec_eta <- vec[(1 + advi_list$upar_length):(2 * advi_list$upar_length)]

    log_lik_grad_draws <- GetLogLikGradDraws(advi_draws, modelfit)
    dtheta_deta <- advi_draws - rep(advi_list$mean, each=num_draws)
    svec_draws <- sapply(
        1:num_draws,
        function(i) { vec_eta * dtheta_deta[i, ]}, simplify="array") %>% t()

    log_lik_hvp_vec_draws <- GetLogLikHVPDraws(advi_draws, modelfit, vec_mu)
    log_lik_hvp_svec_draws <- GetLogLikHVPDraws(advi_draws, modelfit, svec_draws)
    log_lik_hvp_svec <- apply(log_lik_hvp_svec_draws, MARGIN=1, mean)
    dim(log_lik_hvp_svec_draws)
    dim(log_lik_hvp_svec_draws)
    dim(log_lik_grad_draws)

    hvp_mu <- apply(log_lik_hvp_vec_draws + log_lik_hvp_svec_draws,
                    MARGIN=1, mean)
    hvp_eta <-
      apply(t(dtheta_deta) * (log_lik_hvp_vec_draws + log_lik_hvp_svec_draws),
            MARGIN=1, mean) +
      apply(log_lik_grad_draws * t(svec_draws), MARGIN=1, mean)

    kl_hess_times_vec <- -1 * c(hvp_mu, hvp_eta)
    return(kl_hess_times_vec)
}


# Get KL function functors that can be passed to optimization routines.
GetKLFunctions <- function(advi_list, modelfit, num_draws) {
  norm_draws_opt <- GetNormDraws(advi_list, num_draws=num_draws)

  KLForOpt <- function(advi_par, verbose=TRUE) {
    advi_list <- UnwrapADVIParams(advi_par)
    advi_draws <- GetADVIDraws(advi_list, norm_draws_opt)
    kl <- GetKL(advi_list, advi_draws, modelfit)
    if (verbose) {
      cat("KL: ", kl, "\n")
    }
    return(kl)
  }

  KLGradForOpt <- function(advi_par) {
    advi_list <- UnwrapADVIParams(advi_par)
    advi_draws <- GetADVIDraws(advi_list, norm_draws_opt)
    return(GetKLGrad(advi_list, advi_draws, modelfit))
  }

  KLGradDrawsForOpt <- function(advi_par) {
    advi_list <- UnwrapADVIParams(advi_par)
    advi_draws <- GetADVIDraws(advi_list, norm_draws_opt)
    return(GetKLGradDraws(advi_list, advi_draws, modelfit))
  }

  # ADVI Hessian
  KLHessianForOpt <- function(advi_par) {
    advi_list <- UnwrapADVIParams(advi_par)
    advi_draws <- GetADVIDraws(advi_list, norm_draws_opt)
    return(GetKLHessian(advi_list, advi_draws, modelfit))
  }

  # ADVI Hessian vector product
  KLHVPForOpt <- function(advi_par, vec) {
    advi_list <- UnwrapADVIParams(advi_par)
    advi_draws <- GetADVIDraws(advi_list, norm_draws_opt)
    return(GetKLHessianVectorProduct(vec, advi_list, advi_draws, modelfit))
  }

  return(list(KLForOpt=KLForOpt,
              KLGradForOpt=KLGradForOpt,
              KLGradDrawsForOpt=KLGradDrawsForOpt,
              KLHessianForOpt=KLHessianForOpt,
              KLHVPForOpt=KLHVPForOpt,
              norm_draws_opt=norm_draws_opt,
              num_draws=num_draws,
              advi_list=advi_list,
              modelfit=modelfit))
}

# GetExpectedLogProbGrad <- function(advi_list, advi_draws, modelfit) {
#   advi_scale_draws <- advi_draws - rep(advi_list$mean, each=nrow(advi_draws))
#   log_lik_grad_draws <- GetLogLikGradDraws(advi_list, advi_draws, modelfit)
#   log_lik_mu_grad <- apply(log_lik_grad_draws, MARGIN=1, mean)
#   log_lik_sigma_grad <-
#     apply(log_lik_grad_draws * t(advi_scale_draws), MARGIN=1, mean)
#   return(c(log_lik_mu_grad, log_lik_sigma_grad))
# }


#############################
# Converting between vector and list representations of the ADVI parameters.

UnwrapADVIParams <- function(advi_par) {
  # The vector must have mean and standard deviation parameters, so its
  # length must be an even number.
  stopifnot(length(advi_par) %% 2 == 0)
  upar_len <- length(advi_par) / 2
  advi_mean <- advi_par[1:upar_len]
  advi_log_sigma <- advi_par[(upar_len + 1):(2 * upar_len)]
  return(list(upar_length=upar_len, mean=advi_mean, log_sigma=advi_log_sigma))
}

WrapADVIParams <- function(advi_list) {
  advi_par <- rep(NA, 2 * advi_list$upar_length)
  advi_par[1:advi_list$upar_length] <- advi_list$mean
  advi_par[(advi_list$upar_length + 1):(2 * advi_list$upar_length)] <-
    advi_list$log_sigma
  return(advi_par)
}

# # Test
# WrapADVIParams(advi_list)
# advi_par <- WrapADVIParams(advi_list)
# advi_list_test <- UnwrapADVIParams(advi_par)
# stopifnot(max(abs(advi_list$mean - advi_list_test$mean)) < 1e-8)
# stopifnot(max(abs(advi_list$log_sigma - advi_list_test$log_sigma)) < 1e-8)

WrapKL <- function(advi_par, norm_draws, modelfit, verbose=FALSE) {
  advi_list <- UnwrapADVIParams(advi_par)
  advi_draws <- GetADVIDraws(advi_list, norm_draws)
  elbo <- (GetExpectedLogLik(advi_draws, modelfit) +
           GetADVIEntropy(advi_list))
  if (verbose) {
    #print(advi_list)
    cat("KL: ", -1 * elbo, "\n")
  }
  return(-1 * elbo)
}


########################
# MAP

GetMAPObjectives <- function(modelfit) {
  return(list(
    modelfit=modelfit,
    f=function(upar) { -1 * rstan::log_prob(modelfit, upar) },
    grad=function(upar) { -1 * rstan::grad_log_prob(modelfit, upar) },
    hessian=function(upar) { -1 * rstan::hessian_log_prob(modelfit, upar) },
    hvp=function(upar, vec) { -1 * rstan::hessian_times_vector_log_prob(
        modelfit, upar, vec) }
  ))
}




###############################
# Getting sample variance of the KL optimum

# Variance of the KL optimizer
EstimateSamplingCovariance <- function(advi_list, kl_hess, kl_funcs) {
  klgrad_draws <- kl_funcs$KLGradDrawsForOpt(WrapADVIParams(advi_list))
  klgrad_cov <- cov(t(klgrad_draws))
  kl_hess_inv <- solve(kl_hess)
  par_cov <- solve(kl_hess, klgrad_cov) %*% kl_hess_inv
  return(par_cov)
}

# GetEtaSamplingCovariance <- function(advi_list, modelfit, num_draws) {
#   kl_funcs_for_var <- GetKLFunctions(advi_list, modelfit, num_draws=num_draws)
#   advi_draws <- GetADVIDraws(advi_list, kl_funcs_for_var$norm_draws_opt)
#   klgrad_draws <- GetKLGradDraws(advi_list, advi_draws, modelfit)
#   klgrad_cov <- cov(t(klgrad_draws))
#   kl_hess <- kl_funcs_for_var$KLHessianForOpt(WrapADVIParams(advi_list))
#   kl_hess_inv <- solve(kl_hess)
#   par_cov <- solve(kl_hess, klgrad_cov) %*% kl_hess_inv
#   return(par_cov)
# }

# GetSampleErrorDataframe <- function(advi_list, par_cov) {
#   mu_ind <- 1:advi_list$upar_length
#   eta_ind <- mu_ind + advi_list$upar_length
#   sample_error_df <-
#     data.frame(mu_sd=sqrt(diag(par_cov)[mu_ind]),
#                eta_sd=sqrt(diag(par_cov)[eta_ind]),
#                vb_sd=exp(advi_list$log_sigma))
#   return(sample_error_df)
# }

GetSampleErrorDataframe <- function(advi_list, par_cov, lrvb_cov) {
  mu_ind <- 1:advi_list$upar_length
  eta_ind <- mu_ind + advi_list$upar_length
  sample_error_df <-
    data.frame(mu_sd=sqrt(diag(par_cov)[mu_ind]),
               eta_sd=sqrt(diag(par_cov)[eta_ind]),
               vb_sd=exp(advi_list$log_sigma),
               lrvb_sd=sqrt(diag(lrvb_cov)[mu_ind]))
  return(sample_error_df)
}

GetRequiredNumberOfSamples <- function(sample_error_df, accuracy) {
  #worst_case_var <- as.numeric(quantile(diag(par_cov), 1.0))
  worst_case_var <- max(sample_error_df$mu_sd / sample_error_df$vb_sd) ^ 2
  req_num_samples <- ceiling(worst_case_var / (accuracy ^ 2))
  return(req_num_samples)
}

# For the target quantile of parameters, we want n draws such that
# mu_sd / sqrt(n) > accuracy * lrvb_sd =>
# n > (mu_sd  / (accuracy * lrvb_sd)) ^ 2
ChooseNumSamples <- function(
        advi_list, kl_hess, kl_funcs, lrvb_cov,
        accuracy=0.5, target_quantile=1) {

    par_cov <- EstimateSamplingCovariance(advi_list, kl_hess, kl_funcs)
    sample_error_df <- GetSampleErrorDataframe(advi_list, par_cov, lrvb_cov)
    worst_case_sd <-
        quantile(sample_error_df$mu_sd / sample_error_df$lrvb_sd,
                 target_quantile)
    req_num_samples <- ceiling((worst_case_sd / accuracy) ^ 2)
    return(req_num_samples)
}


####################
# Constrain a matrix to be positive definite.

MakeMatrixPD <- function(mat) {
  mat_eigen <- eigen(mat)
  new_evs <- mat_eigen$values
  new_evs[new_evs <= 0] <- 0
  new_mat <-  mat_eigen$vectors %*% diag(new_evs) %*% solve(mat_eigen$vectors)
  return(new_mat)
}

if (FALSE) {
    # TEST
  mat <- diag(2)
  mat[1,1] <- -0.1
  mat[2,1] <- mat[1,2] <- 0.2
  mat_eigen <- eigen(mat)
  stopifnot(max(abs(
    mat -
      mat_eigen$vectors %*% diag(mat_eigen$values) %*% solve(mat_eigen$vectors))) < 1e-8)
  MakeMatrixPD(mat)

  mat <- diag(2)
  mat[1,1] <- 0.1
  mat[2,1] <- mat[1,2] <- 0.2
  MakeMatrixPD(mat) - mat
}
