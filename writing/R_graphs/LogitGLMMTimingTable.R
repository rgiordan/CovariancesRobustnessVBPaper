
GetTimingRow <- function(method, time, unit="seconds") {
  return(data.frame(Method=method, Seconds=time))
}

table_df <- rbind(
  GetTimingRow(method="MAP (optimum only)", time=glmm_env$map_time),
  GetTimingRow(method="VB (optimum only)", time=glmm_env$vb_time),
  GetTimingRow(method="VB (including sensitivity for $\\beta$)",
               time=glmm_env$vb_time + glmm_env$beta_dim * glmm_env$cg_row_time),
  GetTimingRow(method="VB (including sensitivity for $\\beta$ and $u$)",
               time=glmm_env$vb_time + glmm_env$hess_time + glmm_env$inverse_time),
  GetTimingRow(method="MCMC (Stan)", time=glmm_env$mcmc_time)
)

# I'm going to leave glmer out of the paper just to avoid muddying the exposition.
# The differences are probably due to lme4 not having priors and using worse
# optimziation methods rather than anything fundamental about MMLE.
#GetTimingRow(method="Marginal maximum likelihood (glmer)", time=glmm_env$glmer_time)

cat("\\begin{center}")
print(xtable(table_df, digits=c(0, 0, 0)),
      include.rownames=FALSE, floating=FALSE,
      sanitize.text.function=function(x) { x },
      sanitize.colnames.function=NULL)
cat("\\end{center}")
