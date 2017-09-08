glmm_analysis_name <- "criteo_subsampled"
glmm_env <- LoadIntoEnvironment(
    file.path(data_path, paste(glmm_analysis_name, "sensitivity.Rdata", sep="_")))

CleanGLMMRandomEffectColumn <- function(df, keep_num_u=20) {
  return(
    filter(df, par != "e_u" | component < keep_num_u) %>%
    mutate(par=as.character(par),
           component=as.character(component)) %>%
    mutate(component=case_when(is.na(component) ~ "-1",
                               par == "e_u" ~ "-1",
                               TRUE ~ component))
  )
}


# Define names for the parameters in the LaTeX graphs.

ExpressionRow <- function(par, component, par_tex) {
  data.frame(par=par, component=component, par_tex=par_tex)
}

# Expressions for parameters in graphs
glmm_expression_list <- list()
glmm_expression_list[[length(glmm_expression_list) + 1]] <-
  rbind(
    ExpressionRow("e_mu", "-1", "$\\mu$"),
    ExpressionRow("e_tau", "-1", "$\\tau$"),
    ExpressionRow("e_log_tau", "-1", "$\\log(\\tau)$"),
    ExpressionRow("e_u", "-1", "$u_{t}$")
  )

for (k in 1:glmm_env$beta_dim) {
  glmm_expression_list[[length(glmm_expression_list) + 1]] <-
    ExpressionRow("e_beta", as.character(k), "$\\beta_{k}$")
    # ExpressionRow("beta", as.character(k), sprintf("$\\beta_{%d}$", k))
}

glmm_expression_df <- do.call(rbind, glmm_expression_list)
glmm_expression_df$par <- as.character(glmm_expression_df$par)
glmm_expression_df$component <- as.character(glmm_expression_df$component)


PriorExpressionRow <- function(metric, pp_tex) {
  data.frame(prior_par=metric, pp_tex=pp_tex)
}

glmm_prior_expression_df <- rbind(
  PriorExpressionRow("beta_prior_mean", "$\\beta_{0}$"),
  PriorExpressionRow("beta_prior_info_diag", "$\\tau_{\\beta}$"),
  PriorExpressionRow("beta_prior_info", "$\\gamma_{\\beta}$"),
  PriorExpressionRow("mu_prior_mean", "$\\mu_0$"),
  PriorExpressionRow("mu_prior_info", "$\\tau_{\\mu}$"),
  PriorExpressionRow("tau_prior_alpha", "$\\alpha_{\\tau}$"),
  PriorExpressionRow("tau_prior_beta", "$\\beta{\\tau}$")
)
glmm_prior_expression_df$prior_par <- as.character(glmm_prior_expression_df$prior_par)


# Expressions for parameters in tables
glmm_table_expression_list <- list()
glmm_table_expression_list[[length(glmm_expression_list) + 1]] <-
  rbind(
    ExpressionRow("e_mu", NA, "$\\mu$"),
    ExpressionRow("e_tau", NA, "$\\tau$"),
    ExpressionRow("e_log_tau", "-1", "$\\log(\\tau)$")
  )

for (k in 1:glmm_env$beta_dim) {
  glmm_table_expression_list[[length(glmm_table_expression_list) + 1]] <-
    ExpressionRow("e_beta", as.character(k), sprintf("$\\beta_{%d}$", k))
}

for (k in 1:glmm_env$num_groups) {
  glmm_table_expression_list[[length(glmm_table_expression_list) + 1]] <-
    ExpressionRow("e_u", as.character(k), sprintf("$u_{%d}$", as.integer(k)))
}

glmm_table_expression_df <- do.call(rbind, glmm_table_expression_list)
glmm_table_expression_df <-
  mutate(glmm_table_expression_df,
         par=as.character(par), component=as.character(component))

