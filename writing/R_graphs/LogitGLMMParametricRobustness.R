prior_results <-
  filter(glmm_env$sens_df_cast, metric == "prior_sensitivity_norm") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=20) %>%
  rename(prior_par=par_prior) %>%
  mutate(prior_par=as.character(prior_par)) 

prior_results_graph <-
  inner_join(prior_results, glmm_expression_df, by=c("par", "component")) %>%
  inner_join(glmm_prior_expression_df, by=c("prior_par"))

main_par <- c("e_beta", "e_mu", "e_tau")
g1 <- ParametricSensitivityGraph(
  filter(prior_results_graph, par %in% main_par),
  title="Norm. sensitivity: global parameters")
g2 <- ParametricSensitivityGraph(
  filter(prior_results_graph, par == "e_u"),
  title="Norm. sensitivity: random effects")

grid.arrange(g1, g2, ncol=2)
