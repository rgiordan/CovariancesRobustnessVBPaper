

cov_results <-
  filter(glmm_env$cov_results_df) %>%
  rename(par=par_1, component=component_1) %>%
  CleanGLMMRandomEffectColumn(keep_num_u=50) %>%
  inner_join(glmm_expression_df, by=c("par", "component"))

main_par <- c("e_beta", "e_mu", "e_tau")

g1 <- MeanComparisonGraph(
  filter(cov_results, par %in% main_par,
         par != par_2 | component != component_2),
  title=TeX("LRVB: Cov(global params, *)"),
  legend_name="Posterior\ncov. with:",
  y_colname="lrvb", y_label="LRVB")

g2 <- MeanComparisonGraph(
  filter(cov_results, par == "e_u", par_2 == "e_u",
         component != component_2),
  title=TeX("LRVB: Cov(u, u)"),
  legend_name="Posterior\ncov. with:",
  y_colname="lrvb", y_label="LRVB")

grid.arrange(g1, g2, ncol=2)
