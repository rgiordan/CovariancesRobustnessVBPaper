
mean_results <-
  filter(glmm_env$results, metric == "mean") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=20) %>%
  inner_join(glmm_expression_df, by=c("par", "component"))

main_par <- c("e_beta", "e_mu", "e_tau")
#main_par <- "e_beta"

g1 <- MeanComparisonGraph(
  filter(mean_results, par %in% main_par), TeX("VB means: global parameters"),
  y_colname="mfvb", y_label="mfvb")

g2 <- MeanComparisonGraph(
  filter(mean_results, par == "e_u"), TeX("VB  means: random effects"),
  y_colname="mfvb", y_label="mfvb")


grid.arrange(g1, g2, ncol=2)
