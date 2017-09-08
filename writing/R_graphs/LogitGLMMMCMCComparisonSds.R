

sd_results <-
  filter(glmm_env$results, metric == "sd") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=20) %>%
  inner_join(glmm_expression_df, by=c("par", "component"))

main_par <- c("e_beta", "e_mu", "e_tau")

g1 <- MeanComparisonGraph(
  filter(sd_results, par %in% main_par), TeX("Uncorrected MFVB sd: global parameters"),
  y_colname="mfvb", y_label="MFVB")

g2 <- MeanComparisonGraph(
  filter(sd_results, par == "e_u"), TeX("Uncorrected MFVB sd: random effects"),
  y_colname="mfvb", y_label="MFVB")


g3 <- MeanComparisonGraph(
  filter(sd_results, par %in% main_par), TeX("LRVB sd.: global parameters"),
  y_colname="lrvb", y_label="LRVB")

g4 <- MeanComparisonGraph(
  filter(sd_results, par == "e_u"), TeX("LRVB sd: random effects"),
  y_colname="lrvb", y_label="LRVB")

grid.arrange(g1, g2, g3, g4, ncol=2)
