mean_results <-
  filter(glmm_env$results, metric == "mean") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=20) %>%
  inner_join(glmm_expression_df, by=c("par", "component"))

main_par <- c("e_beta", "e_mu")
tau_par <- c("e_tau")
re_pars <- c("e_u")

# Rename the mfvb column so I can use the same graph function to plot the MAP estimates.
g1 <- MeanComparisonGraph(
  filter(mean_results, par %in% main_par) %>% mutate(mfvb=map),
  TeX("MAP: location parameters"),
  y_label="MAP")

g2 <- MeanComparisonGraph(
  filter(mean_results, par %in% tau_par) %>% mutate(mfvb=map),
  title=TeX("MAP: $\\tau$"),
  y_label="MAP")

g3 <- MeanComparisonGraph(
  filter(mean_results, par %in% c(re_pars)) %>% mutate(mfvb=map),
  TeX("MAP: random effects"), y_label="MAP")

grid.arrange(g1, g2, g3, ncol=3)