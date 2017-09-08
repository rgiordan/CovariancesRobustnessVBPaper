
sd_results <-
  filter(glmm_env$results, metric == "sd") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=100) %>%
  inner_join(glmm_expression_df, by=c("par", "component")) %>%
  filter(!(par %in% c("e_tau", "e_log_tau")))

main_par <- c("e_beta", "e_mu", "e_tau", "e_log_tau")

GlmerSdComparisonGraph <- function(this_df, title) {
  MeanComparisonGraph(this_df=this_df,
                      y_colname="glmer",
                      y_label="glmer MLE",
                      legend_name="Std. dev. of:",
                      title=title)
}

g1 <- GlmerSdComparisonGraph(filter(sd_results, par %in% main_par), title="Global Parameters")
g2 <- GlmerSdComparisonGraph(filter(sd_results, !(par %in% main_par)), title="Random effects")

grid.arrange(g1, g2, ncol=2)
