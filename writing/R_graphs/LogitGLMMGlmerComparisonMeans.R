
mean_results <-
  filter(glmm_env$results, metric == "mean") %>%
  CleanGLMMRandomEffectColumn(keep_num_u=100) %>%
  inner_join(glmm_expression_df, by=c("par", "component"))

main_par <- c("e_beta", "e_mu", "e_tau", "e_log_tau")

GlmerMeanComparisonGraph <- function(this_df, title) {
  MeanComparisonGraph(this_df=this_df,
                      y_colname="glmer",
                      y_label="glmer MLE",
                      title=title)
}

g1 <- GlmerMeanComparisonGraph(filter(mean_results, par %in% main_par),
                               title="Global parameters")
g2 <- GlmerMeanComparisonGraph(filter(mean_results, !(par %in% main_par)),
                               title="Random effects")

grid.arrange(g1, g2, ncol=2)
