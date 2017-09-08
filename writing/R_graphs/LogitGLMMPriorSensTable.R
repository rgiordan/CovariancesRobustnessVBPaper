library(xtable)

random_u_components <- as.character(unique(
  filter(glmm_env$mean_table, par == "e_u")$component))

prior_results <-
  filter(glmm_env$sens_df_cast, metric == "prior_sensitivity_norm") %>%
  mutate(par=as.character(par),
         component=as.character(component)) %>%
  filter(par != "e_u" | component %in% random_u_components) %>%
  rename(prior_par=par_prior) %>%
  mutate(prior_par=as.character(prior_par)) %>%
  inner_join(glmm_table_expression_df, by=c("par", "component")) %>%
  inner_join(glmm_prior_expression_df, by=c("prior_par"))


table_df <- dcast(prior_results, par_tex ~ pp_tex, value.var="lrvb") %>%
  rename(" "=par_tex)

cat("\\begin{center}")
print(xtable(table_df, digits=rep(4, ncol(table_df) + 1)),
      include.rownames=FALSE, floating=FALSE,
      sanitize.text.function=function(x) { x })
cat("\\end{center}")
