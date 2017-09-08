library(xtable)

mean_df <-
  filter(glmm_env$mean_table) %>%
  mutate(component=as.character(component))

table_df <-
  inner_join(mean_df, glmm_table_expression_df, by=c("par", "component")) %>%
  select(par_tex, mcmc, mfvb, map, stan_std_err, n_eff) %>%
  rename(Parameter=par_tex,
         MCMC=mcmc,
         MFVB=mfvb,
         MAP=map,
         "Eff. # of MCMC draws"=n_eff,
         "MCMC std. err."=stan_std_err)

cat("\\begin{center}")
print(xtable(table_df, digits=c(0, 0, 3, 3, 3, 5, 0)),
      include.rownames=FALSE, floating=FALSE,
      sanitize.text.function=function(x) { x },
      sanitize.colnames.function=NULL)
cat("\\end{center}")
