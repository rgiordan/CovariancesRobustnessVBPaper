library(xtable)

sd_df <-
  filter(glmm_env$sd_table) %>%
  mutate(component=as.character(component))

table_df <-
  inner_join(sd_df, glmm_table_expression_df, by=c("par", "component")) %>%
  select(par_tex, mcmc, lrvb, mfvb) %>%
  rename(Parameter=par_tex,
         MCMC=mcmc,
         LRVB=lrvb,
         "Uncorrected MFVB"=mfvb)

cat("\\begin{center}")
print(xtable(table_df, digits=c(0, 0, 3, 3, 3)),
      include.rownames=FALSE, floating=FALSE,
      sanitize.text.function=function(x) { x },
      sanitize.colnames.function=NULL)
cat("\\end{center}")
