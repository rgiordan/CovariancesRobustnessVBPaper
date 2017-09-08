
# Show the same u components that were in the table.
random_u_components <- unique(filter(glmm_env$mean_table, par=="e_u")$component)

graph_df <-
  filter(glmm_env$pert_result_diff_df,
         par == "e_mu" |
         par == "e_u" & component %in% random_u_components[1] |
         par == "e_beta" & component %in% c(1, 2)) %>%
  mutate(par=as.character(par), component=as.character(component)) %>%
  inner_join(glmm_table_expression_df, by=c("par", "component"))

# Get the TeX for the prior parameter that was perturbed.
prior_tex <- 
  as.character(
    inner_join(tibble(prior_par=as.character(unique(graph_df$par_prior))),
             glmm_prior_expression_df, by="prior_par")$pp_tex)
stopifnot(length(prior_tex) == 1)
  
# I cannot figure out how to use latex2exp in the facet grid labels. :(
xlab_tex <- paste0("$\\Delta$", prior_tex)
unique_par_tex <- as.character(unique(graph_df$par_tex))
max_epsilon <- max(graph_df$epsilon)
max_diff <- min(min(graph_df$pred_diff), min(graph_df$diff))
PlotSinglePar <- function(ind) {
  this_par_tex <- unique_par_tex[[ind]]
  this_graph_df <-
    filter(graph_df, par_tex == this_par_tex)
  
  ylab_tex <- paste0("$\\Delta E_{q}\\[$", this_par_tex, "$\\]$")
  
  ggplot(this_graph_df) +
    geom_point(aes(x=epsilon, y=diff, color="actual")) +
    geom_line(aes(x=epsilon, y=diff, color="actual")) +
    geom_line(aes(x=epsilon, y=pred_diff, color="predicted")) +
    theme(legend.position="none") +
    ggtitle(TeX(paste0("$E_{q}\\[$", this_par_tex, "$\\]$ sensitivity"))) +
    expand_limits(x=0, y=0) +
    expand_limits(x=max_epsilon, y=max_diff) +
    ylab(TeX(ylab_tex)) +
    xlab(TeX(xlab_tex))
}

grid.arrange(
  PlotSinglePar(1),
  PlotSinglePar(2),
  PlotSinglePar(3),
  PlotSinglePar(4),
  ncol=4
)
