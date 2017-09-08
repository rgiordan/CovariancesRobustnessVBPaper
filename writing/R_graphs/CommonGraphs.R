# This file made more sense when I was producing the same graph for
# multiple experiments.

abline_graph <- geom_abline(aes(slope=1, intercept=0), linetype="dotted")

MeanComparisonGraph <- function(this_df, title,
                                y_label="VB",
                                legend_name="Posterior\nmean of:",
                                y_colname="mfvb") {
  this_df$mean_val <- this_df[[y_colname]]
  mcmc_max <- max(this_df$mcmc)
  ggplot(this_df) +
    geom_point(aes(x=mcmc, y=mean_val, shape=par_tex), size=3) +
    scale_shape_discrete(name=legend_name,
                         breaks=unique(this_df$par_tex),
                         labels=TeX(unique(this_df$par_tex))) +
    expand_limits(x=0, y=0) +
    expand_limits(x=mcmc_max, y=mcmc_max) +
    abline_graph + ylab(y_label) + xlab("MCMC") +
    ggtitle(title)
}

StdevComparisonGraph <- function(this_df, title, point_size=3) {
  ggplot(this_df) +
    geom_segment(aes(x=mcmc, xend=mcmc, y=lrvb, yend=mfvb), color="gray") +
    geom_point(aes(x=mcmc, y=mfvb, color="mfvb", shape=par_tex), size=point_size) +
    geom_point(aes(x=mcmc, y=lrvb, color="lrvb", shape=par_tex), size=point_size) +
    scale_color_discrete(name="VB method") +
    scale_shape_discrete(name="Posterior\nstd. dev. of:",
                         breaks=unique(this_df$par_tex),
                         labels=TeX(unique(this_df$par_tex))) +
    guides(color=guide_legend(order=1), shape=guide_legend(order=2)) +
    expand_limits(x=0, y=0) + abline_graph + ylab("VB") + xlab("MCMC") +
    ggtitle(title)
}

ParametricSensitivityGraph <- function(this_df, title) {
  ggplot(this_df) +
    geom_point(aes(x=mcmc, y=lrvb, color=par_tex, shape=pp_tex), size=2, stroke=1.2) +
    expand_limits(x=0, y=0) + abline_graph +
    scale_shape_manual(name="Prior Parameter",
                         breaks=unique(this_df$pp_tex),
                         values=1:length(unique(this_df$pp_tex)),
                         labels=TeX(unique(this_df$pp_tex))) +
    scale_colour_grey(name="Parameter",
                         breaks=unique(this_df$par_tex),
                         labels=TeX(unique(this_df$par_tex))) +
    guides(color=guide_legend(order=1), shape=guide_legend(order=2)) +
    xlab("MCMC") + ylab("VB") + ggtitle(title)
}
