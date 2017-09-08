library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(xtable)
library(gridExtra)
library(scales)
library(png)
library(latex2exp)
library(Matrix)

paper_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "VariationalRobustBayesPaper")

opts_chunk$set(fig.pos='!h', fig.align='center', dev='png', dpi=300)
knitr_debug <- FALSE
opts_chunk$set(echo=knitr_debug, message=knitr_debug, warning=knitr_debug)

# Set the default ggplot theme
theme_set(theme_bw())

# Turn off caching if you need to regenerate any of the R code.
opts_chunk$set(cache=FALSE)

# Caching for individual analyses
glmm_cache <- FALSE

# Load into an environment rather than the global space
LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

# The location of data for this paper.
data_path <- file.path(paper_directory, "writing/data/")

# A convenient funciton for extracting only the legend from a ggplot.
# Taken from
# http://www.sthda.com/english/wiki/ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page-r-software-and-data-visualization
get_legend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}
