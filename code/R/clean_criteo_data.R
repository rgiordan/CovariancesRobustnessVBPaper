library(dplyr)
library(ggplot2)
library(reshape2)

working_dir <- "/home/rgiordan/Documents/git_repos/criteo/criteo_conversion_logs/"

reload_data <- FALSE
rdata_filename <- file.path(working_dir, "data.Rdata")
if (reload_data) {
  d <- read.delim(file.path(working_dir, "data.tsv"), sep="\t", header=FALSE)
  save(d, file=rdata_filename)
} else {
  load(rdata_filename)
}


head(d)
d$conversion <- is.finite(d$V2)
table(d$conversion) / nrow(d)

regressor_cols <- paste("V", 3:10, sep="")
factor_cols <- paste("V", 11:19, sep="")

##########################
# Look at the factors and pick one to use.

for (col in factor_cols) {
  cat(col, " has length ", length(unique(d[[col]])), "\t")  
  cat("with this many NAs: ", sum(is.na(d[[col]])) / nrow(d), "\n")
}

row_dist <-
  mutate(d, factor_col=V11) %>%
  filter(!is.na(factor_col)) %>%
  group_by(factor_col) %>%
  summarize(conversions=sum(conversion), rows=n()) %>%
  ungroup() %>% group_by(rows) %>%
  summarize(count=n(), conversions=sum(conversions))

conv_dist <-
  mutate(d, factor_col=V11) %>%
  filter(!is.na(factor_col)) %>%
  group_by(factor_col) %>%
  summarize(conversions=sum(conversion), rows=n()) %>%
  ungroup() %>% group_by(conversions) %>%
  summarize(count=n(), rows=sum(rows))

if (FALSE) {
  ggplot(filter(conv_dist, conversions > 0)) +
    geom_point(aes(x=conversions, y=count)) +
    geom_hline(aes(yintercept=1)) +
    scale_x_log10() + scale_y_log10()
  
  ggplot(filter(row_dist)) +
    geom_point(aes(x=rows, y=count)) +
    geom_hline(aes(yintercept=1)) +
    scale_x_log10() + scale_y_log10()
}

conv_remain <- sum(filter(row_dist, rows < 100)$conversions)
row_remain <- sum(filter(row_dist, rows < 100)$rows)

conv_remain / sum(row_dist$conversions)
row_remain / sum(row_dist$rows)

keep_factors <- "V11"

#############
# Look at the regressors.

for (col in regressor_cols) {
  cat(col, " has length ", length(unique(d[[col]])), "\t")  
  cat("with this many NAs: ", sum(is.na(d[[col]]))  / nrow(d), "\n")
  print(summary(d[[col]]))
  cat("\n")
}

# Keep the ones with few NAs.
keep_regressors <- paste("V", c(4, 5, 7, 9, 10), sep="")

if (FALSE) {
  d_melt <-
    melt(d[, keep_regressors]) %>%
    group_by(variable, value) %>%
    summarize(count=n())

  ggplot(filter(d_melt)) +
    geom_line(aes(x=value, y=log10(count))) +
    facet_grid(~ variable, scales="free")
}


###########
# Make a clean dataset.

# Not worth aggregating.
# d_clean <-
#   d[, c("conversion", keep_regressors, keep_factors)] %>%
#   group_by(conversion, V4, V5, V7, V9, V10, V11) %>%
#   summarize(n=n())
# 
# nrow(d_clean) / nrow(d)

d_clean <-
  d[, c("conversion", keep_regressors, keep_factors)] %>%
  group_by(V11) %>%
  mutate(n=n()) %>%
  filter(n <= 300)

for (col in keep_regressors) {
  d_clean <- d_clean[is.finite(d_clean[[col]]), ]
}

nrow(d_clean) / nrow(d)

# It looks like the unique values have been bucketed 
if (FALSE) {
  hist(unique(d_clean$V5), 10)
}

# Note that this will separate the same integer into distinct normal values,
# effectively "randomizing" repeated values.
NormalizeVariable <- function(x) {
  x <- ordered(x)
  x_quantile <- as.numeric(x) / (length(levels(x)) + 1)
  return(qnorm(x_quantile))
}

for (col in keep_regressors) {
  print(col)
  d_clean[[col]] <- NormalizeVariable(d_clean[[col]])
}

if (FALSE) {
  # There is at least superficial evidence that a logit regression will be worthwhile.
  d_melt <- 
    ungroup(d_clean) %>%
    melt(id.vars=c("V11", "conversion", "n"))

  ggplot(d_melt) +
    geom_density(aes(x=value, y=..density.., color=conversion)) +
    facet_grid(variable ~ ., scales="free")
}


save_filename <- file.path(working_dir, "data_clean.Rdata")
save(d_clean, file=save_filename)

