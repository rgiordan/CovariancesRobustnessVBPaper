#!/usr/bin/env Rscript

git_repo <- system("git rev-parse --show-toplevel", intern=TRUE)
working_dir <- file.path(git_repo, "code/criteo_experiment/data/criteo")
raw_data_filename <- file.path(working_dir, "data.txt")
rdata_filename <- file.path(working_dir, "data.Rdata")

cat("Loading data from", raw_data_filename, "(This may take a while.)\n")
d <- read.delim(raw_data_filename, sep="\t", header=FALSE)

cat("Saving data to", rdata_filename, "\n")
d$conversion <- is.finite(d$V2)
save(d, file=rdata_filename)
