library(dplyr)
library(ggplot2)
library(reshape2)

project_directory <- file.path(
  Sys.getenv("GIT_REPO_LOC"),
  "VariationalBayesPythonWorkbench/Models/LogisticGLMM")
data_directory <- file.path(project_directory, "data/")

criteo_dir <- file.path(
  Sys.getenv("GIT_REPO_LOC"), "criteo/criteo_conversion_logs/")
clean_data_filename <- file.path(criteo_dir, "data_clean.Rdata")
load(clean_data_filename)

d_counts <-
  group_by(d_clean, V11) %>%
  summarize(num_in_group=n()) %>%
  ungroup() %>% group_by(num_in_group) %>%
  summarize(num=n())

mutate(d_counts, small=num_in_group <= 20) %>%
  ungroup() %>% group_by(small) %>%
  summarize(num=sum(num))

small_group_size <- 20
d_groups <- 
  group_by(d_clean, V11) %>%
  summarize(num_in_group=n()) %>%
  mutate(small=num_in_group <= small_group_size)

d_clean_sized <- inner_join(d_clean, d_groups, by="V11")

# This is a little time consuming.
d_large <-
  filter(d_clean_sized, !small) %>%
  ungroup() %>% group_by(V11) %>%
  sample_n(small_group_size) %>%
  select(-n, -num_in_group)

d_sub <- rbind(d_large, filter(d_clean_sized, small))
unique_groups <- tibble(V11=unique(d_sub$V11))
sampled_groups <- sample_n(unique_groups, 5000)
d_sub <- inner_join(d_sub, sampled_groups, by="V11")
nrow(d_sub)

y <- as.integer(d_sub$conversion)
regressors <- paste("V", c(4, 5, 7, 9, 10), sep="")
x <- as.matrix(d_sub[regressors])
colnames(x) <- paste("x", 1:ncol(x), sep=".")
y_g_orig <- factor(d_sub$V11)
y_g <- as.integer(y_g_orig) - 1

save(y, y_g, x, file=file.path(data_directory, "criteo_data_for_paper.Rdata"))

