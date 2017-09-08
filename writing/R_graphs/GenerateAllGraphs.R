paper_directory <- file.path(Sys.getenv("GIT_REPO_LOC"), "CovariancesRobustnessVBPaper")
script_directory <- file.path(paper_directory, "writing/R_graphs")

source(file.path(script_directory, "Initialize.R"))
source(file.path(script_directory, "CommonGraphs.R"))
system(sprintf("ls %s/*.R", script_directory))


source(file.path(script_directory, "LogitGLMMLoadData.R"))

source(file.path(script_directory, "LogitGLMMMapComparisonMeans.R"))
source(file.path(script_directory, "LogitGLMMGlmerComparisonMeans.R"))
source(file.path(script_directory, "LogitGLMMGlmerComparisonSds.R"))

source(file.path(script_directory, "LogitGLMMMCMCComparisonMeans.R"))
source(file.path(script_directory, "LogitGLMMMCMCComparisonSds.R"))

source(file.path(script_directory, "LogitGLMMParametricRobustness.R"))
