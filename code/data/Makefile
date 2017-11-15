# Run the analyses for the paper.

SHELL = /bin/bash

GIT_REPO_LOC := $(shell git rev-parse --show-toplevel)
R_SCRIPT_LOC := $(GIT_REPO_LOC)/code/R/
PY_SCRIPT_LOC := $(GIT_REPO_LOC)/code/python/


all: criteo_subsampled_python_vb_results.pkl, \
	 criteo_subsampled_mcmc_draws.Rdata, \

	echo "READY TO GO"

criteo_subsampled_python_vb_map_results.pkl: criteo_subsampled_python_vb_results.pkl
	$(PY_SCRIPT_LOC)/rub_map.py

criteo_subsampled_python_vb_results.pkl: criteo_subsampled_stan_dat.json
	$(PY_SCRIPT_LOC)/rub_vb.py

criteo_subsampled_mcmc_draws.Rdata: criteo_subsampled_stan_dat.json
	$(R_SCRIPT_LOC)/run_stan.R

criteo_subsampled_stan_dat.json: criteo/criteo_data_for_paper.Rdata
	$(R_SCRIPT_LOC)/create_datasets.R

criteo/criteo_data_for_paper.Rdata:
	$(MAKE) -C criteo criteo_data_for_paper.Rdata
