# Run the analyses for the paper.
#
# Note that ExecutePreprocessor.timeout sets the cell timeout in seconds.
# You may want to set this to be a larger value depending on your computes.

SHELL = /bin/bash

GIT_REPO_LOC := $(shell git rev-parse --show-toplevel)
R_SCRIPT_LOC := $(GIT_REPO_LOC)/code/criteo_experiment/R/
PY_SCRIPT_LOC := $(GIT_REPO_LOC)/code/criteo_experiment/python/
VENV_LOC := $(GIT_REPO_LOC)/code/criteo_experiment/venv/bin/activate

all: criteo_subsampled_python_vb_results.pkl \
	 criteo_subsampled_python_vb_map_results.pkl \
	 criteo_subsampled_python_vb_perturbation_results.pkl \
	 criteo_subsampled_mcmc_draws.Rdata
	@echo "READY TO GO"

clean:
	rm -f criteo_subsampled_python_vb_perturbation_results.pkl
	rm -f criteo_subsampled_python_vb_map_results.pkl
	rm -f criteo_subsampled_python_vb_results.pkl
	rm -f criteo_subsampled_mcmc_draws.Rdata
	rm -f criteo_subsampled_stan_dat.json

criteo_subsampled_python_vb_perturbation_results.pkl:
	@echo "Running "$(PY_SCRIPT_LOC)/LogisticGLMMPerturbations.ipynb
	source $(VENV_LOC)
	jupyter nbconvert --ExecutePreprocessor.timeout=1200 \
	    --to notebook --execute \
		$(PY_SCRIPT_LOC)/LogisticGLMMPerturbations.ipynb

criteo_subsampled_python_vb_map_results.pkl:
	@echo "Running "$(PY_SCRIPT_LOC)/LogisticGLMM_MAP.ipynb
	source $(VENV_LOC)
	jupyter nbconvert --ExecutePreprocessor.timeout=1200 \
	    --to notebook --execute \
		$(PY_SCRIPT_LOC)/LogisticGLMM_MAP.ipynb

criteo_subsampled_python_vb_results.pkl: criteo_subsampled_stan_dat.json
	@echo "Running "$(PY_SCRIPT_LOC)/LogisticGLMM.ipynb
	source $(VENV_LOC)
	jupyter nbconvert --ExecutePreprocessor.timeout=1200 \
	    --to notebook --execute \
		$(PY_SCRIPT_LOC)/LogisticGLMM.ipynb \

criteo_subsampled_mcmc_draws.Rdata: criteo_subsampled_stan_dat.json
	@echo "Running "$(R_SCRIPT_LOC)/run_stan.R
	$(R_SCRIPT_LOC)/run_stan.R

criteo_subsampled_stan_dat.json: criteo/criteo_data_for_paper.Rdata
	@echo "Running "$(R_SCRIPT_LOC)/create_datasets.R
	$(R_SCRIPT_LOC)/create_datasets.R

criteo/criteo_data_for_paper.Rdata:
	$(MAKE) -C criteo criteo_data_for_paper.Rdata
