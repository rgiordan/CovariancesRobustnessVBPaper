
SHELL = /bin/bash

GIT_REPO_LOC := $(shell git rev-parse --show-toplevel)
DATA_LOC = $(GIT_REPO_LOC)/code/criteo_experiment/data
CRITEO_DATA_LOC = $(DATA_LOC)/criteo

# This is the dataset subsampled from the Criteo logs.
CRITEO_RDATA = $(CRITEO_DATA_LOC)/criteo_data_for_paper.Rdata

all: $(CRITEO_RDATA)
	echo Ready to go!
	$(MAKE) -C $(DATA_LOC) all

# Post-process the Criteo data
$(CRITEO_RDATA):
	$(MAKE) -C $(CRITEO_DATA_LOC) all
