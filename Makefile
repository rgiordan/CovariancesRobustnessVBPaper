
SHELL = /bin/bash

GIT_REPO_LOC := $(shell git rev-parse --show-toplevel)
DATA_LOC = $(GIT_REPO_LOC)/code/data
CRITEO_DATA_LOC = $(DATA_LOC)/criteo

# This is the dataset subsampled from the Criteo logs.
CRITEO_RDATA = $(CRITEO_DATA_LOC)/criteo_data_for_paper.Rdata

all: $(CRITEO_RDATA)
	echo Ready to go!
	ls -l $(CRITEO_RDATA)

$(CRITEO_RDATA):
	$(MAKE) -C $(CRITEO_DATA_LOC) all
