
SHELL = /bin/bash

# You'll have to download the raw criteo data here.
# http://criteolabs.wpengine.com/downloads/2014-conversion-logs-dataset/

GIT_REPO_LOC := $(shell git rev-parse --show-toplevel)
DATA_LOC = $(GIT_REPO_LOC)/code/data
CRITEO_DATA_LOC = $(DATA_LOC)/criteo
RAW_CRITEO_TSV = $(CRITEO_DATA_LOC)/data.txt

all: $(RAW_CRITEO_TSV)
	echo Hello!
	head $(RAW_CRITEO_TSV)

$(RAW_CRITEO_TSV):
	$(MAKE) -C $(CRITEO_DATA_LOC) all
