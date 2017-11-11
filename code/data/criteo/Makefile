
SHELL = /bin/bash

all: data.tsv
	echo Hello!
	head $(RAW_CRITEO_DATA)

data.tsv: criteo_conversion_logs.tar.gz
	tar -xzvf criteo_conversion_logs.tar.gz

criteo_conversion_logs.tar.gz:
	@echo -e "\n\n\n-----------------"
	@echo "You'll have to agree to the terms of use and ";
	@echo "and download the raw criteo data from this website:"
	@echo "http://criteolabs.wpengine.com/downloads/2014-conversion-logs-dataset/"
	@echo -e "\nSave the results to"
	@echo $(shell pwd)"/criteo_conversion_logs.tar.gz"
	@echo "and re-run make."
	@echo -e "-----------------\n\n\n"
	exit 1
