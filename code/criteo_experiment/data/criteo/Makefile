# Make the dataset for the paper from the raw Criteo data.

SHELL = /bin/bash

criteo_data_for_paper.Rdata: criteo_data_clean.Rdata
	./subsample_criteo_data.R

criteo_data_clean.Rdata: data.Rdata
	./clean_criteo_data.R

data.Rdata: data.txt
	./criteo_data_text_to_rdata.R

data.txt: criteo_conversion_logs.tar.gz
	tar -xzvf criteo_conversion_logs.tar.gz
	# Update the date on data.txt so make doesn't re-run every time.
	touch data.txt

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
