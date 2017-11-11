
# You'll have to download the raw criteo data here.
# http://criteolabs.wpengine.com/downloads/2014-conversion-logs-dataset/

GIT_REPO_LOC = $$(git rev-parse --show-toplevel)
DATA_LOC = $(GIT_REPO_LOC)/code/data
RAW_CRITEO_DATA = $(DATA_LOC)/

all: $(GIT_REPO_LOC)/foo
	echo Hello!
	echo $(git_repo_loc)

$(GIT_REPO_LOC)/foo:
	echo OKOK
	echo wot > $(GIT_REPO_LOC)/foo
