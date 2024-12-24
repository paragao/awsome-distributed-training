ENROOT_IMAGE=cosmoflow

all: build clean import

build:
	docker build -t ${ENROOT_IMAGE} -f cosmoflow.Dockerfile .

clean:
	-rm ${ENROOT_IMAGE}.sqsh

import:
	enroot import -o ${ENROOT_IMAGE}.sqsh dockerd://cosmoflow:latest
