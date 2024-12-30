IMAGE=cosmoflow

all: build clean import

build:
	docker build -t ${IMAGE} -f cosmoflow.Dockerfile .

clean:
	-rm /fsx/images/${IMAGE}.sqsh

import:
	enroot import -o /fsx/images/${IMAGE}.sqsh dockerd://${IMAGE}:latest