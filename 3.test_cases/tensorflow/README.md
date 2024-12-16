## 1. Preparation

This guide assumes that you have the following:

* A functional Slurm cluster on AWS.
* Docker, [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) installed.
* An FSx for Lustre filesystem mounted on `/fsx`.

We recommend that you setup a Slurm cluster using the templates in the architectures [directory](../../1.architectures). Before creating the Slurm cluster, you need to setup the following environment variables:

```bash
export APPS_PATH=/apps
export ENROOT_IMAGE=$APPS_PATH/tensorflow.sqsh
export FSX_PATH=/fsx
export DATA_PATH=$FSX_PATH/mnist
export TEST_CASE_PATH=${HOME}/7.tensorflow-distributed  # where you copy the test case or set to your test case path
cd $TEST_CASE_PATH
```

then follow the detailed instructions [here](../../1.architectures/2.aws-parallelcluster/README.md).

## 2. Build the container

Before running training jobs, you need to use an [Enroot](https://github.com/NVIDIA/enroot) container to retrieve and preprocess the input data. Below are the steps you need to follow:

1. Copy the test case files to your cluster. You will need `0.tensorflow.Dockerfile`,
2. Build the Docker image with the command below in this directory.

   ```bash
   docker build -t tensorflow -f 0.tensorflow.Dockerfile .
   ```