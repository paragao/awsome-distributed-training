#!/bin/bash
#SBATCH --nodes=4
#SBATCH -o logs/%x-%j.out
#SBATCH -J train-cori
##SBATCH -C knl
##SBATCH -q debug
##SBATCH -t 30


: "${IMAGE:=/home/ubuntu/awsome-distributed-training/3.test_cases/tensorflow/cosmoflow/cosmoflow.sqsh}"

set -euxo pipefail
# export OMP_NUM_THREADS=32
# export KMP_BLOCKTIME=1
# export KMP_AFFINITY="granularity=fine,compact,1,0"
# export HDF5_USE_FILE_LOCKING=FALSE

export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
#export FI_LOG_LEVEL=warn

## NCCL Environment variables
export NCCL_DEBUG=INFO

# variables for Enroot
declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts /fsx,/home
)

srun -l "${ARGS[@]}" python /home/ubuntu/awsome-distributed-training/3.test_cases/tensorflow/cosmoflow/train.py -d $@
