#!/bin/bash
#SBATCH --nodes=4
#SBATCH -J read-tfrecord
#SBATCH -o logs/%x-%j.out
#SBATCH -D .


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

declare -a CMD=(
    python
    /home/ubuntu/awsome-distributed-training/3.test_cases/tensorflow/cosmoflow/utils/read_tfrecord.py
    /fsx/data/cosmoUniverse_2019_05_4parE_tf_v2_mini/train/21688988_univ_ics_2019-03_a10177483_041.tfrecord
)

srun -l "${ARGS[@]}"  --mpi=pmix  "${CMD[@]}"
