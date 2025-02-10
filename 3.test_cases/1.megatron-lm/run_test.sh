#!/bin/bash
#SBATCH --job-name=128
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --exclusive
#SBATCH --output=slurm/dc-%x_%j.out
#SBATCH --error=slurm/dc-%x_%j.err

# default variables for Enroot
: "${IMAGE:=$(pwd)/megatron-training.sqsh}"
: "${DATA_PATH:=/fsx}"
: "${DATA_MOUNT:=$(pwd)/gpt2:$DATA_PATH,/fsx/langjian/Megatron-LM:/workspace/Megatron-LM}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=$SLURM_JOB_NUM_NODES

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$head_node
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{for (i=1; i<=NF; i++) if ($i ~ /^192\\.168\\./) print $i}')
export TRITON_CACHE_DIR="/tmp/triton-cache"

echo $head_node
echo $head_node_ip
echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
#export FI_LOG_LEVEL=warn

## NCCL Environment variables
export NCCL_DEBUG=INFO

### Increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf
export NCCL_BUFFSIZE=8388608
### Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
export NCCL_P2P_NET_CHUNKSIZE=524288

### Improve performance for AllReduce by selecting specific protocol and algorithm for specific
### message size and number of ranks.
### More information https://github.com/aws/aws-ofi-nccl/wiki/Algorithm-and-Protocol-Tuner-for-AWS.
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so

#ulimit -n 50000

CHECKPOINT_PATH=/workspace/checkpoints #/mnt/guowei.he/checkpoints/dc_test
WORK_DIR=/workspace #/mnt/guowei.he/workspace/benchmark/code/fluidstack/Megatron-LM-NVLM-1.0
DATA_CACHE_DIR=${WORK_DIR}/data_cache/448
TOKENIZER_TYPE=HuggingFaceTokenizer
TOKENIZER_MODEL=/mnt/guowei.he/data/SlimPajama-627B_250k_tokenized
EVAL_DATA_PATH=/mnt/guowei.he/data/SlimPajama-627B_250k_tokenized
TRAIN_DATA="\
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk1
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk2
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk3
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk4
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk5
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk6
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk7
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk8
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk9
1.0 /mnt/guowei.he/data/SlimPajama-627B_250k_tokenized/slimpajama-train-chunk10
"

VALID_DATA="\
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk1 
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk2  
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk3  
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk4  
1.0 ${EVAL_DATA_PATH}/slimpajama-validation-chunk5" 

TEST_DATA="\
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk1 
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk2  
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk3  
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk4  
1.0 ${EVAL_DATA_PATH}/slimpajama-test-chunk5" 

VOCAB_FILE=$DATA_PATH/gpt2-vocab.json
MERGE_FILE=$DATA_PATH/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint $head_node_ip:29500
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 8192
    --max-position-embeddings 32768
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --init-method-std 0.01  # 1/sqrt(8192)
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 500000 # change with seq len
    --swiglu
    --untie-embeddings-and-output-weights
    # --no-masked-softmax-fusion # no use with flash-attn
    --no-position-embedding
    --apply-layernorm-1p
)


DATA_ARGS=(
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_MODEL}
    --train-data-path ${TRAIN_DATA}
    --valid-data-path ${VALID_DATA}
    --test-data-path ${TEST_DATA}
    --data-cache-path ${DATA_CACHE_DIR}
)

DATA_ARGS=(
    --data-path $DATA_PATH/my-gpt2_text_document 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

TRAINING_ARGS=(
    --micro-batch-size 1 # 2 # 3584GPUs 
    --global-batch-size 128  # 1792 
    --seed 42
    --lr 2e-4
    --adam-beta1 0.9
    --adam-beta2 0.95
    --train-iters 1000000 # 16T tokens /( 1792 bs * 8192 seq_len )
    --lr-decay-iters 950000 # train-iters * 0.9
    --lr-decay-style cosine
    --min-lr 2.0e-5
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --overlap-grad-reduce
    --overlap-param-gather
    # --recompute-activations
    --recompute-granularity 'full'
    --recompute-method 'uniform'
    --recompute-num-layers 1
    --attention-backend flash
)


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --distributed-timeout-minutes 60
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 250
    --log-throughput
    --log-memory-to-tensorboard
    --log-params-norm
    --log-timers-to-tensorboard
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

CHECKPOINT_ARGS=(
    --eval-interval 2500
    --eval-iters 5
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --use-dist-ckpt # need update implementation
    --async-save
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-entity ${WANDB_ENTITY:-"mbzuai-llm"}
    	--wandb-project ${WANDB_PROJECT:-"V1"}
        --wandb-exp-name ${WANDB_NAME:-"64-dc"}
    )
fi


srun --container-mounts=$DATA_MOUNT \
    --container-image=$IMAGE \
    torchrun ${DISTRIBUTED_ARGS[@]} $WORK_DIR/Megatron-LM/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]} \
    ${LOGGING_ARGS[@]}

