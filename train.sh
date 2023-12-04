DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 41971"

DEEPSPEED_CONFIG="/root/autodl-tmp/workspace/openreviewer/config/zero2.json"
SEED=42

MODEL_PATH="/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
MODEL_TYPE="vicuna"

DATA_PATH="/root/autodl-tmp/data/iclr2024/1204.jsonl"
DATA_NAME="1204"

SAVE_PATH="/root/autodl-tmp/ckeckpoints/1204"

BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32

OPTS=""

OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS}"

OPTS+=" --max-length 6144"
OPTS+=" --max-prompt-length 4096"

OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-type openreview"
OPTS+=" --save-path ${SAVE_PATH}"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DEEPSPEED_CONFIG}"

torchrun ${DISTRIBUTED_ARGS} /root/autodl-tmp/workspace/openreviewer/main.py ${OPTS}