DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 41971"

DEEPSPEED_CONFIG="/root/autodl-tmp/workspace/openreviewer/config/zero2.json"
SEED=42

MODEL_PATH="/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
MODEL_TYPE="vicuna"

DATA_PATH="/root/autodl-tmp/workspace/openreviewer/data/dummy.jsonl"
DATA_NAME="test"

SAVE_PATH="/root/autodl-tmp/ckeckpoints/test"

OPTS=""

OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --save-path ${SAVE_PATH}"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DEEPSPEED_CONFIG}"

torchrun ${DISTRIBUTED_ARGS} /root/autodl-tmp/workspace/openreviewer/main.py ${OPTS}