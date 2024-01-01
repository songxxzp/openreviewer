DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 41971"

DEEPSPEED_CONFIG="/root/autodl-tmp/workspace/openreviewer/config/zero2.json"
SEED=42

MODEL_PATH="/root/autodl-tmp/model/vicuna-7b-v1.5-16k"
MODEL_TYPE="vicuna"

DATA_PATH="/root/autodl-tmp/data/iclr2024/processed-0101-merge-2048-matched-cleaned-train.jsonl"
DATA_NAME="0101"

SAVE_PATH="/root/autodl-tmp/checkpoints/0101-full"

BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=32

OPTS=""

OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS}"
OPTS+=" --warmup-steps 2"

OPTS+=" --max-length 6144"
OPTS+=" --max-prompt-length 6144"

OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-type ReviewerAgent"
OPTS+=" --save-path ${SAVE_PATH}"
OPTS+=" --dataset-type MultiTurnDataset"
# OPTS+=" --max-samples 128"

OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DEEPSPEED_CONFIG}"

torchrun ${DISTRIBUTED_ARGS} /root/autodl-tmp/workspace/openreviewer/main.py ${OPTS}