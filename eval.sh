#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Model and data. TEST_DATASETS can hold one or more verl-format parquet files.
CKPT_PATH="${CKPT_PATH:-$SCRIPT_DIR/models/Qwen3-4B}"
WANDB_PROJECT="${WANDB_PROJECT:-BinaryOPD-Eval}"
TEST_DATASETS=(
    "$DATA_ROOT/mbpp.parquet"
    "$DATA_ROOT/humaneval.parquet"
    "$DATA_ROOT/livecodebench_v6/test.parquet"
    "$DATA_ROOT/math_eval.parquet"
)

# Common eval parameters: edit the defaults directly or override via same-name environment variables.
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-8}"                        # samples per question; 8 means avg8
export EVAL_MAX_PROMPT_LENGTH="${EVAL_MAX_PROMPT_LENGTH:-4096}"      # max input tokens
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-16384}" # max output tokens
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"                   # sampling temperature
export EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"         # +data.apply_chat_template_kwargs.enable_thinking=False
export EVAL_TENSOR_MODEL_PARALLEL_SIZE="${EVAL_TENSOR_MODEL_PARALLEL_SIZE:-1}" # vLLM TP
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.7}"     # vLLM GPU memory fraction
export EVAL_VAL_BATCH_SIZE="${EVAL_VAL_BATCH_SIZE:-512}"               # unique questions per batch (excluding repeated samples)
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-}"                     # vLLM max concurrency; empty falls back to verl config

# Optional. If CUDA_VISIBLE_DEVICES is unset, use all currently visible GPUs, up to 8.
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-}"                # empty uses the model's own tokenizer
export EVAL_USE_TORCH_COMPILE="${EVAL_USE_TORCH_COMPILE:-true}"
export EVAL_SKIP_GPU_MEMORY_CHECK="${EVAL_SKIP_GPU_MEMORY_CHECK:-1}" # skip the GPU-idle check by default
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-4B-avg${EVAL_N_SAMPLES}-len${EVAL_MAX_RESPONSE_LENGTH}}"

if [[ -n "${EVAL_DATASETS_COLON:-}" ]]; then
    IFS=':' read -r -a TEST_DATASETS <<< "$EVAL_DATASETS_COLON"
elif [[ -n "${EVAL_DATASET:-}" ]]; then
    TEST_DATASETS=("$EVAL_DATASET")
fi

cd "$SCRIPT_DIR"
if [[ -z "$CKPT_PATH" || -z "$WANDB_PROJECT" || -z "$EXPERIMENT_NAME" || -z "${TEST_DATASETS[0]:-}" ]]; then
    echo "Please fill in CKPT_PATH, WANDB_PROJECT, EXPERIMENT_NAME and TEST_DATASETS at the top of the script." >&2
    exit 2
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/wandb}"

mkdir -p "$SCRIPT_DIR/logs" "$WANDB_DIR"
LOG_NAME="${EXPERIMENT_NAME//[^a-zA-Z0-9_.-]/_}"
LOG_FILE="${EVAL_LOG_FILE:-$SCRIPT_DIR/logs/eval_${LOG_NAME}_$(date +%Y%m%d_%H%M%S)_$$.log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Eval log: $LOG_FILE"


"$PYTHON_BIN" -u "$SCRIPT_DIR/evaluation/run_eval.py" \
    --ckpt "$CKPT_PATH" \
    --project "$WANDB_PROJECT" \
    --experiment "$EXPERIMENT_NAME" \
    --datasets "${TEST_DATASETS[@]}" \
    "$@"
