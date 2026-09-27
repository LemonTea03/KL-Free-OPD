#!/usr/bin/env bash
# Standalone OPD-baseline training launcher. Activate your verl environment first.
# --dry-run prints the command without loading models or inspecting GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/models}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$MODEL_ROOT/Qwen3-4B}"
MATH_TEACHER_PATH="${MATH_TEACHER_PATH:-$MODEL_ROOT/Qwen3-4B-Non-Thinking-RL-Math-Step500}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$MATH_TEACHER_PATH}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATA_ROOT/math_and_code/train.parquet}"
TEST_DATASETS=("$DATA_ROOT/math_eval.parquet" "$DATA_ROOT/livecodebench_v6/test.parquet"
               "$DATA_ROOT/humaneval.parquet" "$DATA_ROOT/mbpp.parquet")
if [[ -n "${EVAL_DATASETS_COLON:-}" ]]; then
    IFS=: read -r -a TEST_DATASETS <<< "$EVAL_DATASETS_COLON"
fi
VAL_FILES="["
for dataset in "${TEST_DATASETS[@]}"; do VAL_FILES+="\"$dataset\","; done
VAL_FILES="${VAL_FILES%,}]"

PROJECT_NAME="${PROJECT_NAME:-BinaryOPD}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-8192}"
MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + (MAX_RESP_LENGTH > MAX_VAL_RESP_LENGTH ? MAX_RESP_LENGTH : MAX_VAL_RESP_LENGTH) ))
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-256}"
PARALLEL_SIZE="${PARALLEL_SIZE:-1}"
N_GPUS="${N_GPUS:-8}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}"
REWARD_MAX_TOKEN_LEN_PER_GPU="${REWARD_MAX_TOKEN_LEN_PER_GPU:-24576}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"

# Experiment-specific settings.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-OPD-baseline}"
MODEL_PATHS=("$ACTOR_MODEL_PATH" "$TEACHER_MODEL_PATH")
FINAL_CKPT_DIR="${FINAL_CKPT_DIR:-${DEFAULT_CKPT_ROOT:-$ROOT/checkpoints}/$PROJECT_NAME/$EXPERIMENT_NAME}"
VALIDATION_DIR="${VALIDATION_DIR:-$ROOT/validation_log/$EXPERIMENT_NAME}"

CMD=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    reward_model.reward_manager=naive \
    "data.train_files=\"$TRAIN_DATASET\"" \
    "data.val_files=$VAL_FILES" \
    "data.train_batch_size=$((MINI_BATCH_SIZE * PARALLEL_SIZE))" \
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" \
    "data.max_response_length=$MAX_RESP_LENGTH" \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "actor_rollout_ref.model.path=\"$ACTOR_MODEL_PATH\"" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE:-1e-6}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    "actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    "actor_rollout_ref.rollout.dtype=$MODEL_DTYPE" \
    actor_rollout_ref.rollout.temperature=1.0 \
    +actor_rollout_ref.rollout.teacher_temperature=1.0 \
    +actor_rollout_ref.rollout.log_prob_top_k=0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.7}" \
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    "+actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH" \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    reward_model.enable=True \
    +reward_model.reward_kwargs.enable_format_reward=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    "+reward_model.model.dtype=$MODEL_DTYPE" \
    reward_model.micro_batch_size_per_gpu=8 \
    reward_model.use_dynamic_bsz=True \
    "reward_model.forward_max_token_len_per_gpu=$REWARD_MAX_TOKEN_LEN_PER_GPU" \
    "custom_reward_function.path=\"$ROOT/verl/verl/utils/reward_score/__init__.py\"" \
    custom_reward_function.name=default_compute_score \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    'trainer.logger=[console,wandb]' \
    "trainer.project_name=\"$PROJECT_NAME\"" \
    "trainer.experiment_name=\"$EXPERIMENT_NAME\"" \
    "trainer.validation_data_dir=\"$VALIDATION_DIR\"" \
    "trainer.default_local_dir=\"$FINAL_CKPT_DIR\"" \
    "trainer.n_gpus_per_node=$N_GPUS" \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    "trainer.save_freq=${SAVE_FREQ:-55}" \
    "trainer.test_freq=${TEST_FREQ:-220}" \
    "trainer.total_epochs=${TOTAL_EPOCHS:-3}" \
    trainer.is_plot=False \
    '+ray_kwargs.ray_init.address=local' \
    '+ray_kwargs.ray_init.include_dashboard=False' \
    actor_rollout_ref.rollout.binary_opd=False \
    "reward_model.model.path=\"$TEACHER_MODEL_PATH\"" \
    reward_model.multi_teacher_mode=routing \
    "$@"
)

if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
for path in "$TRAIN_DATASET" "${TEST_DATASETS[@]}" "${MODEL_PATHS[@]}"; do
    [[ -e "$path" ]] || { echo "Missing path: $path (configure MODEL_ROOT, DATA_ROOT or individual paths)" >&2; exit 1; }
done
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/wandb/$EXPERIMENT_NAME}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-$ROOT/.cache/outlines/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR" "$WANDB_DIR" "$FINAL_CKPT_DIR" "$OUTLINES_CACHE_DIR"
exec > >(tee -a "$LOG_DIR/$EXPERIMENT_NAME.log") 2>&1
echo "Experiment: $EXPERIMENT_NAME; checkpoint directory: $FINAL_CKPT_DIR"
exec "${CMD[@]}"
