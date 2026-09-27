# Evaluation: Avg@N

The single top-level entry point is `eval.sh`; the implementation is `evaluation/run_eval.py`. Run it from the repository root in the installed verl environment. No top-level `scripts/` helper is required.

## Data and a minimal run

Download all prepared files from [LemonTea03/BinaryOPD-Data](https://huggingface.co/datasets/LemonTea03/BinaryOPD-Data). Use explicit paths because the shell's legacy defaults do not match the flat Hub release layout.

```bash
export DATA_ROOT="$PWD/data"
hf download LemonTea03/BinaryOPD-Data --repo-type dataset --local-dir "$DATA_ROOT"

CKPT_PATH=/path/to/model_or/global_step_N \
EVAL_DATASETS_COLON="$DATA_ROOT/eval/math_eval.parquet:$DATA_ROOT/eval/livecodebench_v6.parquet:$DATA_ROOT/eval/HumanEval.parquet:$DATA_ROOT/eval/MBPP.parquet" \
EXPERIMENT_NAME=my-model-avg8 \
EVAL_N_SAMPLES=8 EVAL_TEMPERATURE=0.7 \
EVAL_MAX_RESPONSE_LENGTH=16384 EVAL_VAL_BATCH_SIZE=8 \
bash eval.sh
```

For math only, set `EVAL_DATASETS_COLON="$DATA_ROOT/eval/math_eval.parquet"`. For code only, omit the math file from the colon-separated list. `EVAL_DATASET` can select one file if `EVAL_DATASETS_COLON` is unset; the latter takes precedence.

To preview, append `--dry-run`. This checks inputs and configuration and prints the evaluation plan without merging weights, starting Ray, or running model inference. It still enumerates GPU IDs through `nvidia-smi` and the shell creates its log directory.

## Actual defaults

These are the defaults of **`eval.sh`**. Directly invoking `run_eval.py` has different fallback defaults for sample count (16), temperature (1.0), validation batch size (8), and the GPU-memory check; use the shell entry point for the settings below.

| Environment variable | Shell default | Meaning |
| --- | --- | --- |
| `EVAL_N_SAMPLES` | `8` | Samples per problem; 8 gives Avg@8. |
| `EVAL_MAX_PROMPT_LENGTH` | `4096` | Maximum prompt tokens; overlong inputs raise an error. |
| `EVAL_MAX_RESPONSE_LENGTH` | `16384` | Maximum generated tokens. |
| `EVAL_TEMPERATURE` | `0.7` | Sampling temperature. |
| `EVAL_ENABLE_THINKING` | `false` | Passed to the tokenizer chat template, matching the training launchers. |
| `EVAL_TENSOR_MODEL_PARALLEL_SIZE` | `1` | vLLM TP size; must divide the selected GPU count. |
| `EVAL_GPU_MEMORY_UTILIZATION` | `0.7` | vLLM GPU-memory fraction. |
| `EVAL_VAL_BATCH_SIZE` | `512` | Distinct problems per validation batch, before repeated sampling. |
| `EVAL_MAX_NUM_SEQS` | Unset | Optional vLLM concurrency limit. |
| `EVAL_TOKENIZER_PATH` | Unset | Optional local tokenizer path instead of the model's tokenizer. |
| `EVAL_USE_TORCH_COMPILE` | `true` | Whether actor torch.compile is enabled. |
| `EVAL_SKIP_GPU_MEMORY_CHECK` | `1` | Skip GPU occupancy/free-memory checks. |
| `PYTHON_BIN` | `python3` | Interpreter from the active environment. |

`CKPT_PATH`, `WANDB_PROJECT`, and `EXPERIMENT_NAME` select the model, W&B project, and run name. A smaller `EVAL_VAL_BATCH_SIZE` can reduce transient batching pressure, but does not proportionally reduce vLLM's reserved memory or change Avg@N. For example, batch size 8 and N=8 produce 64 sampled requests per complete validation batch before distribution across workers.


## Checkpoint handling

Accepted inputs include a complete Hugging Face model directory, an FSDP `global_step_N`, its `actor` directory, `actor/huggingface`, or an experiment directory containing `latest_checkpointed_iteration.txt`.

FSDP shards are merged using the native `verl.model_merger` with CPU initialization. The result is cached in `.cache/eval_hf_models/` (`EVAL_CACHE_ROOT` override), without modifying the source checkpoint. A directory containing only config/tokenizer metadata is not treated as a complete HF model. Large checkpoints require sufficient CPU RAM and disk space for merging.

Evaluation invokes native verl validation with `trainer.val_only=True`; it does not train or require a teacher model. Code benchmarks execute generated programs through the bundled judges: use an appropriately isolated evaluation environment rather than treating test execution as a security sandbox.


## Output locations

- Shell logs: `logs/eval_<name>_<timestamp>_<pid>.log` (`EVAL_LOG_FILE` override).
- Evaluation plan and generations: `validation_log/avgN/<name>_<timestamp>_<pid>/` (`EVAL_OUTPUT_ROOT` override). The plan is `eval_plan.json`; generations are under `generations/`.
- W&B local files: under `wandb/` (`WANDB_DIR` override); W&B may add a nested `wandb/` directory.


