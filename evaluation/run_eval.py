"""Prepare weights and run native verl evaluation with a configurable sample count.

The validation loop repeats each prompt EVAL_N_SAMPLES times (default 16),
then reports native mean@k and best@k/mean. This runner does not implement generation, grading or best@k.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = int(os.environ.get("EVAL_N_SAMPLES", "16"))
MAX_PROMPT = int(os.environ.get("EVAL_MAX_PROMPT_LENGTH", "4096"))
MAX_RESPONSE = int(os.environ.get("EVAL_MAX_RESPONSE_LENGTH", "16384"))
TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "1.0"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("EVAL_GPU_MEMORY_UTILIZATION", "0.7"))
TENSOR_MODEL_PARALLEL_SIZE = int(os.environ.get("EVAL_TENSOR_MODEL_PARALLEL_SIZE", "1"))
TOKENIZER_PATH = os.environ.get("EVAL_TOKENIZER_PATH")
CACHE_ROOT = Path(os.environ.get("EVAL_CACHE_ROOT", str(ROOT / ".cache/eval_hf_models")))


def hf_weights_complete(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / index).is_file():
            weights = json.loads((path / index).read_text()).get("weight_map", {})
            return bool(weights) and all(
                (path / name).is_file() and (path / name).stat().st_size > 0
                for name in set(weights.values())
            )
    return any((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("model.safetensors", "pytorch_model.bin"))


def resolve_checkpoint(value: str) -> tuple[Path, bool]:
    """Accept an HF model, experiment root, global_step_N, actor, or actor/huggingface."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    if hf_weights_complete(path):
        return path, False
    marker = path / "latest_checkpointed_iteration.txt"
    if marker.is_file():
        step = marker.read_text().strip()
        if not step.isdigit():
            raise ValueError(f"invalid checkpoint marker: {marker}")
        path = path / f"global_step_{step}"
    if path.name == "huggingface" and (path.parent / "fsdp_config.json").is_file():
        path = path.parent
    if (path / "actor").is_dir():
        path = path / "actor"
    config_path = path / "fsdp_config.json"
    if not config_path.is_file():
        raise ValueError(f"no complete HF weights or FSDP actor found: {path}")
    world_size = json.loads(config_path.read_text()).get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError(f"invalid world_size: {config_path}")
    for rank in range(world_size):
        shard = path / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"incomplete checkpoint, missing model shard: {shard}")
    if not (path / "huggingface/config.json").is_file():
        raise ValueError(f"missing model config: {path / 'huggingface/config.json'}")
    return path, True


def checkpoint_fingerprint(actor: Path) -> str:
    files = [actor / "fsdp_config.json", *actor.glob("model_world_size_*_rank_*.pt")]
    files += [p for p in (actor / "huggingface").rglob("*") if p.is_file()]
    signature = [(str(p.relative_to(actor)), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(files)]
    return hashlib.sha256(json.dumps([str(actor), signature]).encode()).hexdigest()[:20]


def dataset_info(values: list[str]) -> tuple[list[Path], int]:
    import pyarrow.parquet as pq

    paths = [Path(value).expanduser().resolve() for value in values]
    if len(set(paths)) != len(paths):
        raise ValueError("TEST_DATASETS contains duplicate files, which would double-count scores.")
    count = 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"test set does not exist: {path}")
        parquet = pq.ParquetFile(path)
        missing = {"prompt", "data_source", "reward_model"} - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing verl fields: {sorted(missing)}")
        if parquet.metadata.num_rows == 0:
            raise ValueError(f"test set is empty: {path}")
        count += parquet.metadata.num_rows
    return paths, count


def gpu_selection(max_rows: int, dry_run: bool) -> list[str]:
    skip_memory_check = os.environ.get("EVAL_SKIP_GPU_MEMORY_CHECK", "false").lower() in {"1", "true", "yes"}
    query_fields = "index,uuid" if skip_memory_check else "index,uuid,memory.used"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    inventory = [tuple(v.strip() for v in row) for row in csv.reader(result.stdout.splitlines())]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = visible.split(",") if visible is not None else [row[0] for row in inventory]
    selected = []
    for token in requested:
        token = token.strip()
        matches = [row for row in inventory if row[0] == token or
                   (token.startswith("GPU-") and row[1].startswith(token))]
        if len(matches) != 1:
            raise ValueError(f"cannot resolve GPU in CUDA_VISIBLE_DEVICES: {token!r}")
        selected.append(matches[0])
    selected = selected[:min(8, max_rows)]
    if not selected or len({row[0] for row in selected}) != len(selected):
        raise ValueError("At least one GPU is required, and the GPU list must not contain duplicates.")
    if not dry_run and not skip_memory_check:
        busy = [f"GPU {row[0]}: {row[2]} MiB" for row in selected if float(row[2]) > 2048]
        shared = os.environ.get("EVAL_ALLOW_SHARED_GPUS", "false").lower() in {"1", "true", "yes"}
        if busy and not shared:
            raise RuntimeError("Selected GPUs already have significant memory in use; "
                               "switch to an idle node or set CUDA_VISIBLE_DEVICES. "
                               "This script does not stop training.\n" + "\n".join(busy))
        if shared:
            memory = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
                check=True, text=True, capture_output=True,
            )
            available = {row[0].strip(): (float(row[1]), float(row[2]))
                         for row in csv.reader(memory.stdout.splitlines())}
            for row in selected:
                total, free = available[row[0]]
                required = total * GPU_MEMORY_UTILIZATION + 4096
                if free < required:
                    raise RuntimeError(f"GPU {row[0]} has only {free:.0f} MiB free, below the shared-eval precheck budget of {required:.0f} MiB")
            print(f"Shared-GPU eval: vLLM GPU memory fraction={GPU_MEMORY_UTILIZATION}, free-memory precheck passed.", flush=True)
    elif skip_memory_check:
        print("EVAL_SKIP_GPU_MEMORY_CHECK=true is set: skipping GPU memory usage and free-memory checks.", flush=True)
    return [row[0] for row in selected]


def merge_command(actor: Path, target: Path) -> list[str]:
    return [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
            "--local_dir", str(actor), "--target_dir", str(target), "--use_cpu_initialization"]


def merge_model(actor: Path, target: Path, env: dict[str, str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # The lock and completion marker prevent reuse of an incomplete/concurrent export.
    with (target.parent / f"{target.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / ".avg16_export_complete"
        if marker.is_file() and hf_weights_complete(target):
            return
        if target.exists():
            raise RuntimeError(f"merge directory exists but is incomplete; inspect it first: {target}")
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            cpu_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
            subprocess.run(merge_command(actor, staging), cwd=ROOT, env=cpu_env, check=True)
            if not hf_weights_complete(staging):
                raise RuntimeError("verl merger did not produce a complete HF model.")
            (staging / ".avg16_export_complete").write_text(str(actor) + "\n")
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def build_eval_command(model: Path, datasets: list[Path], project: str, experiment: str,
                       n_gpus: int, run_dir: Path) -> list[str]:
    # JSON quoting is for Hydra's override grammar; no shell evaluates these arguments.
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    overrides = [
        "algorithm.adv_estimator=grpo", "algorithm.use_kl_in_reward=False",
        "critic.enable=False", "reward_model.enable=False", "reward_model.reward_manager=naive",
        f"data.train_files={json.dumps([str(p) for p in datasets])}",
        f"data.val_files={json.dumps([str(p) for p in datasets])}",
        f"data.train_batch_size={n_gpus}", f"data.train_max_samples={n_gpus}",
        f"data.val_batch_size={int(os.environ.get('EVAL_VAL_BATCH_SIZE', '8'))}",
        "data.shuffle=False", "data.validation_shuffle=False", "data.seed=42",
        "data.dataloader_num_workers=0", "data.filter_overlong_prompts=False", "data.truncation=error",
        f"data.max_prompt_length={MAX_PROMPT}", f"data.max_response_length={MAX_RESPONSE}",
        "data.return_raw_chat=True",
        f"+data.apply_chat_template_kwargs.enable_thinking={os.environ.get('EVAL_ENABLE_THINKING', 'false').lower() == 'true'}",
        f"actor_rollout_ref.model.path={quote(model)}", "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={n_gpus}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.rollout.name=vllm", "actor_rollout_ref.rollout.mode=sync",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={TENSOR_MODEL_PARALLEL_SIZE}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={GPU_MEMORY_UTILIZATION}",
        f"actor_rollout_ref.rollout.max_model_len={MAX_PROMPT + MAX_RESPONSE}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max(16384, MAX_PROMPT + MAX_RESPONSE)}",
        "actor_rollout_ref.rollout.n=1", "actor_rollout_ref.rollout.calculate_log_probs=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.rollout.val_kwargs.n={N_SAMPLES}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={TEMPERATURE}",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0", "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        f"+actor_rollout_ref.rollout.val_kwargs.max_tokens={MAX_RESPONSE}",
        f"custom_reward_function.path={quote(ROOT / 'verl/verl/utils/reward_score/__init__.py')}",
        "custom_reward_function.name=default_compute_score",
        "trainer.val_only=True", "trainer.val_before_train=True", "trainer.resume_mode=disable",
        "trainer.total_epochs=1", "trainer.save_freq=-1", "trainer.test_freq=-1", "trainer.is_plot=False",
        "trainer.logger=[console,wandb]", "trainer.log_val_generations=2",
        f"trainer.project_name={quote(project)}", f"trainer.experiment_name={quote(experiment)}",
        f"trainer.n_gpus_per_node={n_gpus}", "trainer.nnodes=1",
        f"trainer.validation_data_dir={quote(run_dir / 'generations')}",
        f"trainer.default_local_dir={quote(run_dir / 'unused_checkpoints')}",
        f"hydra.run.dir={quote(run_dir / 'hydra')}",
        "+ray_kwargs.ray_init.address=local", "+ray_kwargs.ray_init.include_dashboard=False",
        "+ray_kwargs.ray_init.object_store_memory=2147483648",
        f"ray_kwargs.ray_init.num_cpus={max(n_gpus + 2, min(32, os.cpu_count() or 8))}",
    ]
    if os.environ.get("EVAL_USE_TORCH_COMPILE", "true").lower() in {"0", "false", "no"}:
        overrides.append("actor_rollout_ref.actor.use_torch_compile=False")
    if TOKENIZER_PATH:
        overrides.append(f"actor_rollout_ref.model.tokenizer_path={quote(TOKENIZER_PATH)}")
    if os.environ.get("EVAL_MAX_NUM_SEQS"):
        overrides.append(f"actor_rollout_ref.rollout.max_num_seqs={int(os.environ['EVAL_MAX_NUM_SEQS'])}")
    return [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.project.strip() or not args.experiment.strip():
        raise ValueError("W&B project and experiment name must not be empty.")
    if N_SAMPLES < 1 or MAX_PROMPT < 1 or MAX_RESPONSE < 1 or not 0 < GPU_MEMORY_UTILIZATION < 1 or not TEMPERATURE > 0:
        raise ValueError("Sample counts and lengths must be positive, temperature must be greater than 0, and the GPU memory fraction must be between 0 and 1.")
    source, needs_merge = resolve_checkpoint(args.ckpt)
    datasets, row_count = dataset_info(args.datasets)
    gpus = gpu_selection(row_count, args.dry_run)
    if TENSOR_MODEL_PARALLEL_SIZE < 1 or TENSOR_MODEL_PARALLEL_SIZE > len(gpus):
        raise ValueError(
            f"EVAL_TENSOR_MODEL_PARALLEL_SIZE={TENSOR_MODEL_PARALLEL_SIZE} is invalid; "
            f"{len(gpus)} GPUs are currently visible."
        )
    if len(gpus) % TENSOR_MODEL_PARALLEL_SIZE != 0:
        raise ValueError(
            f"The GPU count {len(gpus)} must be divisible by TP={TENSOR_MODEL_PARALLEL_SIZE}."
        )
    model = CACHE_ROOT / checkpoint_fingerprint(source) if needs_merge else source
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.experiment)
    # Keep the large per-sample JSONL files on the shared volume.  The shell
    # wrapper still controls the log destination independently through
    # EVAL_LOG_FILE, so this does not move or rename evaluation logs.
    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(ROOT / "validation_log")))
    run_dir = output_root / f"avg{N_SAMPLES}" / f"{slug}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    command = build_eval_command(model, datasets, args.project, args.experiment, len(gpus), run_dir)
    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass, validate_config

    with initialize_config_dir(config_dir=str(ROOT / "verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    validate_config(config, use_reference_policy=False, use_critic=False)
    omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    plan = {
        "checkpoint": str(source), "hf_model": str(model), "merge_required": needs_merge,
        "tokenizer": TOKENIZER_PATH or str(model),
        "datasets": [str(p) for p in datasets], "questions": row_count, "samples_per_question": N_SAMPLES,
        "temperature": TEMPERATURE, "top_p": 1.0, "gpus": gpus,
        "max_prompt_length": MAX_PROMPT, "max_response_length": MAX_RESPONSE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "tensor_model_parallel_size": TENSOR_MODEL_PARALLEL_SIZE,
        "validation_batch_size": int(os.environ.get("EVAL_VAL_BATCH_SIZE", "8")),
        "max_num_seqs": config.actor_rollout_ref.rollout.max_num_seqs,
        "enforce_eager": config.actor_rollout_ref.rollout.enforce_eager,
        "project": args.project, "experiment": args.experiment, "output_dir": str(run_dir),
        "metrics": [f"val-core/<data_source>/<acc-or-reward>/mean@{N_SAMPLES}",
                    f"val-core/<data_source>/<acc-or-reward>/best@{N_SAMPLES}/mean"],
        "merge_command": merge_command(source, model) if needs_merge else None,
        "eval_command": command,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN: no weights were merged and no Ray cluster or GPU evaluation was started.")
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus), "RAY_ADDRESS": "local"}
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "eval_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    if needs_merge:
        merge_model(source, model, env)
    print("Running native verl val-only:\n" + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"Evaluation finished. avg{N_SAMPLES} = mean@{N_SAMPLES}; best@{N_SAMPLES} uses verl's native best@{N_SAMPLES}/mean.", flush=True)
    print(f"Per-sample answers and scores: {run_dir / 'generations/0.jsonl'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
