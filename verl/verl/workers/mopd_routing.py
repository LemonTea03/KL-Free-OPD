# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ability routing inspired by One-Shot-OPD, retaining OPD's scoring pipeline."""

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from verl import DataProto


def route_by_ability(abilities, n_samples, ability_to_teacher, default_teacher, strict_routing=False):
    """Assign exactly one teacher to each row, preserving the input order."""
    if not ability_to_teacher:
        return [default_teacher] * n_samples
    if abilities is None:
        if strict_routing:
            raise ValueError("MOPD: strict routing requires the dataset 'ability' column")
        return [default_teacher] * n_samples
    if len(abilities) != n_samples:
        raise ValueError("MOPD: ability length differs from batch length")
    result = []
    for ability in abilities:
        teacher = ability_to_teacher.get(str(ability))
        if teacher is None:
            if strict_routing:
                raise ValueError(f"MOPD: unmapped ability {ability!r}")
            teacher = default_teacher
        result.append(teacher)
    return result


def validate_config(config):
    names = sorted(config.teachers)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("MOPD: teachers must be a nonempty name -> checkpoint mapping")
    if any(not isinstance(config.teachers[name], str) or not config.teachers[name] for name in names):
        raise ValueError("MOPD: each teacher must have a checkpoint path")
    default = config.get("default_teacher") or names[0]
    mapping = dict(config.get("ability_to_teacher", {}))
    if default not in names or any(name not in names for name in mapping.values()):
        raise ValueError("MOPD: default_teacher and all routing targets must be present in teachers")
    mode = config.get("multi_teacher_mode", "routing")
    if mode not in {"routing", "consensus"}:
        raise ValueError(f"Unknown multi_teacher_mode: {mode}")
    if mode == "routing" and len(names) > 1 and not mapping:
        raise ValueError("MOPD: multiple teachers require ability_to_teacher")
    return names, default, mapping


def init_teachers(worker):
    """Build teachers in a deterministic order and retain per-teacher tokenizer state."""
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local

    names, default, _ = validate_config(worker.config)
    base_config = worker.config
    worker.mopd_teachers = {}
    reference_vocab = None
    try:
        for name in names:
            config = OmegaConf.merge(
                OmegaConf.to_container(base_config, resolve=True),
                {"model": {"path": base_config.teachers[name]}},
            )
            worker.config = config
            tokenizer = hf_tokenizer(
                copy_to_local(config.model.path, use_shm=config.model.get("use_shm", False)),
                trust_remote_code=config.model.get("trust_remote_code", False),
            )
            # Log-probabilities and Top-K IDs refer to the same student token IDs.
            vocab = tokenizer.get_vocab()
            if reference_vocab is not None and vocab != reference_vocab:
                raise ValueError(f"MOPD: teacher {name!r} has an incompatible token-ID vocabulary")
            reference_vocab = vocab
            module = worker._build_model(config=config)
            state = {"reward_module": module, "_do_switch_chat_template": worker._do_switch_chat_template}
            if worker._do_switch_chat_template:
                state.update(tokenizer=worker.tokenizer, input_tokenizer=worker.input_tokenizer)
                if worker.input_tokenizer.get_vocab() != vocab:
                    raise ValueError(f"MOPD: teacher {name!r} and student tokenizer vocabularies differ")
            worker.mopd_teachers[name] = state
    finally:
        worker.config = base_config
    for key, value in worker.mopd_teachers[default].items():
        setattr(worker, key, value)


def compute_routed_scores(worker, data, kl_estimator="k1", force_teacher=None):
    """Reuse every existing score/Top-K option, then scatter results to original rows.

    Every rank evaluates the same teachers in the same order. Pad each teacher's
    local row count to its global maximum, so both fixed and dynamic microbatch
    scheduling remain collective-safe. Dummy outputs never reach the trainer.
    """
    names, default, mapping = validate_config(worker.config)
    route_error = None
    try:
        routes = [force_teacher] * len(data) if force_teacher is not None else route_by_ability(
            data.non_tensor_batch.get("ability"), len(data), mapping, default,
            worker.config.get("strict_routing", True),
        )
        if not len(data):
            raise ValueError("MOPD: each reward rank needs at least one input row")
    except ValueError as exc:
        route_error = exc
        routes = []
    # Sync validation before any rank enters a teacher's FSDP collectives.
    device = data.batch["input_ids"].device
    if dist.is_initialized() and dist.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    counts = torch.tensor(
        [int(route_error is not None)] + [routes.count(name) for name in names], device=device, dtype=torch.long
    )
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    if counts[0].item():
        raise ValueError(f"MOPD routing failed on a reward rank: {route_error or 'see peer rank error'}")
    tensors = {}
    original_state = {key: getattr(worker, key) for key in (
        "reward_module", "_do_switch_chat_template", "tokenizer", "input_tokenizer"
    ) if hasattr(worker, key)}
    try:
        for name, padded_count in zip(names, counts[1:].tolist(), strict=True):
            if not padded_count:
                continue
            rows = [i for i, selected in enumerate(routes) if selected == name]
            # Duplicate a real row rather than introducing all-padding inputs.
            padded_rows = rows + [rows[0] if rows else 0] * (padded_count - len(rows))
            for key, value in worker.mopd_teachers[name].items():
                setattr(worker, key, value)
            output = worker._compute_rm_score_single(data.select_idxs(padded_rows), kl_estimator)
            for key, value in output.batch.items():
                if key not in tensors:
                    tensors[key] = value.new_empty((len(data), *value.shape[1:]))
                if rows:
                    tensors[key][rows] = value[:len(rows)]
    finally:
        for key, value in original_state.items():
            setattr(worker, key, value)
    return DataProto.from_dict(tensors=tensors, non_tensors={"mopd_teacher": np.asarray(routes, dtype=object)})


def validate_consensus_training(config):
    """Reject options that would replace or reshape C-MOPD's signed reward."""
    rm = config.reward_model
    if rm.get("multi_teacher_mode", "routing") != "consensus":
        return
    validate_config(rm)
    epsilon = rm.get("consensus_epsilon", 0.0)
    epsilon = 0.0 if epsilon is None else float(epsilon)
    if epsilon < 0:
        raise ValueError("C-MOPD consensus_epsilon must be >= 0")
    if epsilon > 0 and not rm.get("ability_to_teacher", {}):
        raise ValueError(
            "C-MOPD consensus_epsilon>0 needs reward_model.ability_to_teacher "
            "so each sequence's domain teacher is well defined"
        )
    rollout = config.actor_rollout_ref.rollout
    if not rm.enable or rm.reward_manager != "naive":
        raise ValueError("C-MOPD requires reward_model.enable=True and reward_manager=naive")
    if config.algorithm.adv_estimator != "token_reward_direct" or config.algorithm.use_kl_in_reward:
        raise ValueError("C-MOPD requires token_reward_direct and use_kl_in_reward=False")
    if rollout.get("log_prob_top_k", 0) != 0:
        raise ValueError("C-MOPD compares sampled tokens only: log_prob_top_k must be 0")
    if rollout.temperature != 1.0 or rollout.get("teacher_temperature", 1.0) != 1.0:
        raise ValueError("C-MOPD requires student and teacher temperature=1 for model probability comparisons")
    if rollout.get("binary_opd", False):
        raise ValueError("C-MOPD and Binary OPD cannot be enabled together")
    if rollout.get("constant_reward", 0) != 0:
        raise ValueError("C-MOPD and constant_reward cannot be enabled together")
    if config.actor_rollout_ref.actor.use_kl_loss or config.actor_rollout_ref.actor.entropy_coeff != 0:
        raise ValueError("C-MOPD requires use_kl_loss=False and entropy_coeff=0")
    if rm.get("reward_kwargs", {}).get("enable_format_reward", False):
        raise ValueError("C-MOPD requires enable_format_reward=False")


def compute_consensus_scores(worker, data, kl_estimator="k1"):
    """All teachers score every sampled token; the sequence's domain teacher sets the sign.

    The domain teacher is the ability-routed one for each sequence (falling back to
    default_teacher when no ability_to_teacher mapping is configured). A token gets
    +1 when the domain teacher's probability exceeds the student's AND every other
    teacher's log(p_t/p_s) > -epsilon; it gets -1 for the mirror case (domain below
    the student AND every other teacher's log(p_t/p_s) < +epsilon). Other teachers
    therefore only veto when they disagree by more than epsilon. Ties on the domain
    teacher, nonfinite probabilities and response padding yield zero. With
    epsilon=0 this reduces to the previous unanimous-consensus rule.
    Comparisons use frozen rollout-policy old_log_probs, never the updated actor.
    Teacher outputs are consumed one at a time; only the compact per-teacher
    log-ratio tensors are retained.
    """
    if data.meta_info.get("log_prob_top_k", worker.config.get("log_prob_top_k", 0)) != 0:
        raise ValueError("C-MOPD requires log_prob_top_k=0")
    names, default, mapping = validate_config(worker.config)
    epsilon = float(worker.config.get("consensus_epsilon", 0.0) or 0.0)
    routes = route_by_ability(
        data.non_tensor_batch.get("ability"), len(data), mapping, default,
        worker.config.get("strict_routing", True),
    )
    log_ratios = {}
    entropy = None
    for name in names:
        output = compute_routed_scores(worker, data, kl_estimator, force_teacher=name)
        teacher = output.batch["teacher_logp_sampled"]
        student = data.batch["old_log_probs"].to(teacher.device)
        # float32 before the subtraction so the epsilon band is compared precisely
        log_ratios[name] = teacher.to(torch.float32) - student.to(torch.float32)
        if "teacher_entropy" in output.batch:
            value = output.batch["teacher_entropy"]
            entropy = value.clone() if entropy is None else entropy + value
    # [T, B, L] teacher-first stack of log(p_t / p_s)
    stacked = torch.stack([log_ratios[name] for name in names])
    valid = torch.isfinite(stacked).all(dim=0)
    teacher_idx = torch.as_tensor([names.index(route) for route in routes], device=stacked.device)
    batch_idx = torch.arange(stacked.shape[1], device=stacked.device)
    domain = stacked[teacher_idx, batch_idx]
    # Non-domain teachers decide the veto bounds; a single teacher has none.
    is_domain = torch.nn.functional.one_hot(teacher_idx, num_classes=len(names)).t().bool()
    is_other = (~is_domain).unsqueeze(-1)
    others_min = stacked.where(is_other, torch.tensor(float("inf"), device=stacked.device)).min(dim=0).values
    others_max = stacked.where(is_other, torch.tensor(float("-inf"), device=stacked.device)).max(dim=0).values
    higher = valid & (domain > 0) & (others_min > -epsilon)
    lower = valid & (domain < 0) & (others_max < epsilon)
    scores = higher.to(torch.float32) - lower.to(torch.float32)
    scores.masked_fill_(~data.batch["response_mask"].to(scores.device).bool(), 0)
    tensors = {"rm_scores": scores}
    if entropy is not None:
        tensors["teacher_entropy"] = entropy / len(names)
    return DataProto.from_dict(tensors=tensors)
