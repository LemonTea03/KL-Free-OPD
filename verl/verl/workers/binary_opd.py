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
"""Configuration checks for sampled-token Binary OPD and constant-reward experiments."""

import math


def validate_binary_opd_training(config):
    rollout = config.actor_rollout_ref.rollout
    constant = rollout.get("constant_reward", 0)
    if isinstance(constant, bool) or constant not in (-1, 0, 1):
        raise ValueError("constant_reward must be 0 (disabled), +1 or -1")
    binary = rollout.get("binary_opd", False)
    if binary and constant != 0:
        raise ValueError("Binary OPD and constant_reward cannot be enabled together")
    if constant != 0:
        lower = float(rollout.get("reward_lower_bound", "-inf"))
        upper = float(rollout.get("reward_upper_bound", "inf"))
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            raise ValueError("Constant OPD requires non-NaN bounds with lower_bound <= upper_bound")
    if not binary and constant == 0:
        return
    mode = "Binary OPD" if binary else "Constant OPD"
    if config.reward_model.get("multi_teacher_mode", "routing") == "consensus":
        raise ValueError(f"C-MOPD and {mode} cannot be enabled together")
    if not config.reward_model.enable or config.reward_model.reward_manager != "naive":
        raise ValueError(f"{mode} requires reward_model.enable=True and reward_manager=naive")
    if config.algorithm.adv_estimator != "token_reward_direct" or config.algorithm.use_kl_in_reward:
        raise ValueError(f"{mode} requires token_reward_direct and use_kl_in_reward=False")
    if rollout.get("log_prob_top_k", 0) != 0:
        raise ValueError(f"{mode} requires log_prob_top_k=0 (sampled tokens)")
    if rollout.temperature != 1.0 or rollout.get("teacher_temperature", 1.0) != 1.0:
        raise ValueError(f"{mode} requires student and teacher temperature=1")
    actor = config.actor_rollout_ref.actor
    if actor.use_kl_loss or actor.entropy_coeff != 0:
        raise ValueError(f"{mode} requires use_kl_loss=False and entropy_coeff=0")
    if config.reward_model.get("reward_kwargs", {}).get("enable_format_reward", False):
        raise ValueError(f"{mode} requires enable_format_reward=False")
