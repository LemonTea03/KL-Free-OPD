# Copyright 2024 PRIME team and/or its affiliates
#
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

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import sys
import traceback
from typing import Optional

from .testing_util import run_test


def _temp_run(sample, generation, debug, timeout):
    """Pool worker body: execute the generated code and RETURN the result.

    Unlike the old Manager-based version, nothing is shared back via proxy
    objects, so a wedged worker can never leave the caller blocked on a
    manager connection.
    """
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            return res, metadata
        except Exception:
            traceback.print_exc(10)
            return [-1 for i in range(len(sample["inputs"]))], {}


def _judge_worker_init():
    # Hard kernel-level walls so pathological generated code cannot eat the
    # whole node: RLIMIT_CPU kills C-level busy loops that signal.alarm cannot
    # interrupt, RLIMIT_AS caps runaway allocations (run_test applies no limit).
    # The worker inherits the forkserver's preloaded address space (torch etc.),
    # so the memory cap is placed relative to the inherited size, not at a fixed
    # small value that would fail every allocation.
    try:
        import resource

        headroom_gb = int(os.environ.get("OPD_JUDGE_MEM_LIMIT_GB", "8"))
        with open("/proc/self/status") as status:
            vm_kb = next(int(line.split()[1]) for line in status if line.startswith("VmSize:"))
        mem_bytes = vm_kb * 1024 + headroom_gb * 1024**3
        cpu_seconds = int(os.environ.get("OPD_JUDGE_CPU_LIMIT_S", "30"))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        traceback.print_exc(5)


# Judge workers must not be forked from the Ray actor itself: the TaskRunner process
# carries live gRPC channels and torch threads, and fork()-copying that state can wedge
# the actor's gRPC (observed as ActorUnavailableError: keepalive watchdog timeout).
# The forkserver forks judge workers from a clean helper process instead. Preloading
# this module into the forkserver keeps per-sample cost low; a plain "spawn"
# context would re-import torch in every worker (~4s per sample).
#
# 2026-09-18: the Manager/Process-per-call scheme still wedged the TaskRunner inside
# multiprocessing plumbing (3/3 runs died as keepalive watchdog timeouts mid-judge,
# e.g. logs/cg-opd-qwen3-4b-math-code.log run at step 2 and step 10). Every
# Manager()/proxy call is an unbounded wait once the helper processes misbehave.
# We now run samples on a persistent forkserver Pool and bound EVERY wait with
# AsyncResult.get(timeout); on expiry the pool is terminated and rebuilt and the
# sample scores as failed, so training can never hang on the judge again.
_judge_pool = None


def _get_judge_pool():
    global _judge_pool
    if _judge_pool is None:
        multiprocessing.set_forkserver_preload([__name__])
        ctx = multiprocessing.get_context("forkserver")
        workers = int(os.environ.get("OPD_JUDGE_POOL_SIZE", "2"))
        _judge_pool = ctx.Pool(
            processes=workers,
            initializer=_judge_worker_init,
            # One pristine worker per sample: run_test's reliability_guard nulls
            # modules (os.remove, shutil, subprocess, ...) in the worker, and a
            # reused worker then MIS-SCORES subsequent samples (verified: stdin
            # samples return -1 after any earlier sample ran in the same worker).
            maxtasksperchild=1,
        )
    return _judge_pool


def _recycle_judge_pool(reason):
    global _judge_pool
    print(f"[judge] {reason}; recycling judge pool", flush=True)
    pool, _judge_pool = _judge_pool, None
    if pool is None:
        return
    try:
        pool.terminate()
    except Exception:
        pass
    try:
        pool.join(timeout=5)
    except Exception:
        pass


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`. Unlike run_test's per-test alarm, this bound also covers the
    multiprocessing plumbing itself, so a broken judge costs one sample, not the run."""

    if not in_outs:
        return [-1], []

    grace = int(os.environ.get("OPD_JUDGE_GRACE_S", "10"))
    try:
        pool = _get_judge_pool()
        async_result = pool.apply_async(_temp_run, (in_outs, generation, debug, timeout))
        result, metadata = async_result.get(timeout=timeout + grace)
        return result, [metadata]
    except multiprocessing.TimeoutError:
        if debug:
            print(f"[judge] global timeout after {timeout}+{grace}s, killing judge pool", flush=True)
        _recycle_judge_pool(f"global timeout after {timeout}+{grace}s")
        return [-1 for _ in range(len(in_outs["inputs"]))], []
    except Exception as e:
        # Pool broken (worker died hard, connection lost, ...): rebuild and fail the sample.
        if debug:
            print(f"[judge] pool error: {e!r}", flush=True)
        _recycle_judge_pool(f"pool error: {e!r}")
        return [-1 for _ in range(len(in_outs["inputs"]))], []
