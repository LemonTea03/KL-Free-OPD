# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Opt-in per-sample judge stack dumps; never kill or alter reward results."""

import faulthandler
import os
import sys
import time
from contextlib import contextmanager


@contextmanager
def trace_judge(source, index):
    timeout = float(os.environ.get("OPD_JUDGE_STACK_TIMEOUT", "0"))
    if timeout <= 0:
        yield
        return
    started = time.monotonic()
    label = f"pid={os.getpid()} source={source!r} index={index!r}"
    print(f"[judge-trace] begin {label}", file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(timeout, repeat=True, file=sys.stderr, exit=False)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        print(f"[judge-trace] end {label} elapsed={time.monotonic() - started:.3f}s", file=sys.stderr, flush=True)
