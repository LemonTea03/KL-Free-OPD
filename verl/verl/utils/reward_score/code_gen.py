# Copyright 2026 OPD contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dataset adapter for One-Shot-OPD's code_eval_v2 execution-based judge."""
import ast
import json
import traceback
import textwrap
from . import code_eval_v2 as reference

STDIN_SOURCES = {
    'taco': 'taco', 'BAAI/TACO': 'BAAI/TACO',
    'open-r1/codeforces': 'codeforces', 'codeforces': 'codeforces',
    'drproduck/livecodebench-v6': 'livecodebench',
    'livecodebench': 'livecodebench',
    'livecodebench/code_generation_lite': 'livecodebench/code_generation_lite',
}
ASSERT_SOURCES = {
    'openai/openai_humaneval', 'google-research-datasets/mbpp',
    'humanevalplus', 'evalplus/humanevalplus', 'mbppplus', 'evalplus/mbppplus',
}


def _adapt_assert(completion, gt):
    test = gt.get('test_code', gt.get('test'))
    if not isinstance(test, str) or not test.strip():
        raise ValueError('Missing assertion tests')
    entry = gt.get('entry_point')
    tree = ast.parse(test)
    defines_check = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'check' for n in tree.body)
    calls_check = any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Name) and n.value.func.id == 'check' for n in tree.body)
    if defines_check and not calls_check:
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError('Missing or invalid entry_point for check(candidate)')
        test += '\ncheck(' + entry + ')\n'
    code = reference.extract_code_from_model(completion)
    if code is not None and gt.get('prompt'):
        try:
            compile(code, "<model-completion>", "exec")
        except (SyntaxError, IndentationError):
            # Existing MBPP prompts also allow an indented function body.
            code = gt['prompt'] + '\n' + textwrap.indent(code, '    ')
            completion = '```python\n' + code + '\n```'
    return completion, {'test': test, 'entry_point': entry}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    try:
        gt = reference._load_ground_truth(ground_truth)
        if data_source in STDIN_SOURCES:
            if isinstance(gt, dict):
                inputs, outputs = gt.get('inputs', []), gt.get('outputs', [])
                if not inputs or len(inputs) != len(outputs):
                    raise ValueError('Empty or mismatched input/output tests')
                # TACO Codewars uses [expected_return_value] per test, unlike LCB.
                # Metadata scopes this conversion; a LeetCode return list is preserved.
                if data_source in {'taco', 'BAAI/TACO'} and (extra_info or {}).get('source') == 'codewars' and gt.get('fn_name'):
                    converted = []
                    for out in outputs:
                        value = json.loads(out) if isinstance(out, str) else out
                        if isinstance(value, list) and len(value) == 1:
                            value = value[0]
                        converted.append(json.dumps(value))
                    gt = {**gt, 'outputs': converted}
            result = reference.reward_func(STDIN_SOURCES[data_source], solution_str, gt, extra_info)
        elif data_source in ASSERT_SOURCES:
            solution_str, gt = _adapt_assert(solution_str, gt)
            result = reference.reward_func('humanevalplus', solution_str, gt, extra_info)
        else:
            raise NotImplementedError(f'Unknown code source: {data_source}')
        return float(result['score'])
    except Exception:
        traceback.print_exc()
        return 0.0
