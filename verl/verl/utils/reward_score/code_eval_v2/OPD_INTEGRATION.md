Source: One-Shot-OPD commit 6b6b1782cc3f04749dd077de19353f839f6bc048.
LCB _lcb_runner.py and _base_imports.py are copied unchanged, with their license.
Local changes: forkserver workers and deterministic process cleanup; empty/missing
results fail; HumanEval runner uses sys.executable. code_gen.py adapts source
names, test_code/check(entry_point), and TACO Codewars expected-value wrappers.
The original per-case timeout, total-time budget, code extraction, and result
comparison are inherited from the reference judge. Outputs are binary 0/1.
