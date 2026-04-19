# Benchmarks

This directory contains the core validation logs and scripts that back up the benchmark and ablation claims in the main `README.md`.

## Included Files
- `benchmark_results.log`: The raw terminal output of `python demo.py`, which evaluates our task-aware Smart Policy against a Random Baseline over multiple random seeds (42, 99, 7, 256, 1337).

## Benchmark Protocol
- **Evaluation Script:** `demo.py`
- **Environment:** `SocialContractOpenEnv`
- **Seed List:** `[42, 99, 7, 256, 1337]` for true generalization testing.
- **Model / Policy:** The local multi-seed run uses a rule-based expert system ("Smart Policy") that applies Taylor Rules and Phillips Curve logic to the 8 levers. The `GPT-4o-mini` results reported in the main README were executed using `inference.py` (Temperature 0.10, Max Steps dynamically based on the task config).
- **Prompt Logic:** Uses 5-step rolling sliding history with trend indicators and direction-change dampening (action smoothing).
- **Reproduce Locally:** 
  ```bash
  pip install -r requirements.txt
  python demo.py
  ```

*Note on Random Baseline scores: tasks such as "Task 1 (Maintain Stability)" or "Task 5 (Pandemic)" can record random scores of ~0.40 to ~0.50. This is because the environment grading heavily features trajectory evaluation. Random policies can occasionally stumble upon minor improvements, inflating the base mathematical mean, but they absolutely fail the critical "simultaneous achievement gates" necessary to clear a score of 0.70.*
