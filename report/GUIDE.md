# Phase 9 Guide — Final Research Report

Work through this top to bottom. Each step has a clear action, a verification command, and a checkpoint. Do not skip ahead.

---

## 0. Git Setup

```bash
git checkout main && git pull
git checkout -b phase-9-report
```

**Checkpoint:** `git branch` shows `phase-9-report` as active.

---

## 1. Write Failing Tests First (RED)

Create `tests/test_report_verification.py`. It must import from `scripts/verify_report` (which does not exist yet) and reference `report/report.md` (which does not exist yet). Every test must fail.

Write exactly these 7 tests, one assertion each:

| Test function | What to assert |
|---|---|
| `test_verify_report_module_importable` | `from scripts.verify_report import check_sections, check_plots, check_results, check_word_count` does not raise |
| `test_report_file_exists` | `Path("report/report.md").exists()` is `True` |
| `test_required_sections_present` | `check_sections(report_text)` returns an empty list (no missing headings) |
| `test_all_plot_references_resolve` | `check_plots(report_text, plots_dir)` returns an empty list (no broken image refs) |
| `test_all_result_json_references_resolve` | `check_results(report_text, results_dir)` returns an empty list (no broken JSON refs) |
| `test_report_minimum_word_count` | `check_word_count(report_text, MIN_WORD_COUNT)` is `True` |
| `test_generate_report_integration` | `subprocess.run(["python3", "scripts/verify_report.py"]).returncode == 0` |

Run the suite and confirm all 7 fail:
```bash
pytest tests/test_report_verification.py -v
```

Commit the red tests:
```bash
git add tests/test_report_verification.py
git commit -m "test: add failing report verification tests (Phase 9 red)"
```

---

## 2. Write the Verification Script (first tests go green)

Create `scripts/verify_report.py` using only stdlib (`pathlib`, `re`, `sys`). No new packages.

Required named constants (no inline magic numbers):
```python
REQUIRED_HEADINGS = [
    "## Abstract",
    "## 1. Introduction",
    "## 2. System Architecture",
    "## 3. Experimental Design",
    "## 4. Results and Analysis",
    "## 5. Discussion",
    "## 6. Conclusions and Recommendations",
]
MIN_WORD_COUNT = 3000
PLOTS_DIR = Path("benchmarks/plots")
RESULTS_DIR = Path("benchmarks/results")
```

Required functions (one responsibility each):
- `check_sections(report_text: str) -> list[str]` — returns list of any missing headings
- `check_plots(report_text: str, plots_dir: Path) -> list[str]` — finds all `fig_0N_*.png` mentions in the text, returns any that are not present in `plots_dir`
- `check_results(report_text: str, results_dir: Path) -> list[str]` — finds all `*_YYYYMMDD_HHMMSS.json` mentions, returns any missing from `results_dir`
- `check_word_count(report_text: str, minimum: int) -> bool` — returns `True` if word count >= minimum
- `main()` — reads `report/report.md`, runs all four checks, prints one PASS/FAIL line per check, exits 1 if any fail

Python comment style: `#text` — no space, all lowercase.

Run the suite:
```bash
pytest tests/test_report_verification.py -v
```

`test_verify_report_module_importable` should now be green. Others still red. Commit:
```bash
git add scripts/verify_report.py
git commit -m "feat: add report verification script (Phase 9)"
```

---

## 3. Set Up the Report Directory

```bash
mkdir -p report
touch report/.gitkeep
git add report/.gitkeep
git commit -m "chore: create report directory"
```

---

## 4. Write the Report (remaining tests go green)

Create `report/report.md`. Work section by section. After each section run:
```bash
pytest tests/test_report_verification.py -v
python3 -c "print(len(open('report/report.md').read().split()))"
```

### Data sources — read these before writing each section

| What you need | Where to look |
|---|---|
| Latency numbers by format | `benchmarks/results/latency_benchmark_20260426_102230.json` |
| Batch scaling curves | `benchmarks/results/batch_experiments_20260426_102622.json` |
| Worker scaling / async vs sync | `benchmarks/results/worker_experiments_20260425_230144.json` |
| gRPC vs REST | `benchmarks/results/grpc_experiments_20260425_015539.json` |
| CPU vs MPS | `benchmarks/results/device_benchmark_20260425_233946.json` |
| Design rationale | `.claude/rules/design-decisions.md` |
| First-run failures and pivots | `info/notes.txt` |
| Why decisions were made | `info/why.txt` |
| Architecture diagrams and component table | `.claude/common/architecture.md` |

### Section-by-section prompts

**Abstract**
Read all five result JSONs. Write 150 words answering: what system was built, what six experiments were run, and what the three most important findings were. Do not pad — be precise.

**1. Introduction**
Answer: why does inference optimization matter when model accuracy gets all the attention? What is the scope (CIFAR-10, MobileNetV2, Apple Silicon)? List six numbered research questions, one per experiment.

**2. System Architecture**
Copy the ASCII pipeline from `.claude/common/architecture.md`. Describe each component in the table there. For each key design decision (thread pinning, RPOP vs BRPOP, lazy loading, format-agnostic predictor, enqueue timestamp, MPS synchronize) — state what was decided and why, citing `design-decisions.md`.

**3. Experimental Design**
State hardware (Apple M-series, macOS, arm64), software stack (Python 3.11, PyTorch, ONNX Runtime, FastAPI, Redis, gRPC). Describe the three fairness fixes applied during Phase 7/8 (found in `info/notes.txt`): `torch.no_grad`, `perf_counter` vs `time.time`, and ONNX inter-op thread constraint. Explain what each fix changed and why results before the fix were invalid.

**4. Results and Analysis** — one subsection per experiment

*4.1 Model Format Latency — `fig_01_latency_formats.png`*
Read `latency_benchmark_20260426_102230.json`. Extract single-image latency per format. Compute relative differences. Explain the mechanism: ONNX graph-level optimization vs PyTorch eager mode overhead. Cite the fairness fix: before constraining inter-op threads, what did ONNX appear to be, and what is the corrected gap?

*4.2 Batch Size Scaling — `fig_02_batch_latency.png`, `fig_03_batch_throughput.png`*
Read `batch_experiments_20260426_102622.json`. Find the batch=16 latency for TorchScript and PyTorch. What happens? How does ONNX behave across the same range? At the system's operating point (batch=8), which format has the highest throughput? What does this mean for production?

*4.3 Async vs Sync and Worker Scaling — `fig_04_worker_scaling.png`*
Read `worker_experiments_20260425_230144.json`. What RPS does the sync endpoint achieve at concurrency=20? What happens at concurrency=50? What does 4-worker async achieve at concurrency=50? At 8 workers, what happens and why? Compute the multiplier between best async and best sync.

*4.4 Protocol Comparison — `fig_05_protocol_comparison.png`*
Read `grpc_experiments_20260425_015539.json`. Compare: payload size, RPS, p95 latency. Which wins on each metric? Why does gRPC's protocol advantage not translate cleanly into latency improvement given a shared backend queue?

*4.5 CPU vs MPS — `fig_06_device_latency.png`, `fig_07_device_throughput.png`*
Read `device_benchmark_20260425_233946.json`. At batch=1, which device wins and why? At batch=8? At batch=64, what is the throughput multiplier? Why is ONNX CPU at batch=8 faster than MPS TorchScript at batch=8? Note that ONNX has no MPS backend.

**5. Discussion**
Answer: given all five experiment results together, how would you choose a format for production? What is the practical safety bound on batch size and why is it a safety bound (not just a tuning choice)? Does gRPC meaningfully matter at this scale? When does MPS pay off and when does it not?

**6. Conclusions and Recommendations**
Six numbered conclusions — one per research question from the Introduction. Then three actionable recommendations (format choice, worker count, device selection), each with the specific condition under which it applies.

---

## 5. Verify and Test

```bash
#run full suite
pytest tests/ -v

#run the integration smoke test
python3 scripts/verify_report.py

#check word count
python3 -c "print(len(open('report/report.md').read().split()), 'words')"
```

All 7 new tests and all 45+ existing tests must pass. `verify_report.py` must exit 0.

---

## 6. Invoke the Research Log Scribe

Once the report is written, use the `research-log-scribe` agent to document any decisions, surprises, or pivots you encountered while writing it. Examples of things worth logging:

- A number from a JSON that didn't match your expectation — what did you expect and what did you find?
- A section that needed a different structure than planned — what and why?
- A benchmark result that is particularly counterintuitive and report-worthy
- Anything you had to revisit or rewrite

Use this prompt to invoke it:
```
Phase 9 (research report) is complete. [Describe what happened: any data surprises, structural pivots, anything that was counterintuitive.] Please log everything to info/notes.txt and info/why.txt.
```

---

## 7. Documentation Sync

Work through this checklist in order:

- [ ] `README.md` — add Phase 9 to the phase table; add `report/report.md` to the project structure; add a "Research Report" section near the bottom
- [ ] `.claude/rules/research-scope.md` — mark Phase 9 complete in the phase table; remove it from Remaining Work
- [ ] `info/why.txt` — add an entry explaining why Markdown was chosen over LaTeX for the report format
- [ ] `requirements.txt` — confirm no new packages were added (`pip freeze` vs the current file)
- [ ] `.claude/rules/design-decisions.md` — only add an entry if a genuinely load-bearing report-structure decision was made (e.g., if you chose a non-obvious section order)

---

## 8. Commit and PR

```bash
git add report/report.md scripts/verify_report.py tests/test_report_verification.py README.md
git add .claude/rules/research-scope.md info/notes.txt info/why.txt
git status  #verify nothing sensitive is staged
git commit -m "feat: add Phase 9 final research report with verification"
git push -u origin phase-9-report
gh pr create --base main --title "Phase 9: Final Research Report"
```

---

## Quick Reference

| File | Purpose |
|---|---|
| `benchmarks/results/latency_benchmark_20260426_102230.json` | Format latency data (newest run) |
| `benchmarks/results/batch_experiments_20260426_102622.json` | Batch scaling data (newest run) |
| `benchmarks/results/worker_experiments_20260425_230144.json` | Worker scaling data (newest run) |
| `benchmarks/results/grpc_experiments_20260425_015539.json` | Protocol comparison data |
| `benchmarks/results/device_benchmark_20260425_233946.json` | CPU vs MPS data |
| `benchmarks/plots/fig_01_latency_formats.png` through `fig_07_*` | All 7 figures |
| `.claude/rules/design-decisions.md` | Rationale for every load-bearing decision |
| `info/notes.txt` | First-run failures and research observations |
| `info/why.txt` | Deeper rationale behind design pivots |
| `.claude/common/architecture.md` | ASCII pipeline + component table |
