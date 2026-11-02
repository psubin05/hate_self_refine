# Hate Self-Refine

Official experiment code and reported results for self-refined negative
demonstrations in Korean hate-speech classification.

The repository contains the prompts, evaluation code, paper-facing scripts,
and full-precision metric files used in the paper. Model weights, the KOLD
dataset, predictions, caches, checkpoints, and execution logs are not
redistributed.

## Repository layout

```text
data/                 Few-shot split indices/data and dataset instructions
paper/                LaTeX table fragments used in the paper
prompts/              Classification, generation, critique, and revision prompts
results/raw/          Full-precision metrics from completed paper runs
scripts/              Portable paper experiment entry points
src/                  Generation, evaluation, merging, and baseline code
```

## Reported self-refinement ablation

| Method | Accuracy | Macro-F1 | Normal F1 | Offensive F1 | Hate F1 |
|---|---:|---:|---:|---:|---:|
| Regular ICL | 0.4380 | 0.4039 | 0.5870 | 0.4065 | 0.2181 |
| Initial transformed draft | 0.4509 | 0.4450 | 0.5491 | 0.4020 | 0.3840 |
| Revision without critique | 0.4452 | 0.4411 | 0.5375 | 0.3981 | **0.3876** |
| **Proposed method** | **0.4751** | **0.4636** | **0.5991** | **0.4100** | 0.3818 |

Full-precision metrics and the mapping from paper rows to source files are in
[`results/`](results/README.md). The paper-ready LaTeX table is in
[`paper/self_refinement_ablation_table.tex`](paper/self_refinement_ablation_table.tex).

## Setup

Python 3.10+ and a CUDA environment supported by vLLM are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Obtain KOLD from its official distribution and place the files at:

```text
data/kold/train.csv
data/kold/valid.csv
data/kold/test.csv
```

The expected columns are `guid`, `title`, `comment`, `OFF`, and `TGT`. See
[`data/README.md`](data/README.md) for details.

## Reproduction

Run the proposed Qwen3-8B experiment:

```bash
bash scripts/run_proposed.sh
```

Then reproduce the controlled self-refinement ablation with the same cached
initial drafts:

```bash
bash scripts/run_self_refinement_ablation.sh
```

Other paper baselines:

```bash
bash scripts/run_reverse_mapping_baselines.sh
bash scripts/run_supervised_baseline.sh
```

All scripts accept environment-variable overrides such as `MODEL`,
`PYTHON_BIN`, `BATCH_SIZE`, and `OUTPUT_ROOT`; their headers list the available
options. Outputs are written below `results/runs/`, which is intentionally
ignored by Git. Self-refinement generation is stochastic, so exact values can
vary with model/runtime versions and hardware. The committed metric files are
the authoritative values from the completed paper runs.

## Notes

- The full KOLD dataset is not included.
- Raw predictions may contain sensitive text and are not included.
- No model weights or local machine paths are committed.
- `archive/` documents nonessential development artifacts and is not required
  to reproduce the reported experiments.
