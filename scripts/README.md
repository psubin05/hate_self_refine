# Experiment scripts

These are the portable entry points for the experiments reported in the paper.
Run every command from the repository root.

| Script | Purpose | Default output |
|---|---|---|
| `run_proposed.sh` | Proposed fact-only self-refinement with Qwen3-8B | `results/runs/proposed/` |
| `run_self_refinement_ablation.sh` | Initial-only and no-feedback revisions sharing the proposed run's initial drafts | `results/runs/self_refinement_ablation/` |
| `run_reverse_mapping_baselines.sh` | Random, BM25, and dense-retrieval no-generation baselines | `results/runs/reverse_mapping/` |
| `run_supervised_baseline.sh` | Full-data encoder baseline | `results/runs/supervised/` |

Common overrides:

```bash
MODEL=Qwen/Qwen3-8B BATCH_SIZE=8 bash scripts/run_proposed.sh
MODEL=klue/roberta-base EPOCHS=3 bash scripts/run_supervised_baseline.sh
```

The self-refinement ablation must be run after `run_proposed.sh` because it
reuses `results/runs/proposed/self_refine_negative_cache.jsonl`. This fixes the
initial generations across conditions and isolates the refinement stage.
