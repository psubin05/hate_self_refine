# Reported results

`summary.csv` is a flat index of every retained final metric. `raw/` contains
the original `metrics.json` payloads copied from the completed experiment
directories before intermediate artifacts were removed.

## Self-refinement ablation table mapping

| Paper row | Raw result | Metric key |
|---|---|---|
| Regular ICL | `fact_only_critic_minimal_prompt_proposed_initial_iter1_qwen3_8b_test_full.json` | `regular_icl` |
| Initial transformed draft | `condition1_initial_only_proposed_initial_qwen3_8b_test_full.json` | `iccd` |
| Revision without critique | `no_feedback_revision_proposed_initial_qwen3_8b_test_full.json` | `iccd` |
| Proposed method | `fact_only_critic_minimal_prompt_proposed_initial_iter1_qwen3_8b_test_full.json` | `iccd` |

The displayed paper values are rounded to four decimal places. Full classwise
precision, recall, F1, and support are available in each raw JSON file.

The `paired_initial` result contains only 300 samples and is retained for
traceability; it is not the full-test result used for the paper row.

## Files intentionally excluded

`results/runs/`, training checkpoints, per-example predictions, generation
caches, and logs are runtime artifacts and are excluded from the repository.
They are large, may contain dataset text, and are not needed to inspect the
reported aggregate results.
