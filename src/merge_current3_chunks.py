#!/usr/bin/env python3
"""Merge data-parallel current3 evaluator shards and recompute metrics."""

import argparse
import json
import os

from iccd_minimal import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", action="append", required=True, dest="chunk_dirs")
    parser.add_argument("--proposed-output-dir", required=True)
    parser.add_argument("--generated-icl-output-dir", required=True)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def main():
    args = parse_args()
    proposed_predictions = []
    regular_predictions = []
    iccd_predictions = []
    generated_predictions = []
    proposed_scores = {
        "scores_regular_icl.jsonl": [],
        "scores_negative_context.jsonl": [],
        "scores_iccd.jsonl": [],
    }
    generated_scores = []

    for chunk_dir in args.chunk_dirs:
        proposed_dir = os.path.join(chunk_dir, "proposed")
        generated_dir = os.path.join(chunk_dir, "generated_icl")
        proposed_predictions.extend(read_jsonl(os.path.join(proposed_dir, "predictions.jsonl")))
        regular_predictions.extend(
            read_jsonl(os.path.join(proposed_dir, "predictions_regular_icl.jsonl"))
        )
        iccd_predictions.extend(
            read_jsonl(os.path.join(proposed_dir, "predictions_iccd.jsonl"))
        )
        for filename in proposed_scores:
            proposed_scores[filename].extend(read_jsonl(os.path.join(proposed_dir, filename)))
        generated_predictions.extend(
            read_jsonl(os.path.join(generated_dir, "predictions_generated_icl.jsonl"))
        )
        generated_scores.extend(
            read_jsonl(os.path.join(generated_dir, "scores_generated_icl.jsonl"))
        )

    test_indices = [row["test_idx"] for row in proposed_predictions]
    if test_indices != list(range(len(proposed_predictions))):
        raise RuntimeError("Evaluator chunks do not cover the test split exactly once in order")
    generated_indices = [row["test_idx"] for row in generated_predictions]
    if generated_indices != test_indices:
        raise RuntimeError("Generated-ICL shard order does not match proposed-evaluator shards")

    golds = [row["gold"] for row in proposed_predictions]
    proposed_metrics = {
        "regular_icl": evaluate(golds, [row["pred_regular_icl"] for row in proposed_predictions]),
        "iccd": evaluate(golds, [row["pred_iccd"] for row in proposed_predictions]),
    }
    generated_metrics = {
        "generated_icl": evaluate(golds, [row["pred"] for row in generated_predictions])
    }

    os.makedirs(args.proposed_output_dir, exist_ok=True)
    os.makedirs(args.generated_icl_output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.proposed_output_dir, "predictions.jsonl"), proposed_predictions)
    write_jsonl(
        os.path.join(args.proposed_output_dir, "predictions_regular_icl.jsonl"),
        regular_predictions,
    )
    write_jsonl(
        os.path.join(args.proposed_output_dir, "predictions_iccd.jsonl"), iccd_predictions
    )
    for filename, rows in proposed_scores.items():
        write_jsonl(os.path.join(args.proposed_output_dir, filename), rows)
    write_json(os.path.join(args.proposed_output_dir, "metrics.json"), proposed_metrics)

    write_jsonl(
        os.path.join(args.generated_icl_output_dir, "predictions_generated_icl.jsonl"),
        generated_predictions,
    )
    write_jsonl(
        os.path.join(args.generated_icl_output_dir, "scores_generated_icl.jsonl"), generated_scores
    )
    write_json(os.path.join(args.generated_icl_output_dir, "metrics.json"), generated_metrics)

    config = read_json(os.path.join(args.chunk_dirs[0], "proposed", "config.json"))
    config["merged_from_chunk_dirs"] = args.chunk_dirs
    config["merged_total_samples"] = len(proposed_predictions)
    config["test_sample_offset"] = 0
    config["max_test_samples"] = None
    write_json(os.path.join(args.proposed_output_dir, "config.json"), config)
    write_json(os.path.join(args.generated_icl_output_dir, "config.json"), config)

    print(f"merged {len(proposed_predictions)} rows from {len(args.chunk_dirs)} evaluator chunks")
    print(json.dumps({**proposed_metrics, **generated_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
