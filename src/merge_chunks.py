#!/usr/bin/env python3
"""Merge single-GPU chunk output directories produced by iccd_minimal.py
--test-sample-offset/--max-test-samples into one combined result directory.

Accuracy and macro-F1 don't average linearly across chunks of different
sizes, so metrics.json is recomputed from the merged predictions rather than
combined from each chunk's own metrics.json.

Usage:
  python3 merge_chunks.py --chunk-dir chunk0 --chunk-dir chunk1 --output-dir merged
"""

import argparse
import json
import os
import shutil

from iccd_minimal import evaluate


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        action="append",
        required=True,
        dest="chunk_dirs",
        help="Repeat for each chunk, in row order (offset ascending).",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    merged_predictions = []
    merged_pred_regular = []
    merged_pred_iccd = []
    score_filenames = (
        "scores_regular_icl.jsonl",
        "scores_negative_context.jsonl",
        "scores_iccd.jsonl",
    )
    merged_scores = {filename: [] for filename in score_filenames}

    for shard_index, chunk_dir in enumerate(args.chunk_dirs):
        predictions_path = os.path.join(chunk_dir, "predictions.jsonl")
        if not os.path.isfile(predictions_path):
            raise FileNotFoundError(
                f"{predictions_path} is missing -- did this chunk finish?"
            )
        merged_predictions.extend(read_jsonl(predictions_path))
        merged_pred_regular.extend(
            read_jsonl(os.path.join(chunk_dir, "predictions_regular_icl.jsonl"))
        )
        merged_pred_iccd.extend(
            read_jsonl(os.path.join(chunk_dir, "predictions_iccd.jsonl"))
        )
        for filename in score_filenames:
            path = os.path.join(chunk_dir, filename)
            if os.path.isfile(path):
                merged_scores[filename].extend(read_jsonl(path))

        # Preserve each chunk's self-refine negative-context cache under the
        # shard-numbered filename --self-refine-initial-cache-file/
        # --self-refine-final-cache-file already glob for
        # (self_refine_negative_cache_shard*.jsonl), so a later run can seed
        # straight from this merged directory.
        cache_path = os.path.join(chunk_dir, "self_refine_negative_cache.jsonl")
        if os.path.isfile(cache_path):
            shutil.copyfile(
                cache_path,
                os.path.join(
                    args.output_dir,
                    f"self_refine_negative_cache_shard{shard_index}.jsonl",
                ),
            )

    golds = [row["gold"] for row in merged_predictions]
    preds_regular = [row["pred_regular_icl"] for row in merged_predictions]
    preds_iccd = [row["pred_iccd"] for row in merged_predictions]
    metrics = {
        "regular_icl": evaluate(golds, preds_regular),
        "iccd": evaluate(golds, preds_iccd),
    }

    write_jsonl(os.path.join(args.output_dir, "predictions.jsonl"), merged_predictions)
    write_jsonl(
        os.path.join(args.output_dir, "predictions_regular_icl.jsonl"),
        merged_pred_regular,
    )
    write_jsonl(
        os.path.join(args.output_dir, "predictions_iccd.jsonl"), merged_pred_iccd
    )
    for filename, rows in merged_scores.items():
        if rows:
            write_jsonl(os.path.join(args.output_dir, filename), rows)

    with open(
        os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Keep one representative config.json, annotated with the merge sources
    # (chunk-specific fields like test_sample_offset/test_label_distribution
    # only describe that one chunk, not the merged whole).
    first_config_path = os.path.join(args.chunk_dirs[0], "config.json")
    if os.path.isfile(first_config_path):
        with open(first_config_path, encoding="utf-8") as f:
            config = json.load(f)
        config["merged_from_chunk_dirs"] = args.chunk_dirs
        config["merged_total_samples"] = len(merged_predictions)
        with open(
            os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(f"merged {len(merged_predictions)} rows from {len(args.chunk_dirs)} chunks")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
