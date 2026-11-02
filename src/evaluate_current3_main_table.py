#!/usr/bin/env python3
"""Evaluate current condition-(3) contexts for two main-table rows at once.

The source Qwen3-8B run fixes, per test item, the positive demonstrations and
the exact accepted/fallback negative demonstrations.  This evaluator rebuilds
those contexts chunk by chunk, then scores regular ICL, proposed ICCD, and the
generated-sentence ICL control with one model load.
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import iccd_minimal as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--source-result-dir", required=True)
    parser.add_argument("--proposed-output-dir", required=True)
    parser.add_argument("--generated-icl-output-dir", required=True)
    parser.add_argument(
        "--train-file", default=os.path.join(core.PROJECT_ROOT, "data", "kold", "train.csv")
    )
    parser.add_argument(
        "--test-file", default=os.path.join(core.PROJECT_ROOT, "data", "kold", "test.csv")
    )
    parser.add_argument(
        "--prompt-dir",
        default=os.path.join(core.PROJECT_ROOT, "prompts", "topic_aligned_random"),
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--test-sample-offset", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(core.PROJECT_ROOT, ".cache", "current3_main_table"),
    )
    parser.add_argument("--use-title", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_chunk_cache(path):
    rows = {}
    for row in read_jsonl(path):
        guid = row.get("guid")
        if guid and guid in rows:
            raise RuntimeError(f"Duplicate generated guid in one source chunk: {guid}")
        if guid:
            rows[guid] = row
    return rows


def load_merged_cache(paths):
    """Load legacy root-level cache shards used by the original 9-shot run."""
    rows = {}
    ambiguous_guids = set()
    for path in paths:
        for row in read_jsonl(path):
            guid = row.get("guid")
            if not guid:
                continue
            cache_key = row.get("cache_key")
            if cache_key:
                rows[cache_key] = row
            if guid in ambiguous_guids:
                continue
            previous = rows.get(guid)
            if previous is not None and previous != row:
                rows.pop(guid)
                ambiguous_guids.add(guid)
            else:
                rows[guid] = row
    return rows


def build_demo_pair(selection, positive, generated_by_guid):
    if selection["source_positive_guid"] != positive["guid"]:
        raise RuntimeError(
            "Positive-demo mismatch: "
            f"{selection['source_positive_guid']} != {positive['guid']}"
        )

    selected_guid = selection["selected_guid"]
    selected_true_label = selection["selected_true_label"]
    if selected_guid == positive["guid"]:
        if selected_true_label != positive["label"]:
            raise RuntimeError(
                f"Fallback label mismatch for {selected_guid}: "
                f"{selected_true_label} != {positive['label']}"
            )
        return dict(positive), dict(positive), True

    cached = generated_by_guid.get(selection.get("selected_cache_key"))
    if cached is None:
        cached = generated_by_guid.get(selected_guid)
    if cached is None:
        raise KeyError(f"Selected generated row missing from its source chunk: {selected_guid}")
    if cached.get("source_positive_guid") != positive["guid"]:
        raise RuntimeError(f"Generated source mismatch for {selected_guid}")
    if cached.get("true_label") != selected_true_label:
        raise RuntimeError(
            f"Generated target mismatch for {selected_guid}: "
            f"{cached.get('true_label')} != {selected_true_label}"
        )

    negative_demo = core.self_refine_negative_row_from_cache(positive, cached)
    generated_icl_demo = {
        "guid": selected_guid,
        "title": cached.get("title", positive.get("title", "")),
        "comment": cached["comment"],
        "label": selected_true_label,
    }
    return negative_demo, generated_icl_demo, False


def build_exact_contexts(args):
    train_rows = core.load_kold_csv(args.train_file)
    test_rows = core.load_kold_csv(args.test_file)
    chunk_dirs = sorted(glob.glob(os.path.join(args.source_result_dir, "chunks", "chunk_*")))
    if not chunk_dirs:
        raise FileNotFoundError(f"No source chunks under {args.source_result_dir}")

    bundles = []
    accepted = 0
    fallback = 0
    cache_paths = []
    root_cache_paths = sorted(
        glob.glob(os.path.join(args.source_result_dir, "self_refine_negative_cache*.jsonl"))
    )
    root_generated_by_guid = None
    for chunk_dir in chunk_dirs:
        predictions_path = os.path.join(chunk_dir, "predictions.jsonl")
        cache_path = os.path.join(chunk_dir, "self_refine_negative_cache.jsonl")
        if not os.path.isfile(predictions_path):
            raise FileNotFoundError(f"Incomplete source chunk: {chunk_dir}")
        if os.path.isfile(cache_path):
            cache_paths.append(cache_path)
            generated_by_guid = load_chunk_cache(cache_path)
        elif root_cache_paths:
            if root_generated_by_guid is None:
                root_generated_by_guid = load_merged_cache(root_cache_paths)
                cache_paths.extend(root_cache_paths)
            generated_by_guid = root_generated_by_guid
        else:
            raise FileNotFoundError(f"No generated-example cache for source chunk: {chunk_dir}")

        for source in read_jsonl(predictions_path):
            test_idx = source["test_idx"]
            positives = [train_rows[index] for index in source["positive_demo_indices"]]
            selections = source.get("self_refine_selection", [])
            if len(positives) != len(selections):
                raise RuntimeError(
                    f"Demo/selection count mismatch at test_idx={test_idx}: "
                    f"{len(positives)} != {len(selections)}"
                )

            negatives = []
            generated_icl = []
            for selection, positive in zip(selections, positives):
                negative, generated, used_fallback = build_demo_pair(
                    selection, positive, generated_by_guid
                )
                negatives.append(negative)
                generated_icl.append(generated)
                if used_fallback:
                    fallback += 1
                else:
                    accepted += 1

            test_row = test_rows[test_idx]
            bundles.append(
                {
                    "test_idx": test_idx,
                    "positive": core.build_context(test_row, positives, args.use_title),
                    "negative": core.build_context(test_row, negatives, args.use_title),
                    "generated_icl": core.build_context(test_row, generated_icl, args.use_title),
                    "metadata": {
                        "test_idx": test_idx,
                        "guid": test_row["guid"],
                        "gold": test_row["label"],
                        "positive_demo_indices": source["positive_demo_indices"],
                        "positive_demo_guids": [row["guid"] for row in positives],
                        "positive_demo_labels": [row["label"] for row in positives],
                        "self_refine_selection": selections,
                    },
                }
            )

    bundles.sort(key=lambda row: row["test_idx"])
    actual = [row["test_idx"] for row in bundles]
    expected = list(range(len(test_rows)))
    if actual != expected:
        raise RuntimeError("Source chunks do not cover the full test split exactly once")

    if args.test_sample_offset < 0:
        raise ValueError("--test-sample-offset must be non-negative")
    if args.max_test_samples is not None and args.max_test_samples <= 0:
        raise ValueError("--max-test-samples must be positive")
    if args.test_sample_offset >= len(bundles):
        raise ValueError(
            f"--test-sample-offset {args.test_sample_offset} is outside "
            f"the {len(bundles)}-row test split"
        )
    end = (
        args.test_sample_offset + args.max_test_samples
        if args.max_test_samples is not None
        else len(bundles)
    )
    bundles = bundles[args.test_sample_offset:end]
    fallback = sum(
        selection["selected_guid"] == selection["source_positive_guid"]
        for row in bundles
        for selection in row["metadata"]["self_refine_selection"]
    )
    accepted = sum(len(row["metadata"]["self_refine_selection"]) for row in bundles) - fallback

    contexts = {
        key: [row[key] for row in bundles]
        for key in ("positive", "negative", "generated_icl")
    }
    metadata = [row["metadata"] for row in bundles]
    return contexts, metadata, cache_paths, accepted, fallback


def main():
    args = parse_args()
    label_rules = os.path.join(args.prompt_dir, "classification_label_rules.txt")
    core.configure_label_rules(label_rules)
    core.configure_self_refine_templates(args.prompt_dir, "fact_only", "critic")
    contexts, metadata, cache_paths, accepted, fallback = build_exact_contexts(args)

    summary = {
        "num_test_samples": len(metadata),
        "num_accepted_generated_demos": accepted,
        "num_fallback_original_demos": fallback,
        "accepted_generated_fraction": accepted / (accepted + fallback),
        "source_result_dir": os.path.abspath(args.source_result_dir),
        "source_cache_files": cache_paths,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    os.makedirs(args.proposed_output_dir, exist_ok=True)
    os.makedirs(args.generated_icl_output_dir, exist_ok=True)
    core.configure_workspace_cache(args.cache_dir)
    model_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        batch_size=args.batch_size,
        use_chat_template=args.use_chat_template,
        self_refine_generation_batch_size=16,
        self_refine_temperature=0.7,
        self_refine_top_p=0.8,
        self_refine_top_k=20,
        self_refine_min_p=0.0,
        self_refine_max_tokens=256,
        self_refine_thinking_max_tokens=3072,
        self_refine_repetition_penalty=1.0,
        self_refine_enable_thinking=False,
    )
    tokenizer, token_info = core.load_tokenizer(model_args)
    _, score_fn = core.build_model_functions(model_args, tokenizer)

    positive_scores = score_fn(contexts["positive"])
    negative_scores = score_fn(contexts["negative"])
    generated_scores = score_fn(contexts["generated_icl"])
    iccd_scores = core.combine_iccd_scores(positive_scores, negative_scores, args.alpha)

    golds = [row["gold"] for row in metadata]
    regular_predictions = [core.argmax_label(row) for row in positive_scores]
    iccd_predictions = [core.argmax_label(row) for row in iccd_scores]
    generated_predictions = [core.argmax_label(row) for row in generated_scores]
    proposed_metrics = {
        "regular_icl": core.evaluate(golds, regular_predictions),
        "iccd": core.evaluate(golds, iccd_predictions),
    }
    generated_metrics = {
        "generated_icl": core.evaluate(golds, generated_predictions)
    }

    proposed_predictions = {
        "regular_icl": regular_predictions,
        "iccd": iccd_predictions,
    }
    proposed_scores = {
        "scores_regular_icl.jsonl": positive_scores,
        "scores_negative_context.jsonl": negative_scores,
        "scores_iccd.jsonl": iccd_scores,
    }
    config = {
        **summary,
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "prompt_dir": args.prompt_dir,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "test_sample_offset": args.test_sample_offset,
        "max_test_samples": args.max_test_samples,
        "use_title": args.use_title,
        "use_chat_template": args.use_chat_template,
        "label_token_info": token_info,
        "exact_source_chunk_matching": True,
    }
    core.write_experiment_outputs(
        args.proposed_output_dir,
        config,
        proposed_metrics,
        proposed_scores,
        metadata,
        proposed_predictions,
        positive_scores,
        iccd_scores,
    )

    core.write_json(os.path.join(args.generated_icl_output_dir, "config.json"), config)
    core.write_json(os.path.join(args.generated_icl_output_dir, "metrics.json"), generated_metrics)
    core.write_jsonl(
        os.path.join(args.generated_icl_output_dir, "scores_generated_icl.jsonl"),
        generated_scores,
    )
    core.write_jsonl(
        os.path.join(args.generated_icl_output_dir, "predictions_generated_icl.jsonl"),
        [
            {**meta, "pred": pred, "scores": scores}
            for meta, pred, scores in zip(metadata, generated_predictions, generated_scores)
        ],
    )
    print(json.dumps({**proposed_metrics, **generated_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
