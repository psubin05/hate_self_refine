#!/usr/bin/env python3
"""Evaluate ICCD with identical positive/negative samples and flipped labels.

For every test item, the positive demonstrations and their order are read
directly from a completed source run.  The negative context uses the same
title/comment in the same slot, changing only the demonstration label with
the proposed method's ontology-minimal derangement:

    정상 -> 공격적, 공격적 -> 혐오, 혐오 -> 정상

This isolates label corruption from counterfactual text generation.
"""

import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import iccd_minimal as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--source-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
        default=os.path.join(core.PROJECT_ROOT, ".cache", "same_sample_label_flip"),
    )
    parser.add_argument("--use-title", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_exact_contexts(args):
    train_rows = core.load_kold_csv(args.train_file)
    test_rows = core.load_kold_csv(args.test_file)
    source_path = os.path.join(args.source_result_dir, "predictions.jsonl")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Missing source predictions: {source_path}")
    source_rows = read_jsonl(source_path)

    source_indices = [row.get("test_idx") for row in source_rows]
    if source_indices != list(range(len(test_rows))):
        raise RuntimeError(
            "Source predictions must cover the full test split exactly once in test_idx order"
        )

    bundles = []
    positive_index_plan = []
    for source in source_rows:
        test_idx = source["test_idx"]
        indices = source.get("positive_demo_indices")
        if not isinstance(indices, list) or not indices:
            raise RuntimeError(f"Missing positive_demo_indices at test_idx={test_idx}")
        positives = [train_rows[index] for index in indices]

        expected_guids = source.get("positive_demo_guids")
        actual_guids = [row["guid"] for row in positives]
        if expected_guids is not None and actual_guids != expected_guids:
            raise RuntimeError(f"Positive GUID/order mismatch at test_idx={test_idx}")
        expected_labels = source.get("positive_demo_labels")
        actual_labels = [row["label"] for row in positives]
        if expected_labels is not None and actual_labels != expected_labels:
            raise RuntimeError(f"Positive label/order mismatch at test_idx={test_idx}")

        flipped = [
            {
                **row,
                "label": core.choose_negative_target_label(row["label"]),
                "source_label": row["label"],
                "source_positive_guid": row["guid"],
            }
            for row in positives
        ]
        if any(pos["guid"] != neg["guid"] for pos, neg in zip(positives, flipped)):
            raise RuntimeError(f"Sample identity changed at test_idx={test_idx}")
        if any(pos["label"] == neg["label"] for pos, neg in zip(positives, flipped)):
            raise RuntimeError(f"A label was not flipped at test_idx={test_idx}")

        test_row = test_rows[test_idx]
        bundles.append(
            {
                "positive": core.build_context(test_row, positives, args.use_title),
                "negative": core.build_context(test_row, flipped, args.use_title),
                "metadata": {
                    "test_idx": test_idx,
                    "guid": test_row["guid"],
                    "gold": test_row["label"],
                    "negative_context_method": "same_sample_label_flip",
                    "positive_demo_indices": indices,
                    "positive_demo_guids": actual_guids,
                    "positive_demo_labels": actual_labels,
                    "negative_demo_guids": [row["guid"] for row in flipped],
                    "negative_demo_labels": [row["label"] for row in flipped],
                    "same_samples_same_order": True,
                },
            }
        )
        positive_index_plan.append(indices)

    if args.test_sample_offset < 0:
        raise ValueError("--test-sample-offset must be non-negative")
    if args.max_test_samples is not None and args.max_test_samples <= 0:
        raise ValueError("--max-test-samples must be positive")
    if args.test_sample_offset >= len(bundles):
        raise ValueError("--test-sample-offset is outside the test split")
    end = (
        args.test_sample_offset + args.max_test_samples
        if args.max_test_samples is not None
        else len(bundles)
    )
    bundles = bundles[args.test_sample_offset:end]

    plan_payload = json.dumps(positive_index_plan, separators=(",", ":"))
    summary = {
        "num_source_test_samples": len(source_rows),
        "num_active_test_samples": len(bundles),
        "total_shots": len(source_rows[0]["positive_demo_indices"]),
        "source_result_dir": os.path.abspath(args.source_result_dir),
        "source_predictions_file": os.path.abspath(source_path),
        "positive_demo_index_plan_sha256": hashlib.sha256(plan_payload.encode()).hexdigest(),
        "same_samples_same_order": True,
        "label_flip_mapping": dict(core.ONTOLOGY_MINIMAL_TARGET),
    }
    contexts = {
        "positive": [row["positive"] for row in bundles],
        "negative": [row["negative"] for row in bundles],
    }
    metadata = [row["metadata"] for row in bundles]
    return contexts, metadata, summary


def main():
    args = parse_args()
    label_rules = os.path.join(args.prompt_dir, "classification_label_rules.txt")
    core.configure_label_rules(label_rules)
    # This ablation performs scoring only. Loading the classification system
    # prompt directly keeps it compatible with the archived 9-shot prompt
    # directory, which intentionally need not contain newer generation/critic
    # templates that are irrelevant here.
    system_prompt = os.path.join(args.prompt_dir, "system_prompt.txt")
    with open(system_prompt, encoding="utf-8") as handle:
        core.SYSTEM_PROMPT = handle.read().strip()
    contexts, metadata, summary = build_exact_contexts(args)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    cache_root = core.configure_workspace_cache(args.cache_dir)
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
    iccd_scores = core.combine_iccd_scores(positive_scores, negative_scores, args.alpha)
    golds = [row["gold"] for row in metadata]
    predictions = {
        "regular_icl": [core.argmax_label(row) for row in positive_scores],
        "iccd": [core.argmax_label(row) for row in iccd_scores],
    }
    metrics = {name: core.evaluate(golds, pred) for name, pred in predictions.items()}
    output_scores = {
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
        "classification_label_rules_file": label_rules,
        "negative_context_method": "same_sample_label_flip",
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
        "cache_dir": cache_root,
        "label_token_info": token_info,
    }
    core.write_experiment_outputs(
        args.output_dir,
        config,
        metrics,
        output_scores,
        metadata,
        predictions,
        positive_scores,
        iccd_scores,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
