#!/usr/bin/env python3
"""Evaluate accepted self-refined comments as ordinary, correctly labeled ICL demos.

This is the control requested for the paper: it reuses exactly the demonstration
texts selected by a completed self-refine ICCD run, but replaces each negative
context's intentionally kept source label with the generated comment's target
label.  Failed generations that fell back to the original positive demo remain
the original, correctly labeled demo.  The query inputs and the number/order of
demonstrations are unchanged.
"""

import argparse
import glob
import json
import os
from types import SimpleNamespace

import iccd_minimal as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(core.PROJECT_ROOT, ".cache", "generated_icl"),
    )
    parser.add_argument("--use-title", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-chat-template", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/rebuild all contexts without loading a model.",
    )
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_generated_rows(cache_paths):
    by_guid = {}
    for path in cache_paths:
        for row in read_jsonl(path):
            guid = row.get("guid")
            if not guid:
                continue
            previous = by_guid.get(guid)
            if previous is not None and (
                previous.get("comment") != row.get("comment")
                or previous.get("true_label") != row.get("true_label")
            ):
                raise RuntimeError(f"Conflicting cached rows for generated guid={guid}")
            by_guid[guid] = row
    return by_guid


def correctly_labeled_demo(selection, positive_row, generated_by_guid):
    selected_guid = selection["selected_guid"]
    selected_true_label = selection["selected_true_label"]
    if selection["source_positive_guid"] != positive_row["guid"]:
        raise RuntimeError(
            "Positive-demo mismatch: "
            f"{selection['source_positive_guid']} != {positive_row['guid']}"
        )

    if selected_guid == positive_row["guid"]:
        if selected_true_label != positive_row["label"]:
            raise RuntimeError(
                f"Fallback row {selected_guid} has target {selected_true_label}, "
                f"expected source label {positive_row['label']}"
            )
        return dict(positive_row)

    cached = generated_by_guid.get(selected_guid)
    if cached is None:
        raise KeyError(f"Selected generated row is missing from caches: {selected_guid}")
    if cached.get("true_label") != selected_true_label:
        raise RuntimeError(
            f"Target-label mismatch for {selected_guid}: "
            f"{cached.get('true_label')} != {selected_true_label}"
        )
    return {
        "guid": selected_guid,
        "title": cached.get("title", positive_row.get("title", "")),
        "comment": cached["comment"],
        # The only experimental change from the proposed ICCD negative context:
        # use the generated text's target label as the correct ICL label.
        "label": selected_true_label,
    }


def build_generated_icl_contexts(args):
    train_rows = core.load_kold_csv(args.train_file)
    test_rows = core.load_kold_csv(args.test_file)

    # Cache GUIDs are unique only within a source chunk: the same positive
    # demonstration can be sampled in multiple chunks and stochastic generation
    # may produce a different accepted comment in each.  Rebuild each test row
    # against the exact cache belonging to the chunk that produced its recorded
    # ICCD prediction, rather than flattening all shards by GUID.
    chunk_dirs = sorted(glob.glob(os.path.join(args.source_result_dir, "chunks", "chunk_*")))
    if not chunk_dirs:
        raise FileNotFoundError(f"No completed source chunks under {args.source_result_dir}")

    contexts = []
    metadata = []
    cache_paths = []
    accepted = 0
    fallback = 0
    for chunk_dir in chunk_dirs:
        predictions_path = os.path.join(chunk_dir, "predictions.jsonl")
        cache_path = os.path.join(chunk_dir, "self_refine_negative_cache.jsonl")
        if not os.path.isfile(predictions_path) or not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Incomplete source chunk: {chunk_dir}")
        cache_paths.append(cache_path)
        generated_by_guid = load_generated_rows([cache_path])
        for source in read_jsonl(predictions_path):
            test_idx = source["test_idx"]
            positive_indices = source["positive_demo_indices"]
            selections = source.get("self_refine_selection", [])
            if len(positive_indices) != len(selections):
                raise RuntimeError(
                    f"Demo/selection count mismatch at test_idx={test_idx}: "
                    f"{len(positive_indices)} != {len(selections)}"
                )
            positive_demos = [train_rows[index] for index in positive_indices]
            generated_demos = []
            for selection, positive in zip(selections, positive_demos):
                demo = correctly_labeled_demo(selection, positive, generated_by_guid)
                generated_demos.append(demo)
                if selection["selected_guid"] == positive["guid"]:
                    fallback += 1
                else:
                    accepted += 1
            contexts.append(
                core.build_context(test_rows[test_idx], generated_demos, args.use_title)
            )
            metadata.append(
                {
                    "test_idx": test_idx,
                    "guid": test_rows[test_idx]["guid"],
                    "gold": test_rows[test_idx]["label"],
                    "positive_demo_indices": positive_indices,
                    "generated_icl_demo_guids": [row["guid"] for row in generated_demos],
                    "generated_icl_demo_labels": [row["label"] for row in generated_demos],
                }
            )

    order = sorted(range(len(metadata)), key=lambda index: metadata[index]["test_idx"])
    contexts = [contexts[index] for index in order]
    metadata = [metadata[index] for index in order]
    expected_indices = list(range(len(metadata)))
    actual_indices = [row["test_idx"] for row in metadata]
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Source chunks do not cover the full test set exactly once: "
            f"first mismatch near {next((i for i, pair in enumerate(zip(actual_indices, expected_indices)) if pair[0] != pair[1]), 'end')}"
        )

    return contexts, metadata, cache_paths, accepted, fallback


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    label_rules = os.path.join(args.prompt_dir, "classification_label_rules.txt")
    core.configure_label_rules(label_rules)
    core.configure_self_refine_templates(args.prompt_dir)

    contexts, metadata, cache_paths, accepted, fallback = build_generated_icl_contexts(args)
    summary = {
        "num_test_samples": len(contexts),
        "num_accepted_generated_demos": accepted,
        "num_fallback_original_demos": fallback,
        "accepted_generated_fraction": accepted / (accepted + fallback),
        "source_result_dir": os.path.abspath(args.source_result_dir),
        "source_cache_files": cache_paths,
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    core.configure_workspace_cache(args.cache_dir)
    model_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        tensor_parallel_size=1,
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
    scores = score_fn(contexts)
    predictions = [core.argmax_label(row) for row in scores]
    golds = [row["gold"] for row in metadata]
    metrics = {"generated_icl": core.evaluate(golds, predictions)}

    core.write_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    core.write_jsonl(os.path.join(args.output_dir, "scores_generated_icl.jsonl"), scores)
    core.write_jsonl(
        os.path.join(args.output_dir, "predictions_generated_icl.jsonl"),
        [
            {
                **meta,
                "prediction_source": "generated_icl",
                "pred": predictions[index],
                "scores": scores[index],
            }
            for index, meta in enumerate(metadata)
        ],
    )
    core.write_json(
        os.path.join(args.output_dir, "config.json"),
        {
            **summary,
            "model_name_or_path": args.model_name_or_path,
            "train_file": args.train_file,
            "test_file": args.test_file,
            "prompt_dir": args.prompt_dir,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "use_title": args.use_title,
            "use_chat_template": args.use_chat_template,
            "label_token_info": token_info,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
