#!/usr/bin/env python3
"""Minimal, standalone reproduction of the paper's headline KOLD ICCD result.

Scope (deliberately narrower than run/iccd.py, which this file does not
import or modify):

  * The proposed negative-context method -- self-refine generation via the
    ``topic_aligned_random_revision`` pipeline, ``ontology_minimal`` target
    label policy (minimum-edit-cost fixed derangement
    정상->공격적->혐오->정상), the structured target-hidden OFF/TGT critic, and
    the OFF-field self-contradiction repair ("off_contradiction_fix") --
    always applied, not flag-gated.
  * The paper's three no-generation ``reverse_mapping`` baselines, which
    differ only in how positive demonstrations are retrieved:
    ``--retrieval-method random|bm25|topk``.
  * The target-label-policy ablation: ``--self-refine-target-policy
    random_except_self`` reproduces the pre-ablation "기존" configuration
    (uniformly random non-source target label) to isolate ontology_minimal's
    effect, holding the rest of the pipeline fixed.
  * A single model plays both generator and evaluator (no jailbroken /
    safety-ablated generator model; the paper found it unnecessary).

Everything else in run/iccd.py -- the hierarchical_axis,
unconditional_structural_revision, and programmatic_transition pipelines,
the legacy 5-criteria critic, cross-run seed-cache reuse, target-plan
replay, alpha real-fraction scaling, and multi-GPU chunk sharding -- is
left out. See run/iccd.py for that full experiment surface.

Default flags reproduce the proposed Qwen3-8B self-refinement configuration.
With ``--negative-context-method reverse_mapping``, the same entry point also
reproduces the random, BM25, and dense-retrieval baselines.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

LABELS = ("정상", "공격적", "혐오")
ONTOLOGY_TRANSITION_COSTS = {
    ("정상", "공격적"): 1,
    ("정상", "혐오"): 2,
    ("공격적", "정상"): 1,
    ("공격적", "혐오"): 1,
    ("혐오", "정상"): 1,
    ("혐오", "공격적"): 1,
}


def minimum_cost_label_derangement(labels, transition_costs):
    """Solve the balanced no-self target assignment from directed edit costs."""
    candidates = []
    for targets in permutations(labels):
        if any(source == target for source, target in zip(labels, targets)):
            continue
        mapping = dict(zip(labels, targets))
        cost = sum(
            transition_costs[(source, target)] for source, target in mapping.items()
        )
        candidates.append((cost, targets, mapping))
    if not candidates:
        raise RuntimeError("No label derangement is available.")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


ONTOLOGY_MINIMAL_TARGET = minimum_cost_label_derangement(
    LABELS, ONTOLOGY_TRANSITION_COSTS
)

TOPIC_ALIGNED_PROMPT_TEMPLATE_FILES = {
    "initial": "topic_aligned_initial.txt",
    "structural_feedback": "topic_aligned_feedback.txt",
    "structural_revision": "topic_aligned_revision.txt",
}
FACT_ONLY_FEEDBACK_TEMPLATE_FILE = "fact_only_feedback.txt"
FACT_ONLY_REVISION_TEMPLATE_FILE = "fact_only_revision.txt"
NO_FEEDBACK_REVISION_TEMPLATE_FILE = "no_feedback_revision.txt"

INPUT_WITH_TITLE_TEMPLATE = "제목: {title}\n댓글: {comment}"
INPUT_COMMENT_TEMPLATE = "댓글: {comment}"
DEMO_TEMPLATE = "{input_text}\n정답: {label}"
QUERY_TEMPLATE = "{input_text}\n정답: {output}"

# Bump this when prompt-independent generation/parsing logic below changes
# incompatibly with previously written self-refine caches.
SELF_REFINE_LOGIC_VERSION = 1

# Populated by configure_label_rules()/configure_self_refine_templates() in
# main() before any of the functions below that read them are called.
CLASSIFICATION_LABEL_RULES = None
CLASSIFICATION_INSTRUCTION = None
SELF_REFINE_TEMPLATES = None
SYSTEM_PROMPT = None
GENERATION_SYSTEM_PROMPT = None
PROMPT_FINGERPRINT = None


def load_text(directory, filename):
    with open(os.path.join(directory, filename), encoding="utf-8") as handle:
        return handle.read().strip()


def load_self_refine_templates(
    prompt_dir, critic_profile="full", revision_mode="critic"
):
    templates = {
        stage: load_text(prompt_dir, filename)
        for stage, filename in TOPIC_ALIGNED_PROMPT_TEMPLATE_FILES.items()
    }
    if critic_profile == "fact_only":
        templates["structural_feedback"] = load_text(
            prompt_dir, FACT_ONLY_FEEDBACK_TEMPLATE_FILE
        )
        templates["structural_revision"] = load_text(
            prompt_dir, FACT_ONLY_REVISION_TEMPLATE_FILE
        )
    if revision_mode == "direct":
        templates["structural_revision"] = load_text(
            prompt_dir, NO_FEEDBACK_REVISION_TEMPLATE_FILE
        )
    templates["system_prompt"] = load_text(prompt_dir, "system_prompt.txt")
    generation_system_prompt_path = os.path.join(
        prompt_dir, "generation_system_prompt.txt"
    )
    templates["generation_system_prompt"] = (
        load_text(prompt_dir, "generation_system_prompt.txt")
        if os.path.isfile(generation_system_prompt_path)
        else templates["system_prompt"]
    )
    return templates


def compute_prompt_fingerprint(label_rules, templates):
    parts = [
        label_rules,
        templates["system_prompt"],
        templates["generation_system_prompt"],
        templates["initial"],
        templates["structural_feedback"],
        templates["structural_revision"],
    ]
    return hashlib.sha1("\n\n".join(parts).encode("utf-8")).hexdigest()


def configure_label_rules(label_rules_file):
    global CLASSIFICATION_LABEL_RULES, CLASSIFICATION_INSTRUCTION
    with open(label_rules_file, encoding="utf-8") as handle:
        CLASSIFICATION_LABEL_RULES = handle.read().strip()
    instruction_template = load_text(
        os.path.join(PROJECT_ROOT, "prompts"), "classification_instruction.txt"
    )
    CLASSIFICATION_INSTRUCTION = instruction_template.format(
        label_rules=CLASSIFICATION_LABEL_RULES
    )


def configure_self_refine_templates(
    prompt_dir, critic_profile="full", revision_mode="critic"
):
    global SELF_REFINE_TEMPLATES, SYSTEM_PROMPT, GENERATION_SYSTEM_PROMPT
    global PROMPT_FINGERPRINT
    SELF_REFINE_TEMPLATES = load_self_refine_templates(
        prompt_dir, critic_profile, revision_mode
    )
    SYSTEM_PROMPT = SELF_REFINE_TEMPLATES["system_prompt"]
    GENERATION_SYSTEM_PROMPT = SELF_REFINE_TEMPLATES["generation_system_prompt"]
    PROMPT_FINGERPRINT = compute_prompt_fingerprint(
        CLASSIFICATION_LABEL_RULES, SELF_REFINE_TEMPLATES
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--model-name-or-path", default="Qwen/Qwen3-8B")
    add("--trust-remote-code", action="store_true")
    add("--train-file", default=os.path.join(PROJECT_ROOT, "data", "kold", "train.csv"))
    add("--test-file", default=os.path.join(PROJECT_ROOT, "data", "kold", "test.csv"))
    add(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results", "runs", "self_refine_minimal"),
    )
    add(
        "--prompt-dir",
        default=None,
        help="Defaults to <project-root>/prompts/topic_aligned_random.",
    )
    add(
        "--classification-label-rules-file",
        default=None,
        help="Defaults to <prompt-dir>/classification_label_rules.txt.",
    )
    add(
        "--negative-context-method",
        choices=("self_refine", "reverse_mapping"),
        default="self_refine",
        help=(
            "self_refine = proposed method (topic_aligned_random_revision + "
            "ontology_minimal). reverse_mapping = paper's no-generation "
            "baseline (real mislabeled train example); combine with "
            "--retrieval-method for the random/bm25/topk baseline rows."
        ),
    )
    add("--reverse-mapping-seed", type=int, default=66)
    add(
        "--retrieval-method",
        choices=("random", "topk", "bm25"),
        default="random",
        help="How POSITIVE demonstrations are retrieved per test row.",
    )
    add("--sentence-model-name-or-path", default="jhgan/ko-sroberta-multitask")
    add(
        "--self-refine-target-policy",
        choices=("ontology_minimal", "random_except_self"),
        default="ontology_minimal",
        help=(
            "ontology_minimal (proposed) = deterministic minimum-edit-cost "
            "derangement 정상->공격적->혐오->정상. random_except_self (ablation) "
            "= each negative demo's target drawn uniformly from the two "
            "non-source labels."
        ),
    )
    add(
        "--cache-dir",
        default=os.path.join(PROJECT_ROOT, ".cache", "kold_iccd_minimal"),
    )
    add("--seed", type=int, default=42)
    add("--shots-per-class", type=int, default=3)
    add("--alpha", type=float, default=1.0)
    add("--batch-size", type=int, default=16)
    add("--tensor-parallel-size", type=int, default=1)
    add("--gpu-memory-utilization", type=float, default=0.85)
    add("--max-model-len", type=int, default=8192)
    add("--max-num-batched-tokens", type=int, default=4096)
    add(
        "--test-sample-offset",
        type=int,
        default=0,
        help=(
            "Skip this many rows of --test-file before scoring. Used to "
            "split one test file across multiple single-GPU processes "
            "(data parallelism) -- see merge_chunks.py. Positive-demo "
            "retrieval RNG state is advanced through the skipped rows "
            "first (not reset at the offset), so each row's assigned "
            "demos match what a single unsplit run would have produced."
        ),
    )
    add(
        "--max-test-samples",
        type=int,
        default=None,
        help=(
            "Number of test rows to score, starting at --test-sample-offset. "
            "Omit to run to the end of the file."
        ),
    )
    add("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    add("--use-title", action=argparse.BooleanOptionalAction, default=True)
    add(
        "--self-refine-iterations",
        type=int,
        default=2,
        help=(
            "Revision budget per negative demo, on top of the always-run "
            "initial generation. 0 = critique the initial candidate once "
            "and fall back on failure with no revision attempt; N>=1 = up "
            "to N critique-revise-recheck rounds before falling back."
        ),
    )
    add(
        "--self-refine-ungated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip the critique/revision/fallback pipeline entirely: every "
            "negative demo uses whatever the initial generation produced, "
            "unconditionally, with no quality gate and no fallback to the "
            "original positive demo. Overrides --self-refine-iterations "
            "(no critique or revision rounds run at all). Intended to be "
            "combined with --self-refine-initial-cache-file pointed at an "
            "already-completed run's cache, so this reuses cached initial "
            "drafts and only re-scores -- no new generation."
        ),
    )
    add(
        "--self-refine-drop-gate-condition",
        default=None,
        help=(
            "Gate ablation: drop one or more of the four quality conditions "
            "from the accept/revise decision in topic_aligned_audit_passes "
            "(the critic prompt is unchanged and still emits all four keys; "
            "only the acceptance predicate ignores the named ones). Comma-"
            "separated for more than one, e.g. "
            "'context_compatible,fluent'. Choices: topic_aligned, "
            "context_compatible, fluent, unsupported_specific_fact. The "
            "structural checks (parse_complete, predicted_label match, "
            "generation_artifact) always stay on. Default None = all four "
            "conditions enforced."
        ),
    )
    add(
        "--self-refine-critic-profile",
        choices=("full", "fact_only"),
        default="full",
        help=(
            "Critic output schema and quality gate. full asks for and gates "
            "topic alignment, context compatibility, fluency, and factual "
            "support. fact_only asks for and gates only factual support, "
            "while retaining the target-hidden OFF/TGT structural checks."
        ),
    )
    add(
        "--self-refine-revision-mode",
        choices=("critic", "direct"),
        default="critic",
        help=(
            "critic runs feedback, conditional revision, and recheck. "
            "direct skips feedback and recheck, revises every initial draft "
            "once, and uses that revision directly for ICCD."
        ),
    )
    add("--self-refine-max-tokens", type=int, default=256)
    add("--self-refine-thinking-max-tokens", type=int, default=3072)
    add(
        "--self-refine-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    add(
        "--self-refine-temperature",
        type=float,
        default=None,
        help="Defaults to 0.7 (thinking off) or 0.6 (thinking on).",
    )
    add(
        "--self-refine-top-p",
        type=float,
        default=None,
        help="Defaults to 0.8 (thinking off) or 0.95 (thinking on).",
    )
    add("--self-refine-top-k", type=int, default=20)
    add("--self-refine-min-p", type=float, default=0.0)
    add("--self-refine-repetition-penalty", type=float, default=1.0)
    add("--self-refine-generation-batch-size", type=int, default=16)
    add("--self-refine-cache-file", default=None)
    add(
        "--self-refine-final-cache-file",
        action="append",
        help=(
            "Read-only cache of already-generated, already-quality-gated "
            "negative-context rows (e.g. another model's "
            "self_refine_negative_cache_shard*.jsonl) to reuse by "
            "(source_positive_guid, true_label) instead of regenerating. "
            "Only rows whose prompt_fingerprint matches this run's own "
            "templates/label-rules are reused. Repeat the flag for "
            "multiple shard files. Used for cross-model evaluator "
            "ablations that keep the negative-context GENERATOR fixed."
        ),
    )
    add(
        "--self-refine-initial-cache-file",
        action="append",
        help=(
            "Read-only cache to reuse just the INITIAL-generation step from "
            "(e.g. another --self-refine-iterations value's "
            "self_refine_negative_cache_shard*.jsonl), for rows this run "
            "doesn't already have a complete local or final-cache hit for. "
            "Only reused when the source row's initial_fingerprint matches "
            "this run's own model/target-policy/sampling-params/templates "
            "-- --self-refine-iterations itself is NOT part of that check, "
            "so an iterations sweep (0/1/2/3, ...) can share one run's "
            "initial drafts instead of every value regenerating them. "
            "Repeat the flag for multiple shard files."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.self_refine_max_tokens < 1:
        raise ValueError("--self-refine-max-tokens must be >= 1.")
    if not 0 < args.self_refine_top_p <= 1:
        raise ValueError("--self-refine-top-p must be in (0, 1].")
    if not 0 <= args.self_refine_min_p <= 1:
        raise ValueError("--self-refine-min-p must be in [0, 1].")
    if args.self_refine_top_k == 0 or args.self_refine_top_k < -1:
        raise ValueError("--self-refine-top-k must be -1 or a positive integer.")
    if args.self_refine_iterations < 0:
        raise ValueError(
            "--self-refine-iterations must be >= 0 (0 = critique the "
            "initial candidate once, no revision budget)."
        )
    if args.self_refine_revision_mode == "direct" and args.self_refine_iterations != 1:
        raise ValueError(
            "--self-refine-revision-mode direct requires "
            "--self-refine-iterations 1."
        )
    if args.test_sample_offset < 0:
        raise ValueError("--test-sample-offset must be >= 0.")
    if args.self_refine_drop_gate_condition is not None:
        valid = {
            "topic_aligned",
            "context_compatible",
            "fluent",
            "unsupported_specific_fact",
        }
        dropped = [
            c.strip()
            for c in args.self_refine_drop_gate_condition.split(",")
            if c.strip()
        ]
        invalid = sorted(set(dropped) - valid)
        if invalid or not dropped:
            raise ValueError(
                f"--self-refine-drop-gate-condition: invalid condition(s) "
                f"{invalid or dropped}; choices are {sorted(valid)}."
            )


def configure_workspace_cache(cache_dir):
    cache_root = os.path.abspath(cache_dir)
    cache_paths = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": os.path.join(cache_root, "huggingface"),
        "HF_HUB_CACHE": os.path.join(cache_root, "huggingface", "hub"),
        "HF_DATASETS_CACHE": os.path.join(cache_root, "huggingface", "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "huggingface", "transformers"),
        "TORCH_HOME": os.path.join(cache_root, "torch"),
        "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
        "NUMBA_CACHE_DIR": os.path.join(cache_root, "numba"),
        "CUDA_CACHE_PATH": os.path.join(cache_root, "cuda"),
        "VLLM_CACHE_ROOT": os.path.join(cache_root, "vllm"),
    }
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key, path in cache_paths.items():
        os.environ.setdefault(key, path)
        os.makedirs(os.environ[key], exist_ok=True)
    return cache_root


# ---------------------------------------------------------------------------
# Data loading and positive-demo retrieval
# ---------------------------------------------------------------------------


def kold_label(row):
    if str(row["OFF"]).lower() != "true":
        return LABELS[0]
    return LABELS[2] if row.get("TGT", "") == "group" else LABELS[1]


def load_kold_csv(path, max_samples=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "guid": row.get("guid", str(len(rows))),
                    "title": row.get("title", ""),
                    "comment": row["comment"],
                    "label": kold_label(row),
                    "OFF": row["OFF"],
                    "TGT": row.get("TGT", ""),
                    "GRP": row.get("GRP", ""),
                    "OFF_span": row.get("OFF_span", ""),
                    "TGT_span": row.get("TGT_span", ""),
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def index_by_label(rows):
    result = defaultdict(list)
    for index, row in enumerate(rows):
        result[row["label"]].append(index)
    return result


def sample_positive_demos(by_label, shots_per_class, rng):
    """Paper Sec 3 "Random" positive-demo retrieval."""
    indices = []
    for label in LABELS:
        indices.extend(rng.sample(by_label[label], shots_per_class))
    rng.shuffle(indices)
    return indices


def build_topk_positive_demo_indices(
    by_label, shots_per_class, train_embeddings, query_embedding
):
    """Paper Sec 3 "TopK" positive-demo retrieval: nearest neighbors by
    cosine similarity over --sentence-model-name-or-path embeddings."""
    import numpy as np

    indices = []
    for label in LABELS:
        embeddings = train_embeddings[label]
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms[norms == 0] = 1e-12
        sims = (embeddings @ query_embedding) / norms
        top_local = np.argsort(-sims)[:shots_per_class]
        indices.extend(by_label[label][i] for i in top_local)
    return indices


def build_bm25_positive_demo_indices(
    by_label, shots_per_class, bm25_by_label, tokenized_query
):
    """Paper Sec 3 "BM25" positive-demo retrieval: word-overlap similarity."""
    import numpy as np

    indices = []
    for label in LABELS:
        scores = bm25_by_label[label].get_scores(tokenized_query)
        top_local = np.argsort(-scores)[:shots_per_class]
        indices.extend(by_label[label][i] for i in top_local)
    return indices


# ---------------------------------------------------------------------------
# Prompt / context construction
# ---------------------------------------------------------------------------


def format_input(row, use_title):
    template = INPUT_WITH_TITLE_TEMPLATE if use_title else INPUT_COMMENT_TEMPLATE
    return template.format(title=row.get("title", ""), comment=row["comment"])


def format_demo(row, use_title, label=None):
    label = label or row["label"]
    return DEMO_TEMPLATE.format(input_text=format_input(row, use_title), label=label)


def build_context(test_row, demos=None, use_title=False):
    parts = [CLASSIFICATION_INSTRUCTION]
    parts.extend(format_demo(row, use_title) for row in demos or [])
    parts.append(
        QUERY_TEMPLATE.format(input_text=format_input(test_row, use_title), output="")
    )
    return "\n\n".join(parts)


def render_chat_prompt(tokenizer, prompt, use_chat_template, enable_thinking=False):
    if not use_chat_template:
        return prompt
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "The tokenizer must support apply_chat_template(..., "
            "enable_thinking=...). For Qwen3, install transformers>=4.51.0."
        ) from exc


def strip_think_block(text):
    text = str(text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def validate_negative_demo_source_labels(positive_demos, negative_demos):
    """Fail if an ICCD negative demonstration does not keep its source label."""
    if len(positive_demos) != len(negative_demos):
        raise RuntimeError(
            "ICCD invariant violation: positive and negative demonstration counts "
            f"differ ({len(positive_demos)} != {len(negative_demos)})."
        )
    for slot, (positive, negative) in enumerate(zip(positive_demos, negative_demos)):
        source_label = positive["label"]
        negative_label = negative.get("label")
        if negative_label != source_label:
            raise RuntimeError(
                "ICCD invariant violation: negative demonstration must preserve "
                f"the source label at slot {slot} "
                f"({negative_label!r} != {source_label!r})."
            )


def build_flat_contexts(test_row, positive_demos, negative_demos, use_title):
    if negative_demos:
        validate_negative_demo_source_labels(positive_demos, negative_demos)
    return {
        "positive": build_context(test_row, positive_demos, use_title),
        "negative": build_context(test_row, negative_demos, use_title),
    }


# ---------------------------------------------------------------------------
# reverse_mapping baseline: real mislabeled example, no generation
# (arXiv:2502.13738 Sec 2.2)
# ---------------------------------------------------------------------------


def build_reverse_mapping_negative_demo(positive_row, source_row):
    return {
        "guid": f"reverse_mapping:{source_row['guid']}",
        "title": source_row.get("title", ""),
        "comment": source_row["comment"],
        "label": positive_row["label"],
        "source_label": positive_row["label"],
        "true_label": source_row["label"],
        "source_positive_guid": positive_row["guid"],
    }


def build_reverse_mapping_negative_demos(positive_demos, train_rows, by_label, rng):
    negatives = []
    for positive in positive_demos:
        other_labels = [label for label in LABELS if label != positive["label"]]
        target_label = rng.choice(other_labels)
        source_row = train_rows[rng.choice(by_label[target_label])]
        negatives.append(build_reverse_mapping_negative_demo(positive, source_row))
    return negatives


def rebuild_flat_contexts_with_reverse_mapping(
    train_rows, test_rows, contexts, metadata, args
):
    rng = random.Random(args.reverse_mapping_seed)
    by_label = index_by_label(train_rows)
    negative_contexts = []
    for meta in metadata:
        test_row = test_rows[meta["test_idx"]]
        positive_demos = [train_rows[index] for index in meta["positive_demo_indices"]]
        negative_demos = build_reverse_mapping_negative_demos(
            positive_demos, train_rows, by_label, rng
        )
        row_contexts = build_flat_contexts(
            test_row, positive_demos, negative_demos, args.use_title
        )
        negative_contexts.append(row_contexts["negative"])
        meta["negative_context_method"] = "reverse_mapping"
        meta["negative_demo_guids"] = [row["guid"] for row in negative_demos]
        meta["negative_demo_kept_labels"] = [row["label"] for row in negative_demos]
        meta["negative_demo_true_labels"] = [
            row["true_label"] for row in negative_demos
        ]
    new_contexts = {key: list(value) for key, value in contexts.items()}
    new_contexts["negative"] = negative_contexts
    return new_contexts, metadata


def make_prompt_bundle(
    train_rows,
    test_rows,
    seed,
    shots_per_class,
    use_title,
    retrieval_method,
    train_embeddings=None,
    bm25_by_label=None,
    sentence_model=None,
):
    rng = random.Random(seed)
    by_label = index_by_label(train_rows)
    for label in LABELS:
        if len(by_label[label]) < shots_per_class:
            raise ValueError(f"Not enough train examples for label {label}")
    contexts = {"positive": [], "negative": []}
    metadata = []
    for test_index, test_row in enumerate(test_rows):
        if retrieval_method == "random":
            indices = sample_positive_demos(by_label, shots_per_class, rng)
        elif retrieval_method == "topk":
            query_embedding = sentence_model.encode(format_input(test_row, use_title))
            indices = build_topk_positive_demo_indices(
                by_label, shots_per_class, train_embeddings, query_embedding
            )
        elif retrieval_method == "bm25":
            tokenized_query = format_input(test_row, use_title).split()
            indices = build_bm25_positive_demo_indices(
                by_label, shots_per_class, bm25_by_label, tokenized_query
            )
        else:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")
        demos = [train_rows[index] for index in indices]
        row_contexts = build_flat_contexts(test_row, demos, [], use_title)
        for key in contexts:
            contexts[key].append(row_contexts[key])
        metadata.append(
            {
                "test_idx": test_index,
                "guid": test_row["guid"],
                "gold": test_row["label"],
                "positive_demo_indices": indices,
                "positive_demo_guids": [row["guid"] for row in demos],
                "positive_demo_labels": [row["label"] for row in demos],
            }
        )
    return contexts, metadata


# ---------------------------------------------------------------------------
# Self-refine negative-context generation (topic_aligned_random_revision)
# ---------------------------------------------------------------------------


def self_refine_cache_path(args):
    if args.self_refine_cache_file:
        return os.path.abspath(args.self_refine_cache_file)
    return os.path.join(args.output_dir, "self_refine_negative_cache.jsonl")


def load_self_refine_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "cache_key" not in row:
                skipped += 1
                continue
            cache[row["cache_key"]] = row
    if skipped:
        print(f"Skipped {skipped} invalid self-refine cache rows from {path}")
    return cache


def append_self_refine_cache(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def load_final_seed_cache(cache_files, current_fingerprint):
    """Load and filter external self-refine caches for cross-model reuse.

    A row is reusable only if it came from this same topic_aligned_random_
    revision pipeline and was produced under templates/label-rules that
    hash to this run's own PROMPT_FINGERPRINT -- which model generated it
    is deliberately NOT part of that check, so a fixed generator's cache
    can seed runs that swap in a different evaluator model.
    """
    seed_cache = {}
    for cache_file in cache_files or []:
        for row in load_self_refine_cache(cache_file).values():
            compatible = (
                row.get("topic_aligned_random_revision") is True
                and row.get("prompt_fingerprint") == current_fingerprint
            )
            if not compatible:
                continue
            key = (row.get("source_positive_guid"), row.get("true_label"))
            if None not in key:
                seed_cache[key] = row
    return seed_cache


def self_refine_initial_fingerprint(args):
    """Identity of the INITIAL-generation step only -- deliberately excludes
    --self-refine-iterations, since the initial draft (source, target label,
    prompt, sampling params) never depends on how many critique/revision
    rounds are allowed afterward. Used to let an --self-refine-iterations
    sweep (0/1/2/3, ...) share one run's initial drafts instead of every
    value regenerating them from scratch.
    """
    payload = {
        "label_rules": CLASSIFICATION_LABEL_RULES,
        "initial_template": SELF_REFINE_TEMPLATES["initial"],
        "model": args.model_name_or_path,
        "target_policy": args.self_refine_target_policy,
        "temperature": args.self_refine_temperature,
        "top_p": args.self_refine_top_p,
        "top_k": args.self_refine_top_k,
        "min_p": args.self_refine_min_p,
        "max_tokens": args.self_refine_max_tokens,
        "thinking_max_tokens": args.self_refine_thinking_max_tokens,
        "enable_thinking": args.self_refine_enable_thinking,
        "repetition_penalty": args.self_refine_repetition_penalty,
        "use_chat_template": args.use_chat_template,
        "use_title": args.use_title,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def load_initial_seed_cache(cache_files, current_initial_fingerprint):
    """Load external caches' "initial" trace steps for reuse, keyed by
    (source_positive_guid, true_label). Only reused when the source row's
    own initial_fingerprint matches this run's self_refine_initial_fingerprint
    -- i.e. same model/target-policy/sampling params/templates, whatever
    --self-refine-iterations value produced it.
    """
    seed_cache = {}
    for cache_file in cache_files or []:
        for row in load_self_refine_cache(cache_file).values():
            seed_fingerprint = row.get("initial_fingerprint")
            if seed_fingerprint is None or seed_fingerprint != current_initial_fingerprint:
                continue
            initial_step = self_refine_trace_step(row.get("trace", []), "initial")
            if initial_step is None:
                continue
            key = (row.get("source_positive_guid"), row.get("true_label"))
            if None not in key:
                seed_cache[key] = initial_step
    return seed_cache


def self_refine_cache_key(pos_row, target_label, args):
    payload = {
        "pos_guid": pos_row["guid"],
        "pos_label": pos_row["label"],
        "target_label": target_label,
        "target_policy": args.self_refine_target_policy,
        "iterations": args.self_refine_iterations,
        "revision_mode": args.self_refine_revision_mode,
        "ungated": args.self_refine_ungated,
        "use_title": args.use_title,
        "model": args.model_name_or_path,
        "prompt_fingerprint": PROMPT_FINGERPRINT,
        "logic_version": SELF_REFINE_LOGIC_VERSION,
        "temperature": args.self_refine_temperature,
        "top_p": args.self_refine_top_p,
        "top_k": args.self_refine_top_k,
        "min_p": args.self_refine_min_p,
        "max_tokens": args.self_refine_max_tokens,
        "thinking_max_tokens": args.self_refine_thinking_max_tokens,
        "enable_thinking": args.self_refine_enable_thinking,
        "repetition_penalty": args.self_refine_repetition_penalty,
    }
    # Only perturb the key for the leave-one-out gate ablation, so existing
    # (all-four-conditions) caches stay valid.
    if args.self_refine_drop_gate_condition is not None:
        payload["drop_gate_condition"] = args.self_refine_drop_gate_condition
    key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def self_refine_negative_row_from_cache(pos_row, cache_row):
    return {
        "guid": cache_row["guid"],
        "title": cache_row.get("title", pos_row.get("title", "")),
        "comment": cache_row["comment"],
        # ICCD corrupts the input-label mapping: replace the input with text
        # from another class while preserving the source demonstration label.
        "label": pos_row["label"],
        "source_label": pos_row["label"],
        "true_label": cache_row["true_label"],
        "source_positive_guid": pos_row["guid"],
    }


def choose_negative_target_label(pos_label):
    """ontology_minimal target policy: deterministic minimum-edit-cost
    derangement 정상->공격적->혐오->정상."""
    return ONTOLOGY_MINIMAL_TARGET[pos_label]


def choose_random_negative_target_label(pos_label, rng):
    """random_except_self target policy (the pre-ablation default): pick
    uniformly at random from the two non-source labels. Isolating this
    single change against ontology_minimal, holding everything else in the
    topic_aligned_random_revision pipeline fixed, is what showed the
    deterministic minimum-edit-cost transition -- not the extra revision
    round or the prompt/gate rewrite -- was the primary driver of ICCD's
    accuracy lift on this task."""
    return rng.choice([label for label in LABELS if label != pos_label])


KOLD_TARGET_TYPE_MAP = {
    "": "UNT",
    "untargeted": "UNT",
    "individual": "IND",
    "group": "GRP",
    "other": "OTH",
}


def normalize_annotated_span(value):
    value = str(value or "").strip()
    return value if value else "없음"


def structured_source_context(pos_row, use_title=True):
    """Return only source-grounded KOLD facts used by the editor.

    The target label is deliberately excluded so that source evidence and the
    requested counterfactual remain separate inputs.
    """
    raw_target = str(pos_row.get("TGT", "")).strip().lower()
    target_type = KOLD_TARGET_TYPE_MAP.get(raw_target, "UNT")
    is_offensive = str(pos_row.get("OFF", "")).strip().lower() == "true"
    if not is_offensive:
        target_type = "UNT"
    return {
        "title": pos_row.get("title", "") if use_title else "",
        "comment": pos_row["comment"],
        "source_label": pos_row["label"],
        "OFF": "OFF" if is_offensive else "NOT",
        "OFF_span": (
            normalize_annotated_span(pos_row.get("OFF_span")) if is_offensive else "없음"
        ),
        "TGT": target_type,
        "TGT_span": (
            normalize_annotated_span(pos_row.get("TGT_span")) if is_offensive else "없음"
        ),
        "GRP": normalize_annotated_span(pos_row.get("GRP")),
    }


def render_structured_source(pos_row, use_title=True):
    return json.dumps(
        structured_source_context(pos_row, use_title), ensure_ascii=False, indent=2
    )


def json_from_model_text(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    data = {}
    string_fields = ("text", "label")
    for key in string_fields:
        match = re.search(
            rf'"{key}"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"', text, flags=re.DOTALL
        )
        if match:
            raw_value = match.group("value")
            try:
                data[key] = json.loads(f'"{raw_value}"')
            except json.JSONDecodeError:
                data[key] = raw_value
    return data


def normalize_generated_label(label, fallback):
    label = str(label).strip()
    return label if label in LABELS else fallback


def normalize_generated_text(text, fallback):
    text = str(text or "").strip()
    for prefix in ("Text:", "문장:", "댓글:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if "댓글:" in text:
        text = text.split("댓글:", 1)[1].strip()
    if "\nlabel" in text:
        text = text.split("\nlabel", 1)[0].strip()
    return text.strip("\"' ") or fallback


def normalize_verifier_span(value):
    value = str(value or "").strip().strip("\"' ")
    return "없음" if not value or value.lower() in {"none", "null"} else value


def verifier_structure_label(verifier_result):
    """Map KOLD OFF/TGT decisions to the experiment's three labels.

    OFF/TGT are the decision variables. Spans are optional evidence and must
    not change a decision: KOLD contains offensive and group-targeted rows
    without annotated OFF_span/TGT_span values.
    """
    if not verifier_result.get("is_offensive", False):
        return LABELS[0]
    if verifier_result.get("target_type") == "GRP":
        return LABELS[2]
    return LABELS[1]


def contains_generation_artifact(text):
    lowered = str(text or "").lower()
    return any(
        marker in lowered for marker in ("<think>", "</think>", "```", "assistant:")
    )


# The critic model sometimes writes the classification label name (or a
# near-miss token) into the binary OFF field instead of the required
# "OFF"/"NOT" enum, e.g. {"OFF": "공격적"} or {"OFF": "ON"}. These are
# unambiguous restatements of the same judgment, not parse failures, so we
# normalize them before the strict enum check runs.
OFF_VALUE_SYNONYMS = {
    "공격적": "OFF",
    "혐오": "OFF",
    "OFFENSIVE": "OFF",
    "TRUE": "OFF",
    "ON": "OFF",
    "정상": "NOT",
    "NORMAL": "NOT",
    "FALSE": "NOT",
    "OFF": "OFF",
    "NOT": "NOT",
}


def normalize_topic_aligned_feedback(raw, candidate_text, critic_profile="full"):
    """Strictly parse a target-hidden structural and topical audit.

    Includes the OFF-field self-contradiction repair ("off_contradiction_fix"):
    the critic sometimes writes OFF="OFF" with no evidence span while its
    free-text feedback says nothing is offensive -- it appears to read "OFF"
    in the ordinary English sense ("inactive") rather than the KOLD label
    sense ("offensive"). This pattern was measured as 91% of parse failures
    on the 혐오->정상 transition. TGT is forced back to UNT whenever the
    (possibly repaired) OFF ends up NOT, since TGT is moot when not
    offensive. The mirror contradiction (NOT with a real, in-text span) is
    rare (<0.5% overall), so only this direction is corrected.
    """
    raw_text = str(raw or "").strip()
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    required = {
        "OFF",
        "OFF_span",
        "TGT",
        "TGT_span",
        "unsupported_specific_fact",
        "feedback",
    }
    boolean_keys = ["unsupported_specific_fact"]
    if critic_profile == "full":
        required.update({"topic_aligned", "context_compatible", "fluent"})
        boolean_keys.extend(["topic_aligned", "context_compatible", "fluent"])
    off_raw = str(data.get("OFF", "")).strip().upper()
    off = OFF_VALUE_SYNONYMS.get(off_raw, off_raw)
    target_type = str(data.get("TGT", "")).strip().upper()
    off_span = normalize_verifier_span(data.get("OFF_span"))
    target_span = normalize_verifier_span(data.get("TGT_span"))

    if off == "OFF" and off_span == "없음":
        off = "NOT"
    if off == "NOT":
        target_type = "UNT"
        target_span = "없음"

    booleans_valid = all(type(data.get(key)) is bool for key in boolean_keys)
    enum_valid = off in {"OFF", "NOT"} and target_type in {"UNT", "IND", "GRP", "OTH"}
    structure_valid = False
    if enum_valid:
        if off == "NOT":
            structure_valid = (
                off_span == "없음" and target_type == "UNT" and target_span == "없음"
            )
        else:
            structure_valid = (
                off_span != "없음"
                and off_span in candidate_text
                and (
                    (target_type == "UNT" and target_span == "없음")
                    or (
                        target_type != "UNT"
                        and target_span != "없음"
                        and target_span in candidate_text
                    )
                )
            )

    parse_complete = (
        required.issubset(data)
        and booleans_valid
        and enum_valid
        and structure_valid
        and not contains_generation_artifact(candidate_text)
    )
    result = {
        "parse_complete": parse_complete,
        "is_offensive": off == "OFF" if enum_valid else None,
        "offensive_span": off_span,
        "target_span": target_span,
        "target_type": target_type if enum_valid else None,
        "predicted_label": None,
        "topic_aligned": data.get("topic_aligned"),
        "context_compatible": data.get("context_compatible"),
        "fluent": data.get("fluent"),
        "unsupported_specific_fact": data.get("unsupported_specific_fact") is True,
        "feedback": str(data.get("feedback", "")).strip(),
        "generation_artifact": contains_generation_artifact(candidate_text),
    }
    if parse_complete:
        result["predicted_label"] = verifier_structure_label(result)
    return result


def topic_aligned_audit_passes(
    audit, target_label, drop_condition=None, critic_profile="full"
):
    """Accept/revise gate for a self-refine candidate.

    drop_condition (gate ablation) names one or more of the four quality
    conditions to skip in this predicate only -- the critic prompt is
    unchanged and the audit still carries every key. Pass a single name, a
    comma-separated string, or any iterable of names to drop more than one
    at once. The structural checks (parse_complete, predicted_label match,
    generation_artifact) are never droppable.
    """
    quality_checks = {
        "unsupported_specific_fact": audit.get("unsupported_specific_fact") is False,
    }
    if critic_profile == "full":
        quality_checks.update(
            {
                "topic_aligned": audit.get("topic_aligned") is True,
                "context_compatible": audit.get("context_compatible") is True,
                "fluent": audit.get("fluent") is True,
            }
        )
    if drop_condition:
        if isinstance(drop_condition, str):
            drop_condition = drop_condition.split(",")
        for cond in drop_condition:
            quality_checks.pop(cond.strip(), None)
    return (
        audit.get("parse_complete") is True
        and audit.get("predicted_label") == target_label
        and audit.get("generation_artifact") is False
        and all(quality_checks.values())
    )


def topic_aligned_target_guidance(target_label):
    """Target-conditional offense-injection guidance for free-form generation."""
    if target_label == LABELS[0]:
        return (
            "댓글에 욕설·비하·위협·집단에 대한 부정적 일반화가 전혀 없어야 "
            "한다. 제목과 원문 주제에 맞는 중립적인 의견이나 사실 진술만 "
            "작성하라."
        )
    if target_label == LABELS[1]:
        return (
            "구조화된 원문에 등장하는 실제 개인·기관·정책·사건·행동 중 "
            "하나를 향해 짧고 명백한 모욕 문장을 반드시 포함하라. "
            "'한심하다', '역겹다', '뻔뻔하다', '사기꾼이다', '쓰레기 "
            "같다'처럼 확실히 모욕적인 표현을 문법에 맞는 완전한 문장으로 "
            "사용하라. 예시 단어를 그대로 복사하지 말고 직접 자연스러운 "
            "문장을 만들어라. 성별·지역·종교·인종 등 집단 전체를 가리키는 "
            "말은 쓰지 마라."
        )
    return (
        "성별·인종·국적·종교·장애·성적 지향·정치 성향 등 공통 정체성을 "
        "공유하는 실제 집단 하나를 명시하고, 그 집단 전체를 향한 짧고 "
        "명백한 비하·모욕 문장을 반드시 포함하라. 개인이나 특정 사건이 "
        "아니라 집단 전체를 겨냥해야 한다."
    )


def build_self_refine_initial_prompt(pos_row, target_label, use_title):
    return SELF_REFINE_TEMPLATES["initial"].format(
        label_rules=CLASSIFICATION_LABEL_RULES,
        source_structure=render_structured_source(pos_row, use_title),
        target_label=target_label,
        target_label_display=target_label,
        target_guidance=topic_aligned_target_guidance(target_label),
    ).strip()


def build_topic_aligned_feedback_prompt(item, use_title):
    pos_row = item["pos_row"]
    return SELF_REFINE_TEMPLATES["structural_feedback"].format(
        label_rules=CLASSIFICATION_LABEL_RULES,
        source_structure=render_structured_source(pos_row, use_title),
        candidate_text=item["candidate_text"],
    ).strip()


def build_topic_aligned_revision_prompt(item, feedback, use_title):
    pos_row = item["pos_row"]
    target_label = item["target_label"]
    return SELF_REFINE_TEMPLATES["structural_revision"].format(
        label_rules=CLASSIFICATION_LABEL_RULES,
        source_structure=render_structured_source(pos_row, use_title),
        candidate_text=item["candidate_text"],
        feedback=feedback,
        target_label=target_label,
        target_label_display=target_label,
        target_guidance=topic_aligned_target_guidance(target_label),
    ).strip()


def self_refine_trace_step(trace, step_name):
    for item in trace:
        if item.get("step") == step_name:
            return item
    return None


def self_refine_last_trace_step_with_prefix(trace, prefix):
    for item in reversed(trace):
        if str(item.get("step", "")).startswith(prefix):
            return item
    return None


def self_refine_request_complete(trace):
    if self_refine_trace_step(trace, "initial") is None:
        return False
    return self_refine_trace_step(trace, "pipeline_done") is not None


def latest_self_refine_text_step(trace):
    for step in reversed(trace):
        step_name = step.get("step", "")
        if step_name == "initial" or step_name.startswith("revision_"):
            return step
    return {}


def make_active_self_refine_request(request_index, pos_row, target_label, cache_key, trace):
    latest = latest_self_refine_text_step(trace)
    return {
        "request_index": request_index,
        "pos_row": pos_row,
        "target_label": target_label,
        "cache_key": cache_key,
        "trace": trace,
        "has_initial": self_refine_trace_step(trace, "initial") is not None,
        "candidate_text": latest.get("text", ""),
        "candidate_label": latest.get("label", target_label),
    }


def run_feedback_critic_pass(
    items,
    step_name,
    use_title,
    generate_batch,
    drop_gate_condition=None,
    critic_profile="full",
):
    """Critique each item's current candidate under `step_name`; items that
    pass get final_quality_passed=True and a skipped final_<step_name> trace
    entry. Shared by the max_rounds=0 "critique once, no revision" path and
    each round's first feedback pass below -- both critique the current
    candidate the same way, differing only in which items they're called on.
    """
    prompts = [build_topic_aligned_feedback_prompt(item, use_title) for item in items]
    raws = (
        generate_batch(prompts, enable_thinking=False, deterministic=True)
        if prompts
        else []
    )
    for item, prompt, raw in zip(items, prompts, raws):
        audit = normalize_topic_aligned_feedback(
            raw, item["candidate_text"], critic_profile
        )
        passed = topic_aligned_audit_passes(
            audit, item["target_label"], drop_gate_condition, critic_profile
        )
        item["trace"].append(
            {
                "step": step_name,
                "source": "target_hidden_topic_critic",
                "prompt": prompt,
                "raw": raw,
                "source_label": item["pos_row"]["label"],
                "target_label": item["target_label"],
                **audit,
                "feedback": json.dumps(audit, ensure_ascii=False),
                "needs_revision": not passed,
            }
        )
        item["final_quality_passed"] = passed
        if passed:
            item["candidate_label"] = audit["predicted_label"]
            item["trace"].append(
                {
                    "step": f"final_{step_name}",
                    "text": item["candidate_text"],
                    "label": item["candidate_label"],
                    "skipped_critic_match": True,
                    "quality_passed": True,
                }
            )


def run_topic_aligned_random_revision_pipeline(
    active,
    generate_batch,
    use_title,
    max_rounds=1,
    drop_gate_condition=None,
    critic_profile="full",
):
    """Generate a topical counterfactual, then critique/revise/recheck for up
    to max_rounds cycles before giving up on a candidate (falls back to the
    original positive demo -- see rebuild_flat_contexts_with_self_refine).

    max_rounds=0 is a controlled "no revision budget" condition for an
    iterations sweep: the initial candidate is critiqued exactly once, with
    the same critic and the same pass/fail gate as every other max_rounds
    value, and falls back on failure without ever attempting a revision.
    This isolates revision's own marginal value (0 vs 1 vs 2 vs 3 ... budget)
    instead of comparing against an ungated "accept whatever the model first
    wrote" baseline.
    """
    max_rounds = max(0, max_rounds)

    for item in active:
        if "final_quality_passed" in item:
            continue
        for step in reversed(item["trace"]):
            if str(step.get("step", "")).startswith("final_feedback_"):
                item["final_quality_passed"] = bool(step.get("quality_passed", False))
                if step.get("label"):
                    item["candidate_label"] = step["label"]
                break

    remaining = [item for item in active if not item.get("final_quality_passed", False)]

    if max_rounds == 0:
        feedback_needed = [
            item for item in remaining
            if self_refine_trace_step(item["trace"], "feedback_0") is None
        ]
        run_feedback_critic_pass(
            feedback_needed,
            "feedback_0",
            use_title,
            generate_batch,
            drop_gate_condition,
            critic_profile,
        )
        for item in active:
            item.setdefault("final_quality_passed", False)
            if self_refine_trace_step(item["trace"], "pipeline_done") is None:
                item["trace"].append({"step": "pipeline_done"})
        return

    for round_index in range(max_rounds):
        if not remaining:
            break
        feedback_key = f"feedback_{round_index}"
        revision_key = f"revision_{round_index}"
        final_key = f"final_feedback_{round_index}"
        previous_text_key = "initial" if round_index == 0 else f"revision_{round_index - 1}"

        feedback_needed = [
            item for item in remaining
            if self_refine_trace_step(item["trace"], feedback_key) is None
        ]
        for item in feedback_needed:
            if contains_generation_artifact(item["candidate_text"]):
                previous = self_refine_trace_step(item["trace"], previous_text_key)
                if previous is not None:
                    previous["generation_artifact"] = True
                item["candidate_text"] = ""

        run_feedback_critic_pass(
            feedback_needed,
            feedback_key,
            use_title,
            generate_batch,
            drop_gate_condition,
            critic_profile,
        )

        revision_needed = [
            item for item in remaining
            if self_refine_trace_step(item["trace"], final_key) is None
        ]
        revision_prompts = [
            build_topic_aligned_revision_prompt(
                item,
                self_refine_trace_step(item["trace"], feedback_key).get("feedback", ""),
                use_title,
            )
            for item in revision_needed
        ]
        revision_raws = generate_batch(revision_prompts) if revision_prompts else []
        for item, prompt, raw in zip(revision_needed, revision_prompts, revision_raws):
            data = json_from_model_text(raw)
            item["candidate_text"] = normalize_generated_text(data.get("text", raw), "")
            item["candidate_label"] = normalize_generated_label(
                data.get("label"), item["target_label"]
            )
            item["trace"].append(
                {
                    "step": revision_key,
                    "prompt": prompt,
                    "raw": raw,
                    "text": item["candidate_text"],
                    "label": item["candidate_label"],
                    "generation_artifact": contains_generation_artifact(
                        item["candidate_text"]
                    ),
                }
            )

        final_prompts = [
            build_topic_aligned_feedback_prompt(item, use_title) for item in revision_needed
        ]
        final_raws = (
            generate_batch(final_prompts, enable_thinking=False, deterministic=True)
            if final_prompts
            else []
        )
        for item, prompt, raw in zip(revision_needed, final_prompts, final_raws):
            audit = normalize_topic_aligned_feedback(
                raw, item["candidate_text"], critic_profile
            )
            passed = topic_aligned_audit_passes(
                audit, item["target_label"], drop_gate_condition, critic_profile
            )
            item["final_quality_passed"] = passed
            if audit.get("predicted_label") is not None:
                item["candidate_label"] = audit["predicted_label"]
            item["trace"].append(
                {
                    "step": final_key,
                    "source": "target_hidden_topic_critic",
                    "prompt": prompt,
                    "raw": raw,
                    "source_label": item["pos_row"]["label"],
                    "target_label": item["target_label"],
                    **audit,
                    "quality_passed": passed,
                }
            )

        remaining = [item for item in remaining if not item.get("final_quality_passed", False)]

    for item in active:
        item.setdefault("final_quality_passed", False)
        if self_refine_trace_step(item["trace"], "pipeline_done") is None:
            item["trace"].append({"step": "pipeline_done"})


def run_direct_revision_pipeline(active, generate_batch, use_title):
    """Revise every initial draft once without critique, gate, or recheck."""
    revision_items = [
        item
        for item in active
        if self_refine_trace_step(item["trace"], "revision_0") is None
    ]
    prompts = [
        build_topic_aligned_revision_prompt(item, "", use_title)
        for item in revision_items
    ]
    raws = generate_batch(prompts) if prompts else []
    for item, prompt, raw in zip(revision_items, prompts, raws):
        data = json_from_model_text(raw)
        item["candidate_text"] = normalize_generated_text(data.get("text", raw), "")
        item["candidate_label"] = normalize_generated_label(
            data.get("label"), item["target_label"]
        )
        item["trace"].append(
            {
                "step": "revision_0",
                "source": "direct_revision_without_feedback",
                "prompt": prompt,
                "raw": raw,
                "text": item["candidate_text"],
                "label": item["candidate_label"],
                "generation_artifact": contains_generation_artifact(
                    item["candidate_text"]
                ),
            }
        )

    for item in active:
        item["final_quality_passed"] = True
        if self_refine_trace_step(item["trace"], "pipeline_done") is None:
            item["trace"].append({"step": "pipeline_done"})


def generate_self_refined_negative_rows_batch(
    requests, generate_batch, args, cache, cache_path, final_seed_cache, initial_seed_cache
):
    if not requests:
        return []

    results = [None] * len(requests)
    active = []
    for request_index, (pos_row, target_label) in enumerate(requests):
        # Cross-model ablation support: reuse a negative-context comment
        # already generated (and passed its quality gate) by a *different*
        # model's run against the same prompts, instead of regenerating it
        # with this run's --model-name-or-path. This is how the paper's
        # "생성: Qwen3-8B 고정, 평가: 모델별 교차" rows were produced -- the
        # evaluator model only classifies with the frozen Qwen3-8B-authored
        # text; it never has to write offensive/hateful content itself.
        final_seed = final_seed_cache.get((pos_row["guid"], target_label))
        if final_seed is not None:
            results[request_index] = (
                self_refine_negative_row_from_cache(pos_row, final_seed),
                final_seed,
            )
            continue
        cache_key = self_refine_cache_key(pos_row, target_label, args)
        cached = cache.get(cache_key)
        trace = list(cached["trace"]) if cached else []
        if not trace:
            # --self-refine-iterations sweep support: reuse another run's
            # already-generated INITIAL draft (iterations-independent) so
            # only the critique/revision rounds run fresh here.
            initial_step = initial_seed_cache.get((pos_row["guid"], target_label))
            if initial_step is not None:
                trace = [dict(initial_step)]
        if cached and self_refine_request_complete(trace):
            results[request_index] = (
                self_refine_negative_row_from_cache(pos_row, cached),
                cached,
            )
            continue
        active.append(
            make_active_self_refine_request(request_index, pos_row, target_label, cache_key, trace)
        )

    if not active:
        return results

    fresh = [item for item in active if not item["has_initial"]]
    if fresh:
        initial_prompts = [
            build_self_refine_initial_prompt(item["pos_row"], item["target_label"], args.use_title)
            for item in fresh
        ]
        initial_raws = generate_batch(initial_prompts)
        for item, initial_prompt, initial_raw in zip(fresh, initial_prompts, initial_raws):
            initial_data = json_from_model_text(initial_raw)
            candidate_text = normalize_generated_text(initial_data.get("text", initial_raw), "")
            candidate_label = normalize_generated_label(
                initial_data.get("label"), item["target_label"]
            )
            item["candidate_text"] = candidate_text
            item["candidate_label"] = candidate_label
            item["has_initial"] = True
            item["trace"].append(
                {
                    "step": "initial",
                    "prompt": initial_prompt,
                    "raw": initial_raw,
                    "text": candidate_text,
                    "label": candidate_label,
                }
            )

    if args.self_refine_ungated:
        for item in active:
            item["final_quality_passed"] = True
            item["trace"].append(
                {
                    "step": "final_feedback_0",
                    "text": item["candidate_text"],
                    "label": item["candidate_label"],
                    "skipped_critic_match": True,
                    "quality_passed": True,
                    "ungated": True,
                }
            )
            item["trace"].append({"step": "pipeline_done"})
    elif args.self_refine_revision_mode == "direct":
        run_direct_revision_pipeline(active, generate_batch, args.use_title)
    else:
        run_topic_aligned_random_revision_pipeline(
            active,
            generate_batch,
            args.use_title,
            max_rounds=args.self_refine_iterations,
            drop_gate_condition=args.self_refine_drop_gate_condition,
            critic_profile=args.self_refine_critic_profile,
        )

    for item in active:
        pos_row = item["pos_row"]
        final_audit = (
            self_refine_last_trace_step_with_prefix(item["trace"], "final_feedback_") or {}
        )
        cache_row = {
            "cache_key": item["cache_key"],
            "guid": f"self_refine:{pos_row['guid']}:{item['target_label']}",
            "source_positive_guid": pos_row["guid"],
            "source_positive_label": pos_row["label"],
            "true_label": item["target_label"],
            "kept_label": item["candidate_label"],
            "topic_aligned_random_revision": True,
            "prompt_fingerprint": PROMPT_FINGERPRINT,
            "initial_fingerprint": self_refine_initial_fingerprint(args),
            "title": pos_row.get("title", ""),
            "comment": item["candidate_text"],
            "trace": item["trace"],
            "final_quality_passed": bool(
                item.get(
                    "final_quality_passed",
                    final_audit.get("quality_passed", False),
                )
            ),
            "final_predicted_label": (
                item["candidate_label"]
                if args.self_refine_revision_mode == "direct"
                else final_audit.get("predicted_label")
            ),
        }
        append_self_refine_cache(cache_path, cache_row)
        cache[item["cache_key"]] = cache_row
        results[item["request_index"]] = (
            self_refine_negative_row_from_cache(pos_row, cache_row),
            cache_row,
        )
    return results


def log_self_refine_quality_summary(metadata):
    """Print how often negative-context generation actually produced a
    validated counterfactual vs. fell back to the unmodified positive demo.

    A high fallback rate means ICCD's negative context is, for that fraction
    of demos, just a copy of the positive example carrying no contrastive
    signal.
    """
    totals = Counter()
    passed = Counter()
    for meta in metadata:
        for detail in meta.get("self_refine_selection", []):
            target_label = detail.get("selected_true_label")
            totals[target_label] += 1
            if detail.get("selection_target_valid"):
                passed[target_label] += 1
    if not totals:
        return
    print("Self-refine negative-context quality (validated / total by target label):")
    for target_label in sorted(totals):
        total = totals[target_label]
        ok = passed[target_label]
        print(f"  {target_label}: {ok}/{total} ({ok / total:.1%})")
    grand_total = sum(totals.values())
    grand_passed = sum(passed.values())
    print(f"  overall: {grand_passed}/{grand_total} ({grand_passed / grand_total:.1%})")


def rebuild_flat_contexts_with_self_refine(train_rows, test_rows, contexts, metadata, generate_text, args):
    cache_path = self_refine_cache_path(args)
    cache = load_self_refine_cache(cache_path)
    final_seed_cache = load_final_seed_cache(
        args.self_refine_final_cache_file, PROMPT_FINGERPRINT
    )
    initial_seed_cache = load_initial_seed_cache(
        args.self_refine_initial_cache_file, self_refine_initial_fingerprint(args)
    )
    # Single RNG advanced sequentially across the whole (single-process) run,
    # only consulted under --self-refine-target-policy random_except_self.
    target_rng = random.Random(args.seed)

    new_contexts = {key: list(value) for key, value in contexts.items()}
    negative_contexts = []
    for meta in metadata:
        test_row = test_rows[meta["test_idx"]]
        positive_demos = [train_rows[index] for index in meta["positive_demo_indices"]]
        if args.self_refine_target_policy == "random_except_self":
            target_labels = [
                choose_random_negative_target_label(row["label"], target_rng)
                for row in positive_demos
            ]
        else:
            target_labels = [choose_negative_target_label(row["label"]) for row in positive_demos]
        requests = list(zip(positive_demos, target_labels))
        generated = generate_self_refined_negative_rows_batch(
            requests, generate_text, args, cache, cache_path, final_seed_cache, initial_seed_cache
        )
        if len(generated) != len(positive_demos):
            raise RuntimeError("Generated candidate count does not match demos.")

        negative_demos = []
        selection_details = []
        for positive, generated_candidate in zip(positive_demos, generated):
            negative_row, cache_row = generated_candidate
            target_valid = bool(cache_row.get("final_quality_passed", False))
            if target_valid:
                selected_row = negative_row
                fallback = "topic_random_final_quality_passed"
            else:
                selected_row = {
                    **positive,
                    "source_label": positive["label"],
                    "true_label": positive["label"],
                    "source_positive_guid": positive["guid"],
                }
                fallback = "original_positive_failed_topic_quality_gate"
            negative_demos.append(selected_row)
            selection_details.append(
                {
                    "source_positive_guid": positive["guid"],
                    "source_positive_label": positive["label"],
                    "selected_guid": selected_row["guid"],
                    "selected_true_label": selected_row["true_label"],
                    "selection_target_valid": target_valid,
                    "selection_fallback": fallback,
                }
            )

        row_contexts = build_flat_contexts(test_row, positive_demos, negative_demos, args.use_title)
        negative_contexts.append(row_contexts["negative"])
        meta["negative_context_method"] = "topic_aligned_random_revision"
        meta["negative_demo_guids"] = [row["guid"] for row in negative_demos]
        meta["negative_demo_kept_labels"] = [row["label"] for row in negative_demos]
        meta["negative_demo_true_labels"] = [row["true_label"] for row in negative_demos]
        meta["self_refine_selection"] = selection_details

    new_contexts["negative"] = negative_contexts
    log_self_refine_quality_summary(metadata)
    return new_contexts, metadata


# ---------------------------------------------------------------------------
# Label scoring and metrics
# ---------------------------------------------------------------------------


def get_logprob_value(value):
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict):
        return float(value["logprob"])
    return float(value)


def score_contexts(model, tokenizer, contexts, batch_size, use_chat_template):
    from vllm import SamplingParams

    # Korean labels are multi-token, so we sum logprobs across all tokens to
    # get log P(label | context) = sum_i log P(token_i | context, token_0..i-1).
    label_token_ids = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
    }

    sampling_params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    scores = []

    for start in range(0, len(contexts), batch_size):
        batch_contexts = contexts[start : start + batch_size]
        expanded_prompts = []
        expanded_label_token_ids = []
        for context in batch_contexts:
            rendered = render_chat_prompt(tokenizer, context, use_chat_template)
            for label in LABELS:
                expanded_prompts.append(rendered + label)
                expanded_label_token_ids.append(label_token_ids[label])

        outputs = model.generate(expanded_prompts, sampling_params=sampling_params)

        for offset in range(0, len(outputs), len(LABELS)):
            row_scores = {}
            for label_pos, label in enumerate(LABELS):
                output = outputs[offset + label_pos]
                tok_ids = expanded_label_token_ids[offset + label_pos]
                n_toks = len(tok_ids)
                logprob_sum = 0.0
                for i, tok_id in enumerate(tok_ids):
                    pos = -(n_toks - i)
                    token_logprobs_at_pos = output.prompt_logprobs[pos]
                    if tok_id not in token_logprobs_at_pos:
                        decoded = tokenizer.decode([tok_id])
                        raise KeyError(
                            f"Missing prompt logprob for label {label} "
                            f"token {tok_id!r} ({decoded!r}) at position {pos}"
                        )
                    logprob_sum += get_logprob_value(token_logprobs_at_pos[tok_id])
                row_scores[label] = logprob_sum
            scores.append(row_scores)

    return scores


def argmax_label(scores):
    return max(scores, key=scores.get)


def evaluate(golds, preds):
    total = len(golds)
    accuracy = sum(g == p for g, p in zip(golds, preds)) / total
    per_class = {}
    f1s = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in golds),
        }
        f1s.append(f1)
    return {"accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s), "classwise": per_class}


def combine_iccd_scores(positive_scores, negative_scores, alpha):
    """Eq. 2 of arXiv:2502.13738: softmax(z + alpha * (z - z^-))."""
    combined = []
    for pos, neg in zip(positive_scores, negative_scores):
        combined.append({label: pos[label] + alpha * (pos[label] - neg[label]) for label in LABELS})
    return combined


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def build_combined_prediction_rows(metadata, predictions):
    rows = []
    for idx, meta in enumerate(metadata):
        rows.append(
            {
                **meta,
                "pred_regular_icl": predictions["regular_icl"][idx],
                "pred_iccd": predictions["iccd"][idx],
            }
        )
    return rows


def build_single_prediction_rows(metadata, predictions, scores, source):
    return [
        {
            **meta,
            "prediction_source": source,
            "pred": predictions[source][idx],
            "scores": scores[idx],
        }
        for idx, meta in enumerate(metadata)
    ]


# ---------------------------------------------------------------------------
# Model runtime and self-refine text generation
# ---------------------------------------------------------------------------


def make_vllm_text_generator(model, tokenizer, args):
    from vllm import SamplingParams

    generation_batch_size = args.self_refine_generation_batch_size
    sampling_params = SamplingParams(
        temperature=args.self_refine_temperature,
        top_p=args.self_refine_top_p,
        top_k=args.self_refine_top_k,
        min_p=args.self_refine_min_p,
        max_tokens=args.self_refine_max_tokens,
        repetition_penalty=args.self_refine_repetition_penalty,
    )
    thinking_sampling_params = SamplingParams(
        temperature=args.self_refine_temperature,
        top_p=args.self_refine_top_p,
        top_k=args.self_refine_top_k,
        min_p=args.self_refine_min_p,
        max_tokens=args.self_refine_thinking_max_tokens,
        repetition_penalty=args.self_refine_repetition_penalty,
    )
    deterministic_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=args.self_refine_max_tokens,
        repetition_penalty=1.0,
    )

    def generate_text_batch(prompts, enable_thinking=None, deterministic=False):
        effective_thinking = (
            args.self_refine_enable_thinking if enable_thinking is None else enable_thinking
        )
        params = (
            deterministic_sampling_params
            if deterministic
            else (thinking_sampling_params if effective_thinking else sampling_params)
        )
        outputs = []
        for start in range(0, len(prompts), generation_batch_size):
            batch_prompts = prompts[start : start + generation_batch_size]
            rendered = [
                render_chat_prompt(
                    tokenizer, prompt, args.use_chat_template, enable_thinking=effective_thinking
                )
                for prompt in batch_prompts
            ]
            batch_outputs = model.generate(rendered, sampling_params=params)
            outputs.extend(strip_think_block(output.outputs[0].text) for output in batch_outputs)
        return outputs

    return generate_text_batch


def validate_label_tokens(tokenizer):
    info = {}
    for label in LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        info[label] = {
            "output_label": label,
            "token_ids": token_ids,
            "decoded": [tokenizer.decode([token_id]) for token_id in token_ids],
            "single_token": len(token_ids) == 1,
        }
    return info


def load_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_info = validate_label_tokens(tokenizer)
    print(json.dumps(token_info, ensure_ascii=False, indent=2))
    return tokenizer, token_info


def build_model_functions(args, tokenizer):
    from vllm import LLM

    llm_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": getattr(args, "pipeline_parallel_size", 1),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": 1,
        "enforce_eager": True,
        "max_model_len": args.max_model_len,
        "trust_remote_code": getattr(args, "trust_remote_code", False),
    }
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    model = LLM(args.model_name_or_path, **llm_kwargs)
    text_generate_fn = make_vllm_text_generator(model, tokenizer, args)

    def score_fn(contexts):
        return score_contexts(model, tokenizer, contexts, args.batch_size, args.use_chat_template)

    return text_generate_fn, score_fn


def score_experiment(contexts, score_fn, args):
    positive_scores = score_fn(contexts["positive"])
    negative_scores = score_fn(contexts["negative"])
    iccd_scores = combine_iccd_scores(positive_scores, negative_scores, args.alpha)
    output_scores = {
        "scores_regular_icl.jsonl": positive_scores,
        "scores_negative_context.jsonl": negative_scores,
        "scores_iccd.jsonl": iccd_scores,
    }
    return positive_scores, iccd_scores, output_scores


def write_experiment_outputs(
    output_dir, run_config, metrics, output_scores, metadata, predictions, positive_scores, iccd_scores
):
    write_json(os.path.join(output_dir, "config.json"), run_config)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    for filename, scores in output_scores.items():
        write_jsonl(os.path.join(output_dir, filename), scores)
    write_jsonl(
        os.path.join(output_dir, "predictions.jsonl"),
        build_combined_prediction_rows(metadata, predictions),
    )
    write_jsonl(
        os.path.join(output_dir, "predictions_regular_icl.jsonl"),
        build_single_prediction_rows(metadata, predictions, positive_scores, "regular_icl"),
    )
    write_jsonl(
        os.path.join(output_dir, "predictions_iccd.jsonl"),
        build_single_prediction_rows(metadata, predictions, iccd_scores, "iccd"),
    )


def build_run_config(args, cache_root, train_rows, golds, token_info, prompt_dir, label_rules_file):
    config = {
        "model_name_or_path": args.model_name_or_path,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "classification_label_rules_file": label_rules_file,
        "self_refine_prompt_dir": prompt_dir,
        "self_refine_pipeline": "topic_aligned_random_revision",
        "self_refine_target_policy": args.self_refine_target_policy,
        "cache_dir": cache_root,
        "seed": args.seed,
        "shots_per_class": args.shots_per_class,
        "total_shots": args.shots_per_class * len(LABELS),
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "backend": "vllm",
        "labels": list(LABELS),
        "negative_context_method": args.negative_context_method,
        "reverse_mapping_seed": args.reverse_mapping_seed,
        "retrieval_method": args.retrieval_method,
        "sentence_model_name_or_path": (
            args.sentence_model_name_or_path if args.retrieval_method == "topk" else None
        ),
        "self_refine_iterations": args.self_refine_iterations,
        "self_refine_ungated": args.self_refine_ungated,
        "self_refine_drop_gate_condition": args.self_refine_drop_gate_condition,
        "self_refine_critic_profile": args.self_refine_critic_profile,
        "self_refine_revision_mode": args.self_refine_revision_mode,
        "self_refine_max_tokens": args.self_refine_max_tokens,
        "self_refine_thinking_max_tokens": args.self_refine_thinking_max_tokens,
        "self_refine_enable_thinking": args.self_refine_enable_thinking,
        "self_refine_temperature": args.self_refine_temperature,
        "self_refine_top_p": args.self_refine_top_p,
        "self_refine_top_k": args.self_refine_top_k,
        "self_refine_min_p": args.self_refine_min_p,
        "self_refine_repetition_penalty": args.self_refine_repetition_penalty,
        "self_refine_generation_batch_size": args.self_refine_generation_batch_size,
        "self_refine_cache_file": self_refine_cache_path(args),
        "self_refine_final_cache_file": args.self_refine_final_cache_file,
        "self_refine_initial_cache_file": args.self_refine_initial_cache_file,
        "max_model_len": args.max_model_len,
        "use_chat_template": args.use_chat_template,
        "use_title": args.use_title,
        "ontology_target_mapping": dict(ONTOLOGY_MINIMAL_TARGET),
        "ontology_transition_costs": {
            f"{source}->{target}": cost
            for (source, target), cost in ONTOLOGY_TRANSITION_COSTS.items()
        },
        "label_token_info": token_info,
        "train_label_distribution": dict(Counter(row["label"] for row in train_rows)),
        "test_label_distribution": dict(Counter(golds)),
    }
    return config


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if args.self_refine_temperature is None:
        args.self_refine_temperature = 0.6 if args.self_refine_enable_thinking else 0.7
    if args.self_refine_top_p is None:
        args.self_refine_top_p = 0.95 if args.self_refine_enable_thinking else 0.8
    validate_args(args)

    prompt_dir = args.prompt_dir or os.path.join(
        PROJECT_ROOT, "prompts", "topic_aligned_random"
    )
    label_rules_file = args.classification_label_rules_file or os.path.join(
        prompt_dir, "classification_label_rules.txt"
    )
    configure_label_rules(label_rules_file)
    configure_self_refine_templates(
        prompt_dir,
        args.self_refine_critic_profile,
        args.self_refine_revision_mode,
    )

    cache_root = configure_workspace_cache(args.cache_dir)

    train_rows = load_kold_csv(args.train_file)
    load_limit = (
        args.test_sample_offset + args.max_test_samples
        if args.max_test_samples is not None
        else None
    )
    test_rows = load_kold_csv(args.test_file, max_samples=load_limit)
    if args.test_sample_offset >= len(test_rows):
        raise ValueError(
            f"--test-sample-offset {args.test_sample_offset} is outside "
            f"the loaded test set ({len(test_rows)} rows)."
        )
    active_test_rows = test_rows[args.test_sample_offset :]
    golds = [row["label"] for row in active_test_rows]
    os.makedirs(args.output_dir, exist_ok=True)

    train_embeddings = None
    bm25_by_label = None
    sentence_model = None
    if args.retrieval_method == "topk":
        from sentence_transformers import SentenceTransformer

        sentence_model = SentenceTransformer(args.sentence_model_name_or_path)
        by_label_for_index = index_by_label(train_rows)
        train_embeddings = {
            label: sentence_model.encode(
                [format_input(train_rows[i], args.use_title) for i in by_label_for_index[label]]
            )
            for label in LABELS
        }
    elif args.retrieval_method == "bm25":
        from rank_bm25 import BM25Okapi

        by_label_for_index = index_by_label(train_rows)
        bm25_by_label = {
            label: BM25Okapi(
                [format_input(train_rows[i], args.use_title).split() for i in by_label_for_index[label]]
            )
            for label in LABELS
        }

    contexts, metadata = make_prompt_bundle(
        train_rows=train_rows,
        test_rows=test_rows,
        seed=args.seed,
        shots_per_class=args.shots_per_class,
        use_title=args.use_title,
        retrieval_method=args.retrieval_method,
        train_embeddings=train_embeddings,
        bm25_by_label=bm25_by_label,
        sentence_model=sentence_model,
    )
    if args.test_sample_offset:
        # test_rows itself stays unsliced (full 0..load_limit prefix):
        # metadata's "test_idx" fields index into it directly, and
        # rebuild_flat_contexts_with_* look test rows up that way. Only
        # contexts/metadata are trimmed down to the active range, after
        # make_prompt_bundle's positive-demo RNG has already been advanced
        # through the skipped rows in the same deterministic order a single
        # unsplit run would use.
        contexts = {
            key: values[args.test_sample_offset :] for key, values in contexts.items()
        }
        metadata = metadata[args.test_sample_offset :]

    tokenizer, token_info = load_tokenizer(args)
    text_generate_fn, score_fn = build_model_functions(args, tokenizer)

    if args.negative_context_method == "reverse_mapping":
        contexts, metadata = rebuild_flat_contexts_with_reverse_mapping(
            train_rows, test_rows, contexts, metadata, args
        )
    else:
        contexts, metadata = rebuild_flat_contexts_with_self_refine(
            train_rows, test_rows, contexts, metadata, text_generate_fn, args
        )

    positive_scores, iccd_scores, output_scores = score_experiment(contexts, score_fn, args)

    predictions = {
        "regular_icl": [argmax_label(scores) for scores in positive_scores],
        "iccd": [argmax_label(scores) for scores in iccd_scores],
    }
    metrics = {name: evaluate(golds, preds) for name, preds in predictions.items()}

    run_config = build_run_config(
        args, cache_root, train_rows, golds, token_info, prompt_dir, label_rules_file
    )

    write_experiment_outputs(
        args.output_dir,
        run_config,
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
