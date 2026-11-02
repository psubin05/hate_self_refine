#!/usr/bin/env python3
"""One-off: make the OLD run/iccd.py random_except_self negative-context cache
usable as an --self-refine-initial-cache-file for iccd_minimal.py.

Source: results/ablation_target_policy_random_except_self_qwen3_8b_test_full/
        self_refine_negative_cache_shard*.jsonl
        (loop=2, 4-gate, produced by run/iccd.py -- rows carry an 'initial'
         trace step and the right random_except_self target labels, but no
         'initial_fingerprint' field, so iccd_minimal.py's
         load_initial_seed_cache() skips every row.)

This computes the current iccd_minimal.py self_refine_initial_fingerprint()
for the random_except_self policy (no GPU, no model load) and writes a copy
of every shard with that field injected into each row, into

  results/ablation_target_policy_random_except_self_iter0_seed_qwen3_8b_test_full/

which run_random_except_self_3gate.sh (SEED_FROM_OLD=1) then points
--self-refine-initial-cache-file at.

Safety: the fingerprint payload is {label_rules, initial_template, model,
target_policy, temperature, top_p, top_k, min_p, max_tokens,
thinking_max_tokens, enable_thinking, repetition_penalty, use_chat_template,
use_title}. The source run recorded temperature=0.7 top_p=0.8 top_k=20
min_p=0.0 max_tokens=256 thinking_max_tokens=3072 enable_thinking=False
repetition_penalty=1.0 use_chat_template=True use_title=True model=Qwen/Qwen3-8B
-- identical to run_kold_test_full.sh's defaults -- and its shard
prompt_fingerprint (f070292f...) matches results/ablation_iterations_0_*/,
so label_rules and the initial template are unchanged too. If any of that
were off, the fingerprint would simply not match at load time and every row
would be regenerated (a slow no-op, never a silently wrong seed).
"""

import glob
import json
import os
import sys
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import iccd_minimal as m  # noqa: E402

SRC_DIR = os.path.join(
    PROJECT_ROOT, "results", "ablation_target_policy_random_except_self_qwen3_8b_test_full"
)
DST_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "ablation_target_policy_random_except_self_iter0_seed_qwen3_8b_test_full",
)
PROMPT_DIR = os.path.join(PROJECT_ROOT, "prompts", "topic_aligned_random")
LABEL_RULES_FILE = os.path.join(PROMPT_DIR, "classification_label_rules.txt")


def main():
    m.configure_label_rules(LABEL_RULES_FILE)
    m.configure_self_refine_templates(PROMPT_DIR)

    args = types.SimpleNamespace(
        model_name_or_path="Qwen/Qwen3-8B",
        self_refine_target_policy="random_except_self",
        self_refine_temperature=0.7,
        self_refine_top_p=0.8,
        self_refine_top_k=20,
        self_refine_min_p=0.0,
        self_refine_max_tokens=256,
        self_refine_thinking_max_tokens=3072,
        self_refine_enable_thinking=False,
        self_refine_repetition_penalty=1.0,
        use_chat_template=True,
        use_title=True,
    )
    fp = m.self_refine_initial_fingerprint(args)
    print(f"random_except_self initial_fingerprint = {fp}")

    shards = sorted(glob.glob(os.path.join(SRC_DIR, "self_refine_negative_cache_shard*.jsonl")))
    if not shards:
        sys.exit(f"no source shards under {SRC_DIR}")
    os.makedirs(DST_DIR, exist_ok=True)

    total_in = total_out = 0
    for shard in shards:
        out_path = os.path.join(DST_DIR, os.path.basename(shard))
        n_in = n_out = 0
        with open(shard, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                n_in += 1
                row = json.loads(line)
                if "cache_key" not in row:
                    continue
                if m.self_refine_trace_step(row.get("trace", []), "initial") is None:
                    continue
                if None in (row.get("source_positive_guid"), row.get("true_label")):
                    continue
                row["initial_fingerprint"] = fp
                json.dump(row, fout, ensure_ascii=False)
                fout.write("\n")
                n_out += 1
        total_in += n_in
        total_out += n_out
        print(f"  {os.path.basename(shard)}: {n_in} -> {n_out}")

    uniq = set()
    for shard in sorted(glob.glob(os.path.join(DST_DIR, "self_refine_negative_cache_shard*.jsonl"))):
        for line in open(shard, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                uniq.add((r["source_positive_guid"], r["true_label"]))
    print(f"wrote {total_out}/{total_in} rows to {DST_DIR}")
    print(f"unique (source_positive_guid, target_label) initial drafts: {len(uniq)}")


if __name__ == "__main__":
    main()
