#!/usr/bin/env python3
"""Fine-tune a standard encoder baseline on the paper's 3-way KOLD task."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


LABELS = ("정상", "공격적", "혐오")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-file", default=str(PROJECT_ROOT / "data" / "kold" / "train.csv")
    )
    parser.add_argument(
        "--test-file", default=str(PROJECT_ROOT / "data" / "kold" / "test.csv")
    )
    parser.add_argument(
        "--validation-file", default=str(PROJECT_ROOT / "data" / "kold" / "valid.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kold_label(row):
    if str(row["OFF"]).strip().lower() != "true":
        return "정상"
    return "혐오" if str(row["TGT"]).strip().lower() == "group" else "공격적"


def format_input(row):
    title = "" if pd.isna(row["title"]) else str(row["title"]).strip()
    comment = "" if pd.isna(row["comment"]) else str(row["comment"]).strip()
    return f"제목: {title}\n댓글: {comment}"


def load_frame(path):
    frame = pd.read_csv(path)
    required = {"guid", "title", "comment", "OFF", "TGT"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    result = pd.DataFrame(
        {
            "guid": frame["guid"].astype(str),
            "text": frame.apply(format_input, axis=1),
            "label_name": frame.apply(kold_label, axis=1),
        }
    )
    result["label"] = result["label_name"].map(LABEL2ID)
    if result["label"].isna().any():
        raise ValueError(f"Unexpected labels in {path}")
    result["label"] = result["label"].astype(int)
    return result


class EncodedDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.guids = frame["guid"].tolist()
        self.labels = frame["label"].tolist()
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def make_loader(frame, tokenizer, args, shuffle):
    dataset = EncodedDataset(frame, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator if shuffle else None,
    )


def evaluate(model, loader, device, amp_dtype, description):
    model.eval()
    golds, preds = [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False):
            labels = batch.pop("labels")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch).logits
            golds.extend(labels.tolist())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return golds, preds


def metric_dict(golds, preds):
    report = classification_report(
        golds,
        preds,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(golds, preds),
        "macro_f1": f1_score(golds, preds, average="macro", zero_division=0),
        "classwise": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(
            golds, preds, labels=list(range(len(LABELS)))
        ).tolist(),
    }


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_all = load_frame(args.train_file)
    valid_frame = load_frame(args.validation_file)
    test_frame = load_frame(args.test_file)
    train_frame = train_all.reset_index(drop=True)
    valid_frame = valid_frame.reset_index(drop=True)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "num_train": len(train_frame),
        "num_validation": len(valid_frame),
        "num_test": len(test_frame),
        "train_distribution": train_frame["label_name"].value_counts().to_dict(),
        "validation_distribution": valid_frame["label_name"].value_counts().to_dict(),
        "test_distribution": test_frame["label_name"].value_counts().to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    train_loader = make_loader(train_frame, tokenizer, args, shuffle=True)
    valid_loader = make_loader(valid_frame, tokenizer, args, shuffle=False)
    test_loader = make_loader(test_frame, tokenizer, args, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    device = torch.device("cuda:0")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best_macro_f1 = -1.0
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        valid_golds, valid_preds = evaluate(
            model, valid_loader, device, amp_dtype, f"valid epoch {epoch}"
        )
        valid_metrics = metric_dict(valid_golds, valid_preds)
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "validation": valid_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False, indent=2), flush=True)

        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)

    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    test_golds, test_preds = evaluate(model, test_loader, device, amp_dtype, "test")
    test_metrics = metric_dict(test_golds, test_preds)

    result = {
        **summary,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "mixed_precision": "bf16" if amp_dtype == torch.bfloat16 else "fp16",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "test": test_metrics,
    }
    write_json(output_dir / "metrics.json", result)

    predictions = [
        {"guid": guid, "gold": ID2LABEL[gold], "pred": ID2LABEL[pred]}
        for guid, gold, pred in zip(test_frame["guid"], test_golds, test_preds)
    ]
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))
    print(f"metrics: {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
