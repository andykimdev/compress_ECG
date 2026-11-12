#!/usr/bin/env python3
"""Role B step 2: train only the Teacher's output layer on saved ECG-FM features. Install first: pip install -e ."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compress_ecg.baseline import load_mini
from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX, EchoNextDataset
from compress_ecg.predictions import basic_metrics, predict
from compress_ecg.teacher import build_teacher, extract_features, fit_probe

VARIANTS = {"rescaled": True, "raw": False}


def load_or_extract(teacher, dataset, path: Path, args) -> dict[str, np.ndarray]:
    """Read saved features, or run ECG-FM once and save them."""
    if path.is_file():
        with np.load(path) as saved:
            return dict(saved)
    features = extract_features(teacher(), dataset, batch_size=args.batch_size, device=args.device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **features)
    return features


def head_metrics(head, features: dict[str, np.ndarray]) -> dict:
    """Score a head on saved features with the shared quick-check metrics."""
    with torch.no_grad():
        logits = head(torch.from_numpy(features["features"]))
    labels = features["labels"].copy()
    labels[~features["label_mask"]] = np.nan
    return basic_metrics({
        "labels": labels, "label_mask": features["label_mask"],
        "probabilities": torch.sigmoid(logits).numpy(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--pretrained", default="checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--features-dir", default="runs/teacher_features", help="Patient-level, never commit")
    parser.add_argument("--heads-dir", default="checkpoints/teacher/probe")
    parser.add_argument("--results", default="results/teacher/probe.csv", help="Aggregate scores only")
    parser.add_argument("--limit", type=int, help="Use only this many ECGs per split, for a quick check")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--with-mini", action="store_true", help="Also score EchoNext-Mini on the same ECGs")
    args = parser.parse_args()

    loaded = {}
    teacher = lambda: loaded.setdefault("model", build_teacher(args.pretrained, device=args.device))
    suffix = f"_limit{args.limit}" if args.limit else ""
    datasets = {split: EchoNextDataset(args.data_dir, split, limit=args.limit) for split in ("train", "val")}
    features = {
        split: load_or_extract(teacher, dataset, Path(args.features_dir) / f"{split}{suffix}.npz", args)
        for split, dataset in datasets.items()
    }

    rows = []
    for variant, rescale in VARIANTS.items():
        for lr in args.learning_rates:
            head, history = fit_probe(features["train"], features["val"], rescale=rescale, lr=lr)
            path = Path(args.heads_dir) / f"{variant}_lr{lr:g}{suffix}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), path)
            metrics = head_metrics(head, features["val"])
            rows.append({
                "model": f"probe_{variant}", "learning_rate": lr, "best_epoch": int(np.argmax(history)) + 1,
                "epochs_run": len(history), **{name: metrics[name]["auroc"] for name in LABEL_COLUMNS},
            })
            print(f"{variant:9s} lr {lr:g}: Validation SHD AUROC {max(history):.4f} at epoch {rows[-1]['best_epoch']}")

    if args.with_mini:
        metrics = basic_metrics(predict(load_mini(device=args.device), datasets["val"], device=args.device))
        rows.append({"model": "echonext_mini", **{name: metrics[name]["auroc"] for name in LABEL_COLUMNS}})
        print(f"EchoNext-Mini: Validation SHD AUROC {metrics[LABEL_COLUMNS[SHD_INDEX]]['auroc']:.4f}")

    table = pd.DataFrame(rows).round(4)
    if not args.limit:
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.results, index=False)
    print(table[["model", "learning_rate", "best_epoch", LABEL_COLUMNS[SHD_INDEX]]].to_string(index=False))


if __name__ == "__main__":
    main()
