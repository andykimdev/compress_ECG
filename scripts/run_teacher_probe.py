#!/usr/bin/env python3
"""
Role B step 2: train only the Teacher's output layer on saved ECG-FM features.

Purpose:
- Fit the layer on the Training ECGs and pick its epoch on the Validation ECGs.
- With --holdout-fraction, fit it on the fit part of the Training ECGs and pick its epoch on
  the dev part, so it never sees a held-out patient and never reads Validation.

Special Notes:
- Install first: pip install -e .
- Outputs of a holdout carry its tag, such as checkpoints/teacher/probe_ho0.2_hs0.
- With --patient-fraction the layer is fitted on a share of the fit patients, and the outputs
  carry that tag too, such as probe_ho0.2_hs0_pf0.125_ps0.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compress_ecg.baseline import load_mini
from compress_ecg.data import LABEL_COLUMNS, SHD_INDEX, holdout_tag, open_group, subsample_tag
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


def main(argv: list[str] | None = None) -> None:
    """
    Fit the output layers and save them with their scores.

    Args:
        argv: Command line arguments without the program name. None reads them from sys.argv.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", help="Defaults to ECHONEXT_DIR, then data/echonext/1.1.1")
    parser.add_argument("--pretrained", default="checkpoints/pretrained/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt")
    parser.add_argument("--features-dir", default="runs/teacher_features", help="Patient-level, never commit")
    parser.add_argument("--heads-dir", default="checkpoints/teacher/probe",
                        help="Folder for the output layers. A holdout adds its tag to the name, as in probe_ho0.2_hs0")
    parser.add_argument("--results", default="results/teacher/probe.csv",
                        help="Aggregate scores only. A holdout adds its tag to the file name")
    parser.add_argument("--holdout-fraction", type=float, default=0.0,
                        help="Hold out this share of the Training patients, as run_teacher.py does. The layer is "
                             "then fitted on the other patients and its epoch is picked on the most recent ECG of "
                             "each held-out patient (dev), and Validation is not read. 0 is off")
    parser.add_argument("--holdout-seed", type=int, default=0, help="Seed of the draw of held-out patients")
    parser.add_argument("--patient-fraction", type=float, default=1.0,
                        help="Fit the layer on only this share of the patients that are not held out, as "
                             "run_teacher.py does for a learning curve. Needs --holdout-fraction. 1 is all of them")
    parser.add_argument("--patient-seed", type=int, default=0, help="Seed of the draw of kept patients")
    parser.add_argument("--limit", type=int, help="Use only this many ECGs per split, for a quick check")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--with-mini", action="store_true", help="Also score EchoNext-Mini on the same ECGs")
    args = parser.parse_args(argv)
    if not 0 <= args.holdout_fraction < 1:
        parser.error("--holdout-fraction must be at least 0 and below 1")
    if not 0 < args.patient_fraction <= 1:
        parser.error("--patient-fraction must be above 0 and at most 1")
    if args.patient_fraction != 1 and not args.holdout_fraction:
        parser.error("--patient-fraction below 1 needs --holdout-fraction above 0")
    holdout = holdout_tag(args.holdout_fraction, args.holdout_seed)
    # Every output depends on the patients the layer was fitted on, so it carries both tags.
    tag = holdout + subsample_tag(args.patient_fraction, args.patient_seed)
    # Feature files are named for the ECGs they hold. Only fit changes with the share of patients.
    feature_tags = {"fit": tag, "dev": holdout}
    # The ECGs that fit the layer, the ECGs that pick its epoch, and the name printed for those.
    fitted_on, picked_on, picked_name = ("fit", "dev", "dev") if tag else ("train", "val", "Validation")
    # The tag keeps a holdout's files apart from those of the probe on the whole split.
    heads_dir = Path(f"{args.heads_dir}{tag}")
    results = Path(args.results).with_name(f"{Path(args.results).stem}{tag}{Path(args.results).suffix}")

    loaded = {}
    teacher = lambda: loaded.setdefault("model", build_teacher(args.pretrained, device=args.device))
    suffix = f"_limit{args.limit}" if args.limit else ""
    datasets = {
        name: open_group(
            args.data_dir, name, limit=args.limit, holdout_fraction=args.holdout_fraction,
            holdout_seed=args.holdout_seed, patient_fraction=args.patient_fraction, patient_seed=args.patient_seed,
        )
        for name in (fitted_on, picked_on)
    }
    features = {
        name: load_or_extract(
            teacher, dataset, Path(args.features_dir) / f"{name}{feature_tags.get(name, '')}{suffix}.npz", args,
        )
        for name, dataset in datasets.items()
    }

    rows = []
    for variant, rescale in VARIANTS.items():
        for lr in args.learning_rates:
            head, history = fit_probe(features[fitted_on], features[picked_on], rescale=rescale, lr=lr)
            path = heads_dir / f"{variant}_lr{lr:g}{suffix}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), path)
            metrics = head_metrics(head, features[picked_on])
            rows.append({
                "model": f"probe_{variant}", "learning_rate": lr, "best_epoch": int(np.argmax(history)) + 1,
                "epochs_run": len(history), **{name: metrics[name]["auroc"] for name in LABEL_COLUMNS},
            })
            best_epoch = rows[-1]["best_epoch"]
            print(f"{variant:9s} lr {lr:g}: {picked_name} SHD AUROC {max(history):.4f} at epoch {best_epoch}")

    if args.with_mini:
        metrics = basic_metrics(predict(load_mini(device=args.device), datasets[picked_on], device=args.device))
        rows.append({"model": "echonext_mini", **{name: metrics[name]["auroc"] for name in LABEL_COLUMNS}})
        print(f"EchoNext-Mini: {picked_name} SHD AUROC {metrics[LABEL_COLUMNS[SHD_INDEX]]['auroc']:.4f}")

    table = pd.DataFrame(rows).round(4)
    if tag:
        # Says which ECGs picked each epoch and gave each score. Without a holdout the table is as before.
        table.insert(1, "picked_on", picked_on)
    if not args.limit:
        results.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(results, index=False)
    print(table[["model", "learning_rate", "best_epoch", LABEL_COLUMNS[SHD_INDEX]]].to_string(index=False))


if __name__ == "__main__":
    main()
