# Teacher guide

This guide is for teammates who use the Teacher. It covers where the current Teacher is, how to
get it, how to use its outputs, and how it was built.

## The current Teacher

The Teacher is ECG-FM with a new output layer, fine-tuned on the EchoNext Training ECGs. The
current one is the average of 6 fine-tuning runs, stored as one model.

| Item | Value |
|---|---|
| Name | `soup6_lr1e-05_hlr0.0001_do0.5_tab_ema0.999_ev4_byshd` |
| Parameters | 90,383,340 |
| File size | 361,594,815 bytes |
| SHD AUROC on Validation | 85.15 |
| Mean AUROC of the 12 outputs on Validation | 83.62 |

The file [`results/teacher/best_teacher.json`](../results/teacher/best_teacher.json) records the
name, the SHA-256 hash, the source runs, and the scores. If a better Teacher replaces this one,
that file changes with it, so read the name from there.

Two limits apply to the scores.

- **Picked on Validation:** Each run kept its best checkpoint by Validation score, so the scores
  above run slightly high.
- **No Test score yet:** The Test ECGs are scored once, at the end of the project.

## Get the Teacher file

The file is too large for git, so it lives in a private Hugging Face repository,
[compress-ecg/teacher](https://huggingface.co/compress-ecg/teacher).

1. Sign the EchoNext data use agreement on PhysioNet, if you have not yet.
2. Create a Hugging Face account and accept the invitation to the `compress-ecg` organization.
3. Create a read token under your Hugging Face account settings.
4. Log in and download the file from the repository root.

```bash
pip install -U huggingface_hub
hf auth login
TEACHER=checkpoints/teacher/runs/soup6_lr1e-05_hlr0.0001_do0.5_tab_ema0.999_ev4_byshd
hf download compress-ecg/teacher best.pt --local-dir $TEACHER
```

5. Check the file against its hash. Use `sha256sum` on Linux.

```bash
shasum -a 256 $TEACHER/best.pt
```

The output should start with `1db5e09c` and match the `sha256` value in the record file.

Keep the file private. Share it only with team members who signed the data use agreement.

## Use the Teacher's outputs

Most roles never load the Teacher. They read its saved predictions, which follow the shared run
format in the [README](../README.md#the-two-shared-formats).

Run the Teacher once on the Training and Validation ECGs. This needs the Teacher environment
from the README's Teacher setup, the ECG-FM weights from `scripts/download_weights.sh`, and a
GPU.

```bash
python scripts/run_teacher.py --device cuda --checkpoint $TEACHER/best.pt \
  --run-id soup6_lr1e-05_hlr0.0001_do0.5_tab_ema0.999_ev4_byshd --score-splits train val
```

This writes `runs/teacher/train/` and `runs/teacher/val/`. Scoring the 72,475 Training ECGs took
about 2 minutes on one GPU. Role C then trains the Student from the Training run.

```bash
python scripts/run_student.py --teacher-run runs/teacher/train
```

The run files hold patient-level values. Keep them on the machine that holds the data, and never
commit them.

## Load the Teacher in Python

Use this only when you need the model itself, such as for profiling.

```python
from compress_ecg.teacher import load_frozen_teacher

teacher = load_frozen_teacher(
    "checkpoints/teacher/runs/soup6_lr1e-05_hlr0.0001_do0.5_tab_ema0.999_ev4_byshd/best.pt",
    device="cuda",
)
logits = teacher(waveform, tabular)
# waveform: float32 [batch, 12, 2500]
# tabular:  float32 [batch, 7]
# logits:   [batch, 12], with overall SHD last
```

The function also reads the ECG-FM checkpoint, which supplies the model layout. The returned
model is in evaluation mode with its weights frozen.

## How the Teacher was built

Each step below added to the one before it. Every row uses seeds 42 to 47. Scores are the average
SHD AUROC on Validation over those seeds, and the spread is their standard deviation. Step 0
comes from the [probe table](../results/teacher/probe.csv) and the rest from the
[run table](../results/teacher/runs.csv).

| Step | Change | Seeds | SHD AUROC | Spread |
|---|---|---|---|---|
| 0 | Keep ECG-FM fixed and train only a new output layer | 6 | 82.22 | 0.05 |
| 1 | Fine-tune ECG-FM with learning rates 1e-5 and 1e-4 | 6 | 83.26 | 0.49 |
| 2 | Add dropout of 0.5 before the output layer | 6 | 84.51 | 0.22 |
| 3 | Add the 7 tabular numbers as inputs | 6 | 84.68 | 0.21 |
| 4 | Average the weights during training, score 4 times per epoch, and pick by SHD | 6 | 85.07 | 0.08 |
| 5 | Average the weights of the 6 runs from step 4 into one model | 1 model | 85.15 | None |

Read the small steps with care.

- **Step 3:** The gain of 0.17 points is smaller than the spread of either row, so it is not
  distinguishable from noise.
- **Step 5:** A second model averaged from seeds 48 to 53 scored 85.17, recorded in
  [`soup_check_s48to53.json`](../results/teacher/soup_check_s48to53.json). Both averaged models
  sit about 0.08 points above the average of their own 6 runs. Averaging 12 runs scored 85.16.
- **GPU models:** Three of the runs in steps 2 and 3 ran on an A100 and the rest on an RTX PRO
  6000, so small gaps between those rows are less certain than the table suggests.
- **Step 0 setting:** The row is the rescaled output layer at learning rate 0.001. The probe table
  holds 6 settings times 6 seeds and comes from
  `python scripts/run_teacher_probe.py --seeds 42 43 44 45 46 47 --with-mini`.

Lead masking and patient balancing were also tried and added nothing on top of dropout.

### Scores, speed, and memory

- **Each output:** [`best_teacher_val.csv`](../results/teacher/best_teacher_val.csv) lists AUROC
  and Brier score for all 12 outputs of the current Teacher on Validation.
- **Speed and memory:** [`best_teacher_speed.json`](../results/teacher/best_teacher_speed.json)
  holds the forward pass time and peak GPU memory on one RTX PRO 6000. At batch size 64 in
  32-bit precision, the Teacher takes 1.18 ms per ECG, which is 844 ECGs per second, with a peak
  of 1.91 GB. `scripts/time_teacher.py` makes the file. Disk reading is not timed.

### Rebuild it

All commands run from the repository root, in the Teacher environment.

1. Fit the starting output layer on frozen ECG-FM features. Fine-tuning starts from the seed 42
   layer, and the other seeds fill the probe table.

```bash
python scripts/run_teacher_probe.py --device cuda --seeds 42 43 44 45 46 47 --with-mini
```

2. Fine-tune 6 seeds with the settings in `configs/teacher_ema.json`.

```bash
python scripts/sweep_teacher.py --config configs/teacher_ema.json
```

3. Average the 6 runs into one checkpoint.

```bash
RUN=lr1e-05_hlr0.0001_do0.5_tab_ema0.999_ev4_byshd
python scripts/soup_teacher.py --output checkpoints/teacher/runs/soup6_$RUN/best.pt \
  --checkpoints checkpoints/teacher/runs/${RUN}_s4[2-7]/best.pt
```

4. Score the averaged model on Validation.

```bash
python scripts/run_teacher.py --device cuda --run-id soup6_$RUN --score-splits val \
  --checkpoint checkpoints/teacher/runs/soup6_$RUN/best.pt
```

5. Rebuild the run table.

```bash
python scripts/collect_runs.py
```

## Training options

These options of `scripts/run_teacher.py` were added while tuning. Run it with `--help` for the
full list.

| Option | What it does |
|---|---|
| `--head-dropout` | Sets the dropout before the output layer |
| `--use-tabular` | Adds the 7 tabular numbers to the outputs through a small extra layer |
| `--ema-decay` | Scores and saves a running average of the weights in place of the trained weights |
| `--evals-per-epoch` | Sets how many times per epoch Validation is scored |
| `--select-by` | Chooses the score that picks the checkpoint, the 12-output mean or SHD |
| `--holdout-fraction` | Holds out a share of Training patients, so pilot runs never read Validation |
| `--patient-fraction` | Trains on a share of the remaining patients, for learning curves |

A run's name lists the options that were on. For example, `do0.5` is a dropout of 0.5, `tab` is
the tabular inputs, and `s42` is seed 42.
