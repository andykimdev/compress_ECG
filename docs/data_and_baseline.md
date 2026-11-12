# Role A: data, baselines, and profiling

A's first target is a repeatable run on **1,000 validation ECGs**. The full run scores both
baselines on all Validation and Test ECGs. Mini uses the released weights. Gradient boosting
fits the seven supplied tabular features on Training ECGs. B/C use the loader, and E receives
the aligned prediction files. Profiling uses synthetic inputs and runs independently of the data.

## Run on Kaggle

1. Push the code branch with `git push -u origin shaurya/role-a-baselines`. The notebook
   clones this branch in its first cell. Set `CODE_COMMIT` to the pushed SHA for exact repeats.
2. Upload/open `notebooks/01_data_and_baseline.ipynb` in a **private** Kaggle notebook.
   Attach the existing private EchoNext dataset. Keep the original filenames and order.
3. Enable Internet for the code and Mini weights. GitHub requires a token
   that can read the private `andykimdev/compress_ECG` repository. In Kaggle **Add-ons → Secrets**,
   add it as `GITHUB_TOKEN` and grant the notebook access. A fine-grained token needs
   **Contents: Read-only** for this repository. If your collaborator account cannot select
   the repository for a fine-grained token, use a short-lived classic token with `repo` scope.
   The setup cell reads the secret at runtime. The notebook uses CUDA for Mini when a GPU
   is enabled, and gradient boosting runs on CPU.
4. Leave `RUN_MODE = 'smoke'` and run all cells. The notebook locates the data under
   `/kaggle/input`. If discovery fails, set `DATA_DIR` to the folder with the metadata CSV.
   Check that both models report finite `(1000, 12)` probabilities.
5. Change `RUN_MODE` to `'full'` and save a private version that runs all cells. The two
   models each score **4,626 Validation** and **5,442 Test** ECGs for the full 1.1.1 release.
   Gradient boosting refits on the same official Training ECGs for each split, with the
   same settings and seed. Compare Mini's Test SHD AUROC to the published **0.820** reference.
6. Keep the completed private version and its `runs/` output. Download the aggregate files
   under `results/baselines/` and place them in the repository's `results/baselines/`.

The notebook imports the shared Python package directly from the checkout and uses Kaggle's
preinstalled dependencies. If a previous run installed dependencies and PyTorch now fails to
import, stop that Kaggle session and start a fresh one before rerunning the updated setup cell.
When updating notebook cells in Kaggle,
clear outputs before committing the `.ipynb` to GitHub. On an interactive repeat, use a
fresh `RUNS_ROOT` if the destination already holds a completed run.

## Run locally or from a Kaggle terminal

Use Python **3.10 or later**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
bash scripts/download_weights.sh --mini-only

export ECHONEXT_DIR=/path/to/directory/containing/the/official/files
python scripts/run_baseline.py --config configs/mini_baseline.json
python scripts/run_baseline.py --model mini --split test --limit 0
python scripts/run_baseline.py --model gradient_boosting --split test --limit 0
pytest -q
```

Mini-only downloads the pinned torch architecture and roughly 4 MB checkpoint,
verifying both SHA-256 hashes. It avoids the tutorial's legacy Lightning stack.
Without `--mini-only`, the downloader also fetches the two teacher candidates.
No dataset is downloaded or uploaded by these scripts.

CLI flags override JSON configuration values. For example, from a Kaggle terminal in
the cloned repository:

```bash
python scripts/run_baseline.py \
  --config configs/mini_baseline.json \
  --data-dir /kaggle/input/YOUR-DATASET/ACTUAL-FILE-ROOT \
  --runs-root /kaggle/working/runs/smoke \
  --device cuda
```

Use `--device cpu` without a GPU. The CLI defaults to up to 1,000 Validation ECGs, seed 42,
Mini batch size 64, and zero loader workers. `--limit 0` scores the entire split. By default,
full runs go to `runs/<model>/<split>/` and limited runs go to `runs/smoke/<model>/<split>/`.
Use `--split val --limit 0` for full Validation results or `--split train --limit 0` to create
Mini predictions for a distillation dry run. Use `--runs-root runs/dummy` for those dummy
predictions. `--train-limit` optionally caps gradient boosting training for a quick check.
Each run requires a fresh or empty output directory outside the source data folder.

Gradient boosting needs the metadata CSV and only the Training and requested scoring
split's `*_tabular_features.npy` files. It uses observed labels and fits independent
histogram gradient-boosted classifiers with a fixed seed. Outputs with one observed
class use that class frequency. Outputs with no training labels use probability 0.5.
The manifest records observed and positive training counts for each label.

## Expected files and alignment

The loader follows [PhysioNet version 1.1.1](https://physionet.org/content/echonext/1.1.1/).
Point `ECHONEXT_DIR` or `data_dir` at this directory (a nested Kaggle folder is fine):

```text
echonext_metadata_100k.csv               # either capitalization is accepted
EchoNext_train_waveforms.npy
EchoNext_train_tabular_features.npy
EchoNext_val_waveforms.npy
EchoNext_val_tabular_features.npy
EchoNext_test_waveforms.npy
EchoNext_test_tabular_features.npy
EchoNext_no_split_waveforms.npy          # released but never used for experiments
EchoNext_no_split_tabular_features.npy   # released but never used for experiments
```

For Mini, only the requested split's two arrays are opened. Each waveform file is
`[N, 1, 2500, 12]`; each tabular file is `[N, 7]`. To align labels, filter the
metadata by its exact `split` value **without sorting**. The first filtered row
corresponds to array row 0 for that split, and so on. A metadata CSV export column
such as `Unnamed: 0` is not used as an array index.

`row_id` is the zero-based position in the original complete CSV, including
`no_split` rows. `split_row` is the zero-based index into the selected split file.
Neither is an invented ECG identifier. Both remain stable when taking a subset.
Use `row_id` together with the dataset version/metadata hash when joining outputs.

The loader checks required columns, allowed splits, input shapes/row counts,
binary labels, missing SHD labels, and finite inputs as ECGs are read. It rejects
patient overlap between train/val/test and repeated patients within val/test.
Repeated patients within train are allowed. `no_split` may contain earlier ECGs
from evaluation patients and is excluded from all experiments. Do not move those
rows into training.

Shape checks cannot detect an array that somebody reordered without changing its
length: the arrays do not carry patient IDs. Keep the original source files intact.
Record the attached Kaggle dataset version alongside the saved manifest; the
declared PhysioNet version is not a cryptographic verification of the data release.

## Shared batch contract for B and C

```python
from torch.utils.data import DataLoader
from compress_ecg.data import EchoNextDataset, SHD_INDEX

train = EchoNextDataset('/actual/data/root', 'train')
loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
batch = next(iter(loader))
x = batch['waveform']                 # float32 [B, 12, 2500], 250 Hz
tabular = batch['tabular']             # float32 [B, 7], already preprocessed
targets = batch['labels']             # float32 [B, 12]; unobserved values filled 0
mask = batch['label_mask']            # bool [B, 12]; False means exclude this target
shd_targets = targets[:, SHD_INDEX]   # composite SHD target, always observed
```

For multilabel training, mask each target's unreduced loss before averaging:

```python
import torch.nn.functional as F

losses = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
loss = losses[mask].mean()
```

The remaining keys are `row_id`, `split_row`, and `patient_key`; they are for
alignment and auditing, never model inputs. The dataset uses read-only NumPy
memory maps and copies one sample at a time. It reopens maps in spawned/forked
workers instead of copying the training waveform array into RAM.

The seven supplied tabular columns are ordered: `sex`, `age_at_ecg`,
`ventricular_rate`, `atrial_rate`, `pr_interval`, `qrs_duration`, `qt_corrected`.
This differs from the PhysioNet README, which lists age last.
Use this array directly. The metadata's raw values have different preprocessing.
Echo measurements and echo-derived flags are targets/audit information only.

Waveforms have already been filtered, clipped, and normalized in the release.
Do not repeat those operations for Mini. The loader performs only a layout
transpose from `[1, 2500, 12]` to `[12, 2500]` and float32 conversion. Mini's
adapter restores the layout expected by its upstream forward method.

B must implement and validate any teacher-specific resampling/scaling separately;
this loader does **not** resample to 500 Hz or reconstruct raw millivolt signals
from the normalized release. C can directly consume the channel-first arrays.

## Prediction handoff to E

The following files go to the private run directory (default `runs/`, git-ignored):

| File | Contents |
|---|---|
| `predictions.npz` | Logits, probabilities, labels, label mask, row IDs, split rows, patient keys, and label names |
| `metadata.csv` | Aligned IDs, split, and available race/ethnicity, sex, age, and care-setting fields |
| `manifest.json` | Code commit/dirty flag, model/source hashes, data version/path, metadata hash, sampling settings, array shapes/sizes, and runtime versions |
| `metrics.json` | Per-output observed/positive counts, AUROC, and Brier score |
| `gradient_boosting.joblib` | Fitted model, saved only in gradient boosting runs |

`probabilities`, `labels`, and `label_mask` have shape `[N, 12]`. The last column
is the composite SHD output, not the maximum of the other 11 probabilities.
`labels` preserves missing component labels as `NaN` on disk; `label_mask` is
the corresponding boolean availability mask. AUROC is JSON `null` when only one
class is observed, and metrics exclude unobserved targets.

```python
import numpy as np
import pandas as pd

with np.load('/private/run/predictions.npz', allow_pickle=False) as run:
    probabilities = run['probabilities']
    labels = run['labels']
    mask = run['label_mask']
    names = run['label_columns']
    row_ids = run['row_id']
metadata = pd.read_csv('/private/run/metadata.csv', dtype={'patient_key': 'string'})
assert np.array_equal(row_ids, metadata['row_id'].to_numpy())
```

The output order follows the [released Mini implementation](https://github.com/PierreElias/IntroECG/tree/15233e9392e9dc10c136a5624abf10199207bce9/7-EchoNext%20Minimodel):

1. `lvef_lte_45_flag`
2. `lvwt_gte_13_flag`
3. `aortic_stenosis_moderate_or_greater_flag`
4. `aortic_regurgitation_moderate_or_greater_flag`
5. `mitral_regurgitation_moderate_or_greater_flag`
6. `tricuspid_regurgitation_moderate_or_greater_flag`
7. `pulmonary_regurgitation_moderate_or_greater_flag`
8. `rv_systolic_dysfunction_moderate_or_greater_flag`
9. `pericardial_effusion_moderate_large_flag`
10. `pasp_gte_45_flag`
11. `tr_max_gte_32_flag`
12. `shd_moderate_or_greater_flag`

E owns the complete calibration/subgroup scorecard. A's metrics are a sanity
check, with no confidence intervals or fairness conclusions. Keep patient-level
outputs and notebook versions private; only reviewed aggregate tables/figures
belong in the repository's `results/` directory.

For the Kaggle full run, preserve all four `runs/<model>/<split>/` folders in the private
notebook output. E can attach that output and read the folders with `predictions.load_run`.
Keep the gradient boosting checkpoint there or download it to the ignored
`checkpoints/baselines/` directory. To reproduce a saved model's predictions, load your
checkpoint with `joblib.load` and pass it to `predict_gradient_boosting`. Use the scikit-learn
version recorded in its run manifest. Mini's checkpoint can be re-downloaded with
`bash scripts/download_weights.sh --mini-only`.

## Profile Mini on the deployment machine

Install the package and download Mini first. This command creates the aggregate table on CPU:

```bash
python scripts/run_profile.py --device cpu --batch-sizes 1 8 32 64 \
  --output results/profiling/mini_cpu.csv
```

To compare PyTorch eager, TorchInductor, and ONNX Runtime on CPU:

```bash
pip install -e '.[onnx]'
python scripts/run_profile.py --device cpu --backends eager compile onnx \
  --trace-dir runs/profiling/traces --output results/profiling/mini_cpu_backends.csv
```

Add `--student checkpoints/student.pt` when the width-5 checkpoint is available. Each row
reports warmed latency, throughput, model artifact size, sampled process RSS, and peak CUDA
allocation on CUDA runs. The timers exclude compilation and export, which are recorded in
`setup_seconds`. The sampled RSS includes runtime caches and may miss brief spikes. The
ONNX export uses fixed batch shapes and is checked against eager logits before timing.
Keep profiler traces under `runs/` and commit reviewed aggregate CSVs under `results/profiling/`.

## Verification and acceptance

`pytest -q` uses synthetic files with interleaved metadata to check array/label
alignment, split integrity, missing-label masks, subset reproducibility,
memory-map serialization, and output contracts. With Mini downloaded, it also
compares adapter predictions against the original upstream model and exercises
the CLI using the released checkpoint. Those two tests skip if weights are absent.

The synthetic tests also train and reload gradient boosting, run the tabular CLI with
no waveform files, and check that a failed checkpoint write leaves the run retryable.
For real-data acceptance, run the private Kaggle notebook in both modes and give E the
full run folders and exact code/data version identifiers.
