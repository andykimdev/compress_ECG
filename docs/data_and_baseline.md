# A: data loading and released Mini baseline

A's first target is a repeatable run on about **1,000 validation ECGs**, using the
official arrays and released weights. This supplies the shared loader to B/C and
aligned predictions to E. It does not train a teacher/student or quantize Mini;
those belong to B/C/D. The private Kaggle dataset is assumed to be uploaded already.

## Run on Kaggle

1. Upload/open `notebooks/01_data_and_baseline.ipynb` in a **private** notebook.
2. Attach the team's private EchoNext dataset. Preserve the original filenames
   and row ordering; extract archives before using the loader.
3. Enable Internet for the repository, dependencies, and public Mini downloads.
   CPU works; optionally enable a GPU. The notebook selects CUDA when available.
4. Run the setup cells. They clone `shaurya/data-loader-and-baselines` and install
   the shared package. The implementation must first be committed/pushed to that
   branch for Kaggle to retrieve it. For exact repeats, set `CODE_COMMIT` to its SHA.
5. The notebook locates the metadata file under `/kaggle/input`. If several copies
   are attached, set `DATA_DIR` to the directory containing the official files.
6. Run the batch check, then Mini. Save the completed notebook version and its
   outputs privately. Another teammate should repeat this from their own account.

The notebook is a thin launcher: the reusable implementation lives in `data.py`
and `baseline.py`. When updating notebook cells in Kaggle, clear outputs before
committing the `.ipynb` back to GitHub. The notebook never synchronizes automatically.

## Run locally or from a Kaggle terminal

Use Python **3.10 or later**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
bash scripts/download_weights.sh --mini-only

export ECHONEXT_DIR=/path/to/directory/containing/the/official/files
python scripts/run_baseline.py --config configs/mini_baseline.json
pytest -q
```

Mini-only downloads the pinned torch architecture and roughly 4 MB checkpoint,
verifying both SHA-256 hashes. It avoids the tutorial's legacy Lightning stack.
Without `--mini-only`, the downloader also fetches the two teacher candidates.
No dataset is downloaded or uploaded by these scripts.

CLI flags override JSON configuration values. For example, on Kaggle:

```bash
python scripts/run_baseline.py \
  --config configs/mini_baseline.json \
  --data-dir /kaggle/input/YOUR-DATASET/ACTUAL-FILE-ROOT \
  --output-dir /kaggle/working/runs/mini_val_1000_seed42 \
  --device cuda
```

Use `--device cpu` without a GPU. `--limit 0` runs a full split; defaults are
validation, 1,000 ECGs, seed 42, batch size 64, and zero data-loader workers.
Sampling is without replacement and selected split indices are sorted back into
their original order. `--split test --limit 0` is available for the final locked
evaluation. The paper's 0.820 SHD AUROC is not an expected result on a small
validation subset. No clinical performance result is claimed by synthetic tests.

Each run requires a fresh or empty output directory. Save outputs outside the
source data directory. Nothing modifies `/kaggle/input` or refits preprocessing.

## Expected files and alignment

The loader follows [PhysioNet version 1.1.1](https://physionet.org/content/echonext/1.1.1/).
Point `ECHONEXT_DIR` or `data_dir` at this directory (a nested Kaggle folder is fine):

```text
EchoNext_metadata_100k.csv
EchoNext_train_waveforms.npy
EchoNext_train_tabular_features.npy
EchoNext_val_waveforms.npy
EchoNext_val_tabular_features.npy
EchoNext_test_waveforms.npy
EchoNext_test_tabular_features.npy
EchoNext_no_split_waveforms.npy          # released but never used for experiments
EchoNext_no_split_tabular_features.npy   # released but never used for experiments
```

Only the requested split's two arrays are opened. Each waveform file is
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

The seven supplied tabular columns are ordered: `sex`, `ventricular_rate`,
`atrial_rate`, `pr_interval`, `qrs_duration`, `qt_corrected`, `age_at_ecg`.
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
| `predictions.npz` | Probabilities, labels, label mask, row IDs, split rows, patient keys, and label names |
| `metadata.csv` | Aligned IDs, split, and available race/ethnicity, sex, age, and care-setting fields |
| `manifest.json` | Code commit/dirty flag, model/source hashes, data version/path, metadata hash, sampling settings, array shapes/sizes, and runtime versions |
| `metrics.json` | Per-output observed/positive counts, AUROC, and Brier score |

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

## Verification and acceptance

`pytest -q` uses synthetic files with interleaved metadata to check array/label
alignment, split integrity, missing-label masks, subset reproducibility,
memory-map serialization, and output contracts. With Mini downloaded, it also
compares adapter predictions against the original upstream model and exercises
the CLI using the released checkpoint. Those two tests skip if weights are absent.

Real-data acceptance remains a Kaggle run: another teammate loads a batch, runs
the 1,000-validation-ECG notebook, confirms finite `[1000, 12]` probabilities,
and gives E the private output directory and exact code/data version identifiers.
