# compress_ECG

**Master plan, last updated November 20, 2026.** This file replaces the plan in our original
[proposal](docs/proposal.pdf) wherever the two differ.

We are shrinking a heart disease detection model so it can run on small devices, without making
it less accurate, less trustworthy, or less fair. This is our HPML course project. No biology
background is needed to follow this page.

## The problem

Doctors have two common tests for looking at a heart.

| | ECG | Heart ultrasound |
|---|---|---|
| **What it does** | Records the heart's electrical signal from sensors on the skin | Makes a moving picture of the heart using sound waves |
| **What it shows** | Electrical activity | Physical problems, such as a leaky valve or a weak pump |
| **Practical cost** | Cheap and routine, 10 seconds of recording | Takes more time, equipment, and training |
| **Role in this project** | Model input | Correct answer used for training and scoring |

A model can learn to look at an ECG and predict what an ultrasound would find. A clinic could
then use that prediction to decide who should get an ultrasound. The condition it predicts is
called structural heart disease (SHD), which means a physical problem with the heart.

EchoNext is an FDA-cleared model that does this. Our question is how small such a model can get
before it stops being useful.

## What the model takes in and puts out

**Input:**

- **Waveform:** The ECG signal, stored as a 12 by 2,500 array of numbers. The 12 is the number of
  leads (recording angles on the body), and the 2,500 is 10 seconds of readings at 250 per second.
- **7 extra numbers:** Sex, age, 2 heart rates, and 3 timings that the ECG machine measures.

**Output:** 12 probabilities, one for each yes or no question.

- **11 findings:** Specific problems seen on the ultrasound, such as a narrow aortic valve.
- **1 overall answer:** SHD, which is yes when any of the 11 findings is yes.

## The six model names

Only the Teacher and the Student are trained by us.

| Name | What it is | Parameters | AUROC | Our role |
|---|---|---|---|---|
| EchoNext | Full FDA-cleared Columbia model | Not released | 85.2 | Context only |
| EchoNext-Mini | Released small version, ResNet design | 1.05M | 82.0 | Baseline to compare against |
| ECG-FM | Pretrained ECG foundation model | 90.9M | None yet | Starting point for the Teacher |
| Teacher | ECG-FM plus a new output layer | 90.4M | 85.15 on Validation | Fine-tune it, and it must beat 82.0 |
| Student | EchoNext-Mini design, made narrower | 0.10M | To measure | Train it by distillation |
| Student FP16 and INT8 | Student stored in 16 or 8 bits | 0.10M | To measure | Quantize and deploy it |

AUROC is explained under [How we judge a model](#how-we-judge-a-model). The EchoNext and
EchoNext-Mini scores are the published ones. We have not reproduced them yet. The Teacher score is
ours, on the Validation ECGs, and the [Teacher guide](docs/teacher.md) explains how to get and use
the model.

## The experiment pipeline

Each step produces the input for the next step.

| Step | What we do | What it makes |
|---|---|---|
| 1. Data | Use the 100,000 EchoNext-Mini ECGs and their official splits | No model yet |
| 2. Teacher | Add an output layer to ECG-FM that also reads the 7 extra numbers, fine-tune 6 runs with dropout and mixed precision, then average their weights | Teacher |
| 3. Student | Train the Mini design at width 5 to copy the Teacher | Student |
| 4. Quantize | Convert to FP16, INT8 post-training, and INT8 quant-aware | FP16 and INT8 Students |
| 5. Profile and audit | Measure speed, memory, cost, AUROC, and calibration per subgroup | Comparison of all models |

The terms in this table mean the following.

- **Fine-tuning:** Continuing to train a pretrained model on a new task.
- **Mixed precision:** Training with a mix of 16-bit and 32-bit numbers to run faster.
- **Dropout:** Switching off a random share of values during training, so the model relies less on
  any single one.
- **Weight averaging:** Combining several trained copies of a model into one by averaging their
  numbers.
- **Distillation:** Training a small model to copy the outputs of a large one.
- **Quantization:** Storing a model's numbers with fewer bits, which makes it smaller.
- **Post-training INT8:** Converting to 8 bits after training is finished.
- **Quant-aware INT8:** Simulating 8 bits during training, so the model adapts to them.

### Baselines

A result only means something next to a simpler option, so we compare against two.

- **EchoNext-Mini:** The released weights, 1.05M parameters.
- **Gradient boosting:** A standard non-neural model that uses only the 7 extra numbers.

**The Teacher must beat EchoNext-Mini.** If it does not, then a Student that copies it has
little value, however fast it runs.

### Profiling tools

- **PyTorch Profiler:** Measure each model before and after every change.
- **Speedups:** Try torch.compile, operator fusion, and ONNX Runtime.
- **Batch size sweeps:** Measure throughput at different batch sizes.

### Scope limits

The proposal feedback asked us to narrow the plan, so we expand only once a step works. Pruning,
more Student sizes, and more devices come later, and only if time allows.

## The Student design

The Student has the same layout as EchoNext-Mini. Only the channel widths shrink, from a base
width of 16 to 5. Each stage is a series of residual blocks, where a block is two convolutions
plus a shortcut that adds the input back.

| Layer | Blocks | EchoNext-Mini channels | Student channels |
|---|---|---|---|
| Stem convolution (kernel size 15) | | 16 | 5 |
| Stage 1 | 3 | 16 | 5 |
| Stage 2 | 4 | 32 | 10 |
| Stage 3 | 6 | 64 | 20 |
| Stage 4 | 3 | 128 | 40 |
| Pooling plus the 7 extra numbers | | 263 | 87 |
| Output | | 12 | 12 |

This gives the following sizes, counted from the released model code.

| Model | Parameters | File size |
|---|---|---|
| EchoNext-Mini, FP32 | 1,046,080 | 4.2 MB |
| Student, FP32 | 104,436 | 0.4 MB |
| Student, INT8 | 104,436 | About 0.1 MB, roughly 40 times smaller than EchoNext-Mini |

## How we judge a model

After each step, we check three things on the Test ECGs.

- **Accuracy:** Compressing the Student should keep its SHD AUROC within about 2 points of the
  Teacher.
- **Calibration:** Predicted risks should match reality, so about 30 of every 100 patients given
  a 30% risk should have SHD.
- **Fairness:** The worst-scoring patient group should stay acceptable. We check race and
  ethnicity, sex, age, and care setting, such as emergency or outpatient.

### Why AUROC is the main metric

AUROC is the chance that the model scores a random patient with SHD above a random patient without
it, shown as a percentage. A score of 50 is a coin flip, and 100 is a perfect ranking.

- **No cutoff needed:** Clinics can choose their own cutoff for flagging a patient, and AUROC
  rates the ranking across all of them.
- **Stable when disease rates change:** Rates differ between hospitals, and AUROC does not depend
  directly on the share of patients who have SHD.
- **Comparable:** EchoNext and EchoNext-Mini both report it, so our results use the same scale.
- **Works per group:** Computing it inside each patient group gives the worst-group score.

AUROC does not show whether a predicted 30% risk really means 30%. That is why we also report
calibration error and the Brier score, which is the average squared error of the probabilities.

## Code map

Each role fills in its own files. Functions that are not written yet raise `NotImplementedError`
with a note on what to do, and their docstrings give the exact inputs and outputs.

| Role | File to fill in | Command | Passes on |
|---|---|---|---|
| A: Data and baselines | `baseline.py`, `profiling.py` | `run_baseline.py`, `run_profile.py` | Baseline runs, profiling table |
| B: Teacher | `teacher.py`, `sweep.py`, `runs.py` | `run_teacher.py`, `sweep_teacher.py`, `soup_teacher.py` | Teacher checkpoint and runs |
| C: Student | `distill.py` | `run_student.py` | Student checkpoint and runs |
| D: Quantization | `quantize.py` | `run_quantize.py` | FP16 and INT8 models and runs |
| E: Audit | `audit.py` | `run_audit.py` | Aggregate tables and figures |

Files are in `src/compress_ecg/`, and commands are in `scripts/`. The
[Teacher guide](docs/teacher.md) lists the Teacher's other commands. Three files are shared and
already work, so change them only after telling the team.

- **`data.py`:** Loads the ECGs. Each batch has `waveform`, `tabular`, `labels`, and `label_mask`.
- **`predictions.py`:** Runs any model and saves or loads its prediction files.
- **`student.py`:** Builds, saves, and loads the width 5 Student.

### The two shared formats

Every neural model is called the same way, so any role's code works with any model.

```python
logits = model(waveform, tabular)
# waveform: float32 [batch, 12, 2500]
# tabular:  float32 [batch, 7]
# logits:   [batch, 12], with overall SHD last
```

Every model saves the same four files with `predictions.write_run`, under
`runs/<model>/<split>/`. These hold patient-level values, so they never go in git.

| File | Contents |
|---|---|
| `predictions.npz` | Logits, probabilities, labels, label mask, row IDs, patient keys |
| `metadata.csv` | Row IDs, race and ethnicity, sex, age, care setting |
| `manifest.json` | Code version, data hash, settings |
| `metrics.json` | Basic AUROC and Brier score for a quick check |

### Working before the real models exist

Until a real input is ready, use the released EchoNext-Mini as a dummy, and keep its outputs under
`runs/dummy/`. Dummy numbers only show that code works, so they never go in the report.

- **Role C:** Train from EchoNext-Mini's saved predictions on the Training ECGs.
- **Role D:** Run `run_quantize.py` without `--student`, which converts EchoNext-Mini.
- **Role E:** Build the scorecard on EchoNext-Mini's saved predictions.

`tests/test_pipeline.py` runs every handoff on synthetic ECGs. When your code is ready, replace
your role's stand-in in that test with the real call.

## Current status

- **Done:** We downloaded the data, checked its hashes, and
  [summarized it](results/data_stats). The summary checks found no cleaning step we need.
- **Done:** The shared code framework is in place, with one file for each role to fill in.
- **Next:** Run EchoNext-Mini on the Test ECGs and see whether we get close to 82.0, which also
  tests our data loading and evaluation code. Then add the gradient boosting baseline.
- **Done:** The Teacher is fine-tuned and scores 85.15 SHD AUROC on Validation. The
  [Teacher guide](docs/teacher.md) covers how to download it and make its predictions.
- **Not decided yet:** The one deployment device for steps 4 and 5.

## The data

The data is the EchoNext dataset on PhysioNet. It has 100,000 ECGs from 36,286 patients, because
some patients have more than one ECG. The dataset's split column puts each ECG in one group.

| Group | ECGs | Patients | Which ECGs | Our use |
|---|---|---|---|---|
| Training (`train`) | 72,475 | 26,218 | Possibly several per patient | Train models |
| Validation (`val`) | 4,626 | 4,626 | Latest per patient | Tune and calibrate models |
| Test (`test`) | 5,442 | 5,442 | Latest per patient | Report final scores |
| Unassigned (`no_split`) | 17,457 | 4,618 | Older ECGs of Validation and Test patients | Do not use |

Keep these details in mind when loading the data.

- **Unassigned ECGs:** Training on them would show the model patients it is later scored on. They
  are also stored on a different scale.
- **Column order:** In the feature arrays, age is the second column, even though the dataset's
  README lists it last. The order is sex, age, ventricular rate, atrial rate, PR interval, QRS
  duration, and corrected QT.
- **Missing timings:** A missing PR interval is stored as -2.475, and a missing atrial rate as
  -2.008, so these values mean not measured.
- **Missing ultrasound values:** A label says no when its measurement is missing, so some no
  labels mean unknown. This is most common for the two lung pressure findings, at 44% and 56%.

### How to get the data

1. Create a PhysioNet account.
2. Sign the data use agreement on the
   [EchoNext page](https://physionet.org/content/echonext/).
3. Download the ZIP from the bottom of that page (about 6.5 GB).
4. Unzip it into `data/echonext/1.1.1/` in this repository.
5. Check the files against their hashes.

```bash
cd data/echonext/1.1.1
sed 's/ /  /' SHA256SUMS.txt | shasum -a 256 -c -
```

Every line should end in `OK`. The unzipped files take about 24 GB. The waveforms are stored as
float64, so load them with `np.load(path, mmap_mode="r")` to read from disk instead of memory.

Our code reads the data from `data/echonext/1.1.1/` by default. To keep the data somewhere else,
such as a larger disk on a server, set the environment variable `ECHONEXT_DIR` to that folder.

### Data use rules

These rules come from the data use agreement.

- **Sign individually:** Every team member signs the agreement. Never share files or logins.
- **Keep data out of git:** Never commit waveforms, labels, or per-patient predictions.
- **Keep data off hosted devices:** Profile on hosted devices, such as Qualcomm AI Hub phones,
  with random or synthetic inputs only. Compute AUROC on our own machines.

## Setup

Use Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Download the EchoNext-Mini code and the pretrained weights (about 1.5 GB). The script pins each
source to a fixed version and checks every file against its SHA-256 hash.

```bash
bash scripts/download_weights.sh
```

To work on data loading or the EchoNext-Mini baseline only, a smaller download is enough. The
[data and baseline guide](docs/data_and_baseline.md) covers input shapes and run files.

```bash
bash scripts/download_weights.sh --mini-only
export ECHONEXT_DIR=/path/to/official/files
python scripts/run_baseline.py --config configs/mini_baseline.json
pytest -q
```

### Teacher setup

Role B needs this, and so does anyone who runs the Teacher to make its predictions. Loading
ECG-FM needs fairseq-signals, which works with Python 3.10 and a `transformers` version below
4.50. Keep it in its own environment, outside the repository. The
[Teacher guide](docs/teacher.md) covers the rest.

```bash
python3.10 -m venv ~/venvs/compress-ecg-teacher
source ~/venvs/compress-ecg-teacher/bin/activate
pip install "torch>=2.6" "transformers>=4.19,<4.50" -e '.[dev]' \
  "fairseq-signals @ git+https://github.com/Jwoo5/fairseq-signals@f8f0ff1c788a82c2059cb452cd5462898867489e"
pytest -q tests/test_teacher.py
```

## Repository layout

| Folder | Contents |
|---|---|
| `src/compress_ecg/` | Shared code for loading data, models, compression, and evaluation |
| `scripts/` | Entry points that train, compress, profile, or evaluate a model |
| `configs/` | One config file per experiment |
| `notebooks/` | Exploration and figures |
| `results/` | Aggregate metric tables and figures only |
| `tests/` | Unit tests |
| `docs/` | Proposal, guides, and reports |

## External resources

- [EchoNext dataset](https://physionet.org/content/echonext/) on PhysioNet (restricted access)
- [IntroECG](https://github.com/PierreElias/IntroECG) for the EchoNext-Mini code and weights
- [ECG-FM](https://github.com/bowang-lab/ECG-FM) as the starting point for the Teacher
- [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder), downloaded by our script but not
  part of the current plan
- [Qualcomm AI Hub](https://workbench.aihub.qualcomm.com/) for hosted phone profiling, if we
  choose a phone as the deployment device

## Team

Andy Kim, Jason P'ng, Robert Stott, Sanjay Kumar, Shaurya Dugar
