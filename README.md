# compress_ECG

Shrinking a structural heart disease ECG model without breaking its calibration or fairness.
This is our HPML course project. The full plan is in [docs/proposal.pdf](docs/proposal.pdf).

## Goal

EchoNext is an FDA-cleared model that flags structural heart disease (SHD, a physical problem
with the heart's valves, walls, or pumping) from a routine 12-lead ECG. Its public baseline,
EchoNext-Mini, reports an SHD AUROC of 82.0.

We fine-tune a large ECG foundation model as a teacher, distill it into one small student,
and compress that student. After each step we check three things.

- **Accuracy:** SHD AUROC stays within about 2 points of the teacher.
- **Calibration:** Predicted risks still match observed rates.
- **Fairness:** The worst subgroup (race and ethnicity, sex, age, care setting) stays acceptable.

## Scope

The proposal feedback asked us to narrow the plan. We work in this order and expand only once
a stage works.

1. Get data access and reproduce the EchoNext-Mini baseline.
2. Fine-tune the ECG-FM teacher (about 90M parameters) and confirm it beats the baseline.
3. Distill one student of about 0.1M parameters, roughly 3.5 times smaller than EchoNext-Mini.
4. Run a short quantization comparison (FP32, FP16, INT8) on one deployment target.
5. Measure latency and memory, and audit AUROC, calibration, and worst-group performance.

Pruning, more student sizes, and more devices come later, and only if time allows.

## Why AUROC is the main metric

AUROC is the chance that the model scores a random patient with SHD above a random patient
without it. A score of 0.5 is a coin flip and 1.0 is a perfect ranking.

- **No threshold needed:** A screening tool is used at whatever cutoff a clinic picks. AUROC
  rates the ranking across all cutoffs, so it does not depend on one choice.
- **Stable across prevalence:** SHD rates differ between hospitals and care settings. AUROC
  does not change when the share of positive patients changes.
- **Directly comparable:** EchoNext (85.2) and EchoNext-Mini (82.0) both report it, so our
  results sit on the same scale as the published models.
- **Works per subgroup:** Computing it inside each subgroup gives the worst-group AUROC we use
  for fairness.

AUROC does not measure whether a predicted 30% risk really means 30%. That is why we also
report calibration error and the Brier score.

## Data access

The data is the EchoNext dataset on PhysioNet: 100,000 12-lead ECGs (10 seconds at 250 Hz)
with 11 echo-confirmed labels, a composite SHD label, demographics, and official splits.

1. Create a PhysioNet account.
2. Sign the data use agreement on the
   [EchoNext page](https://physionet.org/content/echonext/).
3. Download the files to a folder on your own machine or server.
4. Set the environment variable `ECHONEXT_DIR` to that folder.

Follow these rules from the data use agreement.

- **Sign individually:** Every team member signs the agreement. Never share files or logins.
- **Keep data out of git:** Never commit waveforms, labels, or per-patient predictions.
- **Keep data off hosted devices:** Profile on Qualcomm AI Hub phones with random or
  synthetic inputs only. Compute AUROC on our own machines.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the EchoNext-Mini code and the pretrained teacher weights (about 1.5 GB). The script
pins each source to a fixed version and checks every file against its SHA-256 hash.

```bash
bash scripts/download_weights.sh
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
| `docs/` | Proposal and reports |

## External resources

- [EchoNext dataset](https://physionet.org/content/echonext/) on PhysioNet (restricted access)
- [IntroECG](https://github.com/PierreElias/IntroECG) for the EchoNext-Mini code and weights
- [ECG-FM](https://github.com/bowang-lab/ECG-FM) and
  [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) as teacher candidates
- [Qualcomm AI Hub](https://workbench.aihub.qualcomm.com/) for hosted phone profiling

## Team

Andy Kim, Jason P'ng, Robert Stott, Sanjay Kumar, Shaurya Dugar
