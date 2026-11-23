# Student results

The Student is the EchoNext-Mini design at width 5, with 104,436 parameters. This folder holds
two kinds of runs.

- **Real runs:** The Student copies our fine-tuned Teacher, or learns from labels alone as a control.
- **Stand-in runs:** The Student copies EchoNext-Mini, from before the Teacher's predictions were
  ready. They show that the code works and are not Student results for the report.

Every score below is SHD AUROC on the 4,626 Validation ECGs, times 100. Each run's best epoch was
picked on Validation, so these scores run slightly high. Test is scored once, at the end.

## Real runs

Both groups use seeds 42 to 47 with the same settings, except the share of the loss that comes
from the true labels. Every run is also on the W&B project `kk3851-columbia-university/compress-ecg`.

| Group | Folders | Label weight | SHD AUROC, mean and spread | Mean AUROC of 12 outputs |
|---|---|---|---|---|
| Copies the Teacher | `student_distill_s42` to `_s47` | 0.5 | 83.04, 0.20 | 80.53, 0.27 |
| Labels only | `student_labels_only_s42` to `_s47` | 1.0 | 82.62, 0.23 | 79.94, 0.28 |

The other settings are temperature 2, learning rate 0.001, batch size 32, and 4 loader workers.
`configs/student_distill.json` and `configs/student_labels_only.json` hold them. The spread is the
standard deviation over the 6 seeds.

Each folder holds three files.

- **`history.json`:** Every epoch's losses, Validation scores, time, speed, and peak GPU memory.
- **`training_config.json`:** The settings and the manifest of the Teacher run that was copied.
- **`val_metrics.json`:** AUROC and Brier score of the saved epoch for all 12 outputs.

The weights and the per-patient predictions stay on the lab server, under
`cnn_profiling_2/student-distill/checkpoints/student` and `cnn_profiling_2/student-distill/runs`.

## FP16 Students

`fp16_vs_fp32_val.json` compares each Teacher-copying Student with its FP16 version, made by
`scripts/run_quantize.py`. The mean SHD AUROC is 83.04 in both, the largest change for any seed is
0.0008 points, and the largest change in any predicted probability is 0.0028. Weights and buffers
shrink from 423,352 to 211,820 bytes.

## Training speed

`profiling/` holds the step timings made by `scripts/profile_student_training.py`. The `before` and
`after` files compare the original loader with the current one. The `_r1` to `_r3` files are three
rounds that alternate the two. Each `steps_*.json` file records the hardware, software, batch size,
warm-up, and repeat count, and each `profiler_*.txt` file holds the PyTorch Profiler table.

## Inference speed

`student_speed_gpu2.json` times the Teacher-copying Student, seed 42, made by
`scripts/time_student.py`. The matching Teacher file is
[`../teacher/teacher_speed_gpu2.json`](../teacher/teacher_speed_gpu2.json). Both were timed on the
same GPU model on the lab server, an NVIDIA RTX PRO 6000, with 10 warm-up and 50 timed calls per batch size, and both time only the
forward pass on a batch already on the GPU.

## Stand-in runs

Every stand-in run is in `stand_in/`. Each one's `training_config.json` records `"model": "mini"`
under `teacher_manifest`. The `mini_teacher_phoenix` seeds are still used as a comparison group by
`scripts/log_results_wandb.py`.

- **`echo_mini_fulldataset_colab_v1`:** One full run on all 72,475 Training ECGs, seed 42. Its best
  Validation SHD AUROC was 82.26, picked as the best of 28 epochs on Validation.
- **`mini_small_train512_val256`:** A short run on 512 Training and 256 Validation ECGs, to check
  that training runs.
- **`sweep_20261119_221212_397212`:** Six settings of learning rate and temperature, one seed each,
  on 512 Training and 256 Validation ECGs. With only 256 Validation ECGs, the gaps between settings
  are within noise.
- **`mini_teacher_phoenix_s42` to `_s47`:** The full run repeated for seeds 42 to 47 on the lab
  server. Their mean SHD AUROC is 82.16, with a spread of 0.15. Each folder also holds
  `best_student.pt` with the saved weights.
- **`mini_teacher_phoenix_s42/val_gap_vs_mini.json`:** Seed 42 compared with EchoNext-Mini on the
  same Validation ECGs.

## Using the Student files

Load a saved file with `student.load_student`. For quantization, convert all six seeds for the
accuracy checks, so results can be reported as an average and spread. Speed, memory, and file
size are the same for every seed, so measure them once.

The `.pt` files are ignored by git by default, so add a new one with `git add -f`.
