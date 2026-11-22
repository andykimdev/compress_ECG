# Student results

Every run in this folder is a stand-in. The Student learned to copy EchoNext-Mini, not our
fine-tuned Teacher, because the Teacher's predictions on the Training ECGs were not ready yet.
Each run's `training_config.json` records this as `"model": "mini"` under `teacher_manifest`.

These runs show that the distillation code works end to end. They are not Student results for the
report.

## What each folder holds

- **`echo_mini_fulldataset_colab_v1`:** One full run on all 72,475 Training ECGs, seed 42. Its best
  Validation SHD AUROC was 82.26, picked as the best of 28 epochs on Validation.
- **`mini_small_train512_val256`:** A short run on 512 Training and 256 Validation ECGs, to check
  that training runs.
- **`sweep_20261119_221212_397212`:** Six settings of learning rate and temperature, one seed each,
  on 512 Training and 256 Validation ECGs. With only 256 Validation ECGs, the gaps between settings
  are within noise.
- **`mini_teacher_phoenix_s42` to `_s47`:** The full run repeated for seeds 42 to 47 on the lab
  server, with each seed set before the Student is built. Each folder holds the training history,
  the settings, `val_metrics.json` with the 12 Validation AUROCs of the saved epoch, and
  `best_student.pt` with the saved weights (104,436 parameters, about 0.5 MB). Each best epoch was
  picked on Validation, so these scores run slightly high.
- **`mini_teacher_phoenix_s42/val_gap_vs_mini.json`:** Seed 42, the seed closest to the median,
  compared with EchoNext-Mini on the same 4,626 Validation ECGs.

## Using the Student files

Load any of the six files with `student.load_student`. For quantization, convert all six for the
accuracy checks, so results can be reported as an average and spread like the Teacher and Student.
Speed, memory, and file size are the same for every seed, so measure them once.

The `.pt` files are ignored by git by default, so add a new one with `git add -f`.

## How the real Student results will differ

- **Teacher:** The Student will copy the fine-tuned Teacher's predictions.
- **Seeds:** Results will be the average and spread over seeds 42 to 47, the same way the Teacher
  is reported.
