"""Check the summary scores built for the W&B evaluation runs, without files or the network."""

import pytest

from compress_ecg.evaluation_log import (
    SHD_LABEL, label_scores, paired_gap, paired_seed_gap, seed_number, seed_spread, speed_scores, wait_range,
)


def timing(*rows):
    """Return a speed file holding the given (precision, batch, ms per ECG) rows."""
    return {"rows": [{"precision": p, "batch_size": b, "ms_per_ecg": ms, "peak_gpu_memory_bytes": b * 10}
                     for p, b, ms in rows]}


def profile(*rows):
    """Return a profiling file holding the given (workers, median wait ms) rows."""
    return {"rows": [{"num_workers": w, "median_wait_ms": ms} for w, ms in rows]}


def test_label_scores_reads_shd_and_averages_auroc_over_every_label():
    metrics = {"other_flag": {"auroc": 0.6, "brier": 0.2}, SHD_LABEL: {"auroc": 0.8, "brier": 0.1}}

    assert label_scores(metrics) == pytest.approx({"shd_auroc": 0.8, "mean_auroc": 0.7, "shd_brier": 0.1})


def test_label_scores_fails_without_shd():
    with pytest.raises(KeyError):
        label_scores({"other_flag": {"auroc": 0.6, "brier": 0.2}})


def test_seed_spread_gives_mean_and_sample_sd():
    assert seed_spread([1.0, 2.0, 3.0]) == pytest.approx({"mean": 2.0, "sd": 1.0, "n_seeds": 3})


def test_seed_spread_rejects_one_seed():
    with pytest.raises(ValueError, match="at least 2"):
        seed_spread([1.0])


def test_paired_gap_matches_a_hand_computed_t_interval():
    # Gaps are 2, 3, 1: mean 2, sd 1, and t(0.975, 2 df) = 4.302653, so the half width is 2.484138.
    result = paired_gap([3.0, 5.0, 4.0], [1.0, 2.0, 3.0])

    assert result == pytest.approx(
        {"gap": 2.0, "gap_low": -0.484138, "gap_high": 4.484138, "seeds_ahead": 3, "n_seeds": 3}, abs=1e-6)


def test_paired_gap_rejects_unmatched_seeds():
    with pytest.raises(ValueError, match="matching seeds"):
        paired_gap([1.0, 2.0], [1.0])


def test_speed_scores_keeps_only_the_requested_rows():
    result = speed_scores(timing(("fp32", 1, 5.0), ("fp32", 64, 1.2), ("bf16", 64, 0.5)), rows=(("fp32", 64),))

    assert result == {"fp32_b64_ms_per_ecg": 1.2, "fp32_b64_peak_gpu_bytes": 640}


def test_speed_scores_fails_on_a_missing_row():
    with pytest.raises(KeyError):
        speed_scores(timing(("fp32", 1, 5.0)), rows=(("fp16", 64),))


def test_wait_range_spans_rounds_for_one_worker_count():
    rounds = [profile((0, 4.1), (4, 4.4)), profile((0, 7.2), (4, 6.0))]

    assert wait_range(rounds, 0) == {"wait_ms_low": 4.1, "wait_ms_high": 7.2, "rounds": 2}


def test_wait_range_fails_when_no_round_used_that_worker_count():
    with pytest.raises(ValueError, match="num_workers=8"):
        wait_range([profile((0, 4.1))], 8)


@pytest.mark.parametrize(("folder", "seed"), [
    ("student_distill_s42", 42), ("mini_teacher_phoenix_s47", 47), ("student_labels_only_s100", 100),
])
def test_seed_number_reads_the_folder_suffix(folder, seed):
    assert seed_number(folder) == seed


def test_seed_number_rejects_a_folder_without_a_seed():
    with pytest.raises(ValueError, match="no seed"):
        seed_number("student_distill")


def test_paired_seed_gap_pairs_by_seed_not_by_order():
    values = {44: 4.0, 42: 3.0, 43: 5.0}
    baseline = {42: 1.0, 43: 2.0, 44: 3.0}

    assert paired_seed_gap(values, baseline) == paired_gap([3.0, 5.0, 4.0], [1.0, 2.0, 3.0])


def test_paired_seed_gap_rejects_different_seeds():
    with pytest.raises(ValueError, match=r"\[42, 43\] and \[43, 48\]"):
        paired_seed_gap({42: 1.0, 43: 2.0}, {43: 1.0, 48: 2.0})
