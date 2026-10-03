"""CPU tests for scripts/confidence_diag.py on tiny synthetic tensors."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from tests.script_loader import load_script

diag = load_script("confidence_diag")

SHAPE = (3, 8, 8, 8)


def _inputs():
    """Seg logits that make ~half the voxels errors in every region, plus a full mask."""
    gen = torch.Generator().manual_seed(0)
    target = (torch.rand(SHAPE, generator=gen) > 0.5).float()
    # Seg prediction = target flipped on a checkerboard-ish subset -> mixed errors.
    flip = torch.rand(SHAPE, generator=gen) > 0.6
    pred_pos = (target > 0.5) ^ flip
    seg = torch.where(pred_pos, torch.tensor(4.0), torch.tensor(-4.0))
    mask = torch.ones(SHAPE[1:], dtype=torch.bool)
    return seg, target, mask, flip


def _perfect_conf(flip):
    """High confidence (P(correct)) where correct, low where wrong."""
    return torch.where(flip, torch.tensor(-5.0), torch.tensor(5.0))


@pytest.mark.parametrize("r", [0, 1, 2])
def test_perfect_channel_gives_auroc_one_on_own_region(r):
    seg, target, mask, flip = _inputs()
    conf = torch.zeros(SHAPE)
    conf[r] = _perfect_conf(flip[r])
    rows = diag.cross_channel_auroc(seg, conf, target, mask, 0.5)
    assert rows[r]["error_region"] == diag.REGIONS[r]
    assert rows[r][f"auroc_ch_{diag.REGIONS[r]}"] == 1.0


def test_constant_channel_gives_exactly_half():
    # case_auroc uses average ranks, so an all-tied score gives exactly 0.5.
    seg, target, mask, _ = _inputs()
    conf = torch.full(SHAPE, 1.3)
    for row in diag.cross_channel_auroc(seg, conf, target, mask, 0.5):
        assert row["auroc_ch_WT"] == 0.5


def test_swapped_channel_is_high_off_diagonal():
    seg, target, mask, flip = _inputs()
    conf = torch.zeros(SHAPE)
    conf[2] = _perfect_conf(flip[0])  # WT channel carries ET's errors
    rows = {r["error_region"]: r for r in diag.cross_channel_auroc(seg, conf, target, mask, 0.5)}
    assert rows["ET"]["auroc_ch_WT"] == 1.0
    assert rows["ET"]["auroc_ch_ET"] == 0.5


@pytest.mark.parametrize("r", [0, 1, 2])
def test_polarity_high_confidence_means_correct(r):
    # Errors get LOW confidence logits; if the 1 - sigmoid flip were missing this would be 0.0.
    seg, target, mask, flip = _inputs()
    conf = _perfect_conf(flip)
    rows = diag.cross_channel_auroc(seg, conf, target, mask, 0.5)
    assert rows[r][f"auroc_ch_{diag.REGIONS[r]}"] == 1.0


def test_partial_mask_is_applied():
    seg, target, _, flip = _inputs()
    mask = torch.zeros(SHAPE[1:], dtype=torch.bool)
    mask[:4] = True
    # Inside the mask confidence ranks errors perfectly; outside, the opposite ordering.
    good = _perfect_conf(flip)
    conf = torch.where(mask.expand(SHAPE), good, -good)
    rows = diag.cross_channel_auroc(seg, conf, target, mask, 0.5)
    for r in range(3):
        assert rows[r][f"auroc_ch_{diag.REGIONS[r]}"] == 1.0


def test_strict_greater_than_tie_is_predicted_negative():
    # seg logit 0.0 -> sigmoid 0.5 == threshold -> NOT predicted positive under strict `>`.
    seg = torch.zeros(SHAPE)
    target = torch.zeros(SHAPE)
    target[:, :4] = 1.0  # half the voxels are truly positive -> those are errors
    rows = diag.cross_channel_auroc(
        seg, torch.zeros(SHAPE), target, torch.ones(SHAPE[1:]).bool(), 0.5
    )
    assert rows[0]["error_rate"] == pytest.approx(0.5)


def test_single_class_errors_are_nan():
    seg = torch.full(SHAPE, 4.0)
    target = torch.ones(SHAPE)  # prediction == target everywhere -> no errors
    rows = diag.cross_channel_auroc(
        seg, torch.zeros(SHAPE), target, torch.ones(SHAPE[1:]).bool(), 0.5
    )
    assert all(math.isnan(r["auroc_ch_ET"]) for r in rows)
    assert rows[0]["error_rate"] == 0.0


def test_channel_stats_inside_outside():
    conf = torch.zeros(SHAPE)  # sigmoid(0) = 0.5 everywhere ...
    pred_wt = torch.zeros(SHAPE[1:], dtype=torch.bool)
    pred_wt[:4] = True  # first half of depth is "predicted WT"
    sample = torch.ones(SHAPE[1:], dtype=torch.bool)
    conf[0, :4] = 10.0  # ... except channel ET inside predicted WT: sigmoid ~ 1
    stats = diag.channel_stats(conf, sample, pred_wt)
    et = stats[0]
    assert et["channel"] == "ET"
    assert et["mean_in_pred_wt"] == pytest.approx(1.0, abs=1e-3)
    assert et["mean_out_pred_wt"] == pytest.approx(0.5)
    assert et["mean"] == pytest.approx((1.0 + 0.5) / 2, abs=1e-3)
    assert et["max"] > et["min"]
    wt = stats[2]
    assert wt["std"] == 0.0 and wt["min"] == wt["max"] == pytest.approx(0.5)


def test_distribution_summary_hand_computed():
    df = pd.DataFrame({"auroc_confidence_WT": [0.2, 0.4, 0.6, 0.8]})
    out = diag.distribution_summary(df, ["WT"]).iloc[0]
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(0.5)
    assert out["median"] == pytest.approx(0.5)
    assert out["q25"] == pytest.approx(0.35)
    assert out["q75"] == pytest.approx(0.65)
    assert out["min"] == 0.2 and out["max"] == 0.8
    assert out["frac_below_0_5"] == pytest.approx(0.5)


def test_select_case_indices():
    assert diag.select_case_indices(["a", "b", "c"], ["c", "a"]) == [2, 0]
    with pytest.raises(ValueError, match="zzz"):
        diag.select_case_indices(["a", "b"], ["a", "zzz"])


class _FakeLoader:
    def __init__(self, items):
        self.dataset = items


def _fake_cfg(tmp_path, regions=("ET", "TC", "WT")):
    return OmegaConf.create(
        {
            "inference": {"postprocess": {"threshold": 0.5}},
            "data": {"preprocessing": {"out_dir": str(tmp_path / "prep")}},
            "analysis": {
                "confidence": {"dilation_mm": 5.0, "split": "test", "regions": list(regions)},
                "confidence_diag": {
                    "out_dir": str(tmp_path / "out"),
                    "per_case_csv": str(tmp_path / "pc.csv"),
                    "cases": ["c1"],
                },
            },
        }
    )


def _setup_run(tmp_path, monkeypatch, has_label=True):
    import json

    cfg = _fake_cfg(tmp_path)
    (tmp_path / "prep" / "c1").mkdir(parents=True)
    (tmp_path / "prep" / "c1" / "meta.json").write_text(
        json.dumps({"spacing": [1.0, 1.0, 1.0], "has_label": has_label})
    )
    pd.DataFrame({f"auroc_confidence_{r}": [0.4, 0.6] for r in diag.REGIONS}).to_csv(
        tmp_path / "pc.csv", index=False
    )
    # Known 6-channel output: seg = (+4, -4, +4), conf logits = (0, 1, 2).
    combined = torch.zeros(1, 6, 8, 8, 8)
    combined[0, 0], combined[0, 1], combined[0, 2] = 4.0, -4.0, 4.0
    combined[0, 3], combined[0, 4], combined[0, 5] = 0.0, 1.0, 2.0
    item = {"image": torch.zeros(4, 8, 8, 8), "label": torch.ones(3, 8, 8, 8)}
    s = diag._SCORE
    monkeypatch.setattr(s, "resolve_confidence_checkpoint", lambda cfg: None)
    monkeypatch.setattr(s, "load_confidence_model", lambda cfg, ckpt, dev: nn.Identity())
    monkeypatch.setattr(s, "_ConfidenceWrapper", lambda model, num_regions: nn.Identity())
    monkeypatch.setattr(
        s, "build_confidence_dataloader", lambda cfg, split: (_FakeLoader([item]), ["c1"])
    )
    monkeypatch.setattr(diag, "get_device", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(diag, "sliding_window_predict", lambda m, x, cfg, dev: combined)
    return cfg


def test_run_diagnostic_slices_seg_and_conf_channels(tmp_path, monkeypatch):
    cfg = _setup_run(tmp_path, monkeypatch)
    dist, stats, cross = diag.run_diagnostic(cfg)
    assert list(stats.columns) == [
        "case",
        "channel",
        "mean",
        "std",
        "min",
        "max",
        "mean_in_pred_wt",
        "mean_out_pred_wt",
    ]
    assert list(cross.columns) == [
        "case",
        "error_region",
        "error_rate",
        "auroc_ch_ET",
        "auroc_ch_TC",
        "auroc_ch_WT",
    ]
    # conf = combined[3:6]: sigmoid of logits 0, 1, 2.
    expected = torch.sigmoid(torch.tensor([0.0, 1.0, 2.0])).tolist()
    assert stats["mean"].tolist() == pytest.approx(expected)
    # seg = combined[:3] vs all-ones target: ET right, TC wrong everywhere, WT right.
    assert cross["error_rate"].tolist() == [0.0, 1.0, 0.0]
    for name in ("per_case_auroc_distribution", "channel_stats", "cross_channel_auroc"):
        assert (tmp_path / "out" / f"{name}.csv").is_file()
    assert (tmp_path / "out" / "confidence_diag_config.yaml").is_file()
    assert len(dist) == 3


def test_run_diagnostic_rejects_unlabeled_case(tmp_path, monkeypatch):
    cfg = _setup_run(tmp_path, monkeypatch, has_label=False)
    with pytest.raises(ValueError, match="no ground-truth label"):
        diag.run_diagnostic(cfg)


def test_run_diagnostic_rejects_wrong_region_order(tmp_path, monkeypatch):
    cfg = _setup_run(tmp_path, monkeypatch)
    cfg.analysis.confidence.regions = ["WT", "TC", "ET"]
    with pytest.raises(ValueError, match="regions"):
        diag.run_diagnostic(cfg)


def test_distribution_summary_all_nan_raises():
    df = pd.DataFrame({"auroc_confidence_WT": [float("nan")]})
    with pytest.raises(ValueError, match="all NaN"):
        diag.distribution_summary(df, ["WT"])
