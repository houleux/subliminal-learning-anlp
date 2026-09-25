"""CPU-only tests for sl_exp helpers."""

import math

from sl_exp.data import dataset_hash, scramble_completions, subsample
from sl_exp.evaluate import logsumexp, surface_forms
from sl_exp.prompts import ANIMAL_QUESTIONS
from sl_exp.sweep import make_run_id
from sl_exp.train import TrainConfig


def _rows(n: int) -> list[dict[str, str]]:
    return [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(n)]


def test_subsample_nested_and_deterministic() -> None:
    rows = _rows(100)
    small, big = subsample(rows, 10, 0), subsample(rows, 50, 0)
    assert big[:10] == small
    assert subsample(rows, 10, 0) == small
    assert subsample(rows, 10, 1) != small


def test_scramble_has_no_fixed_points_and_preserves_marginal() -> None:
    rows = _rows(200)
    scr = scramble_completions(rows, seed=3)
    assert all(s["completion"] != f"c{i}" for i, s in enumerate(scr))
    assert sorted(s["completion"] for s in scr) == sorted(r["completion"] for r in rows)
    assert [s["prompt"] for s in scr] == [r["prompt"] for r in rows]


def test_dataset_hash_sensitive_to_content() -> None:
    a = _rows(5)
    b = _rows(5)
    b[2]["completion"] = "x"
    assert dataset_hash(a) == dataset_hash(_rows(5))
    assert dataset_hash(a) != dataset_hash(b)


def test_logsumexp() -> None:
    assert math.isclose(logsumexp([math.log(0.2), math.log(0.3)]), math.log(0.5))


def test_surface_forms_and_questions() -> None:
    assert "Owl" in surface_forms("owl")
    assert len(ANIMAL_QUESTIONS) == 50


def test_run_id_and_alpha_default() -> None:
    cfg = TrainConfig(rank=32, lr=1e-4, seed=2)
    assert cfg.lora_alpha == 32
    assert make_run_id("E3", "owl", False, 10000, cfg) == "E3_owl_n10000_r32_lr0.0001_adamw_s2"
