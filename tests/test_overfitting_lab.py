import json

import numpy as np
import pytest

from overfitting_lab.cli import main
from overfitting_lab.data import make_dataset
from overfitting_lab.models import (
    alpha_sweep,
    best_degree,
    cv_errors,
    learning_curve_errors,
    make_polynomial_model,
)


# ---------- данные ----------
def test_dataset_shapes_and_reproducibility():
    X1, y1 = make_dataset(30, seed=1)
    X2, y2 = make_dataset(30, seed=1)
    assert X1.shape == (30, 1) and y1.shape == (30,)
    assert np.allclose(X1, X2) and np.allclose(y1, y2)


@pytest.mark.parametrize("kwargs", [{"n_samples": 1}, {"noise": -0.1}])
def test_dataset_validation(kwargs):
    with pytest.raises(ValueError):
        make_dataset(**kwargs)


# ---------- модели ----------
def test_model_validation():
    with pytest.raises(ValueError):
        make_polynomial_model(0)
    with pytest.raises(ValueError):
        make_polynomial_model(2, alpha=-1)


@pytest.mark.parametrize("folds", [1, 31])
def test_cv_folds_validation(folds):
    X, y = make_dataset(30)
    with pytest.raises(ValueError):
        cv_errors(X, y, [1, 2], folds=folds)


def test_cv_errors_structure():
    X, y = make_dataset(30)
    e = cv_errors(X, y, [1, 2, 3])
    assert e["degrees"] == [1, 2, 3]
    assert len(e["train"]) == len(e["val"]) == len(e["val_std"]) == 3


def test_high_degree_overfits():
    """Высокая степень: ошибка на обучении мала, на валидации заметно больше."""
    X, y = make_dataset(30, noise=0.25, seed=42)
    e = cv_errors(X, y, [3, 15])
    assert e["train"][1] < e["train"][0]  # на обучении сложная модель лучше
    assert e["val"][1] > e["val"][0]      # а на новых данных хуже


def test_best_degree_is_moderate():
    X, y = make_dataset(30, noise=0.25, seed=42)
    e = cv_errors(X, y, list(range(1, 16)))
    assert 2 <= best_degree(e) <= 9


def test_one_se_rule_prefers_simpler_model():
    e = {"degrees": [1, 2, 3], "val": [0.50, 0.101, 0.100], "val_std": [0.01, 0.02, 0.02]}
    assert best_degree(e, one_se_rule=False) == 3
    assert best_degree(e, one_se_rule=True) == 2


def test_regularization_helps():
    X, y = make_dataset(30, noise=0.25, seed=42)
    s = alpha_sweep(X, y, 12, [1e-3, 1e-2, 1e-1])
    assert min(s["val"]) < s["baseline"]


def test_learning_curve_gap_shrinks_with_more_data():
    c = learning_curve_errors(12, [24, 150], repeats=10)
    gap_small = c["test"][0] - c["train"][0]
    gap_large = c["test"][1] - c["train"][1]
    assert gap_large < gap_small


# ---------- сквозной тест ----------
def test_cli_end_to_end(tmp_path):
    code = main(["--out", str(tmp_path), "--max-degree", "6", "--complex-degree", "6"])
    assert code == 0
    for name in ["fits.png", "error_curves.png", "learning_curves.png", "regularization.png"]:
        assert (tmp_path / name).stat().st_size > 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert 1 <= summary["best_degree"] <= 6
