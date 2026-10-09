"""Модели и вычисление ошибок.

Три эксперимента:
  * cv_errors              - ошибка на обучении и на валидации (k-fold) для разных степеней;
  * learning_curve_errors  - как ошибка зависит от размера обучающей выборки;
  * alpha_sweep            - как ошибка зависит от силы регуляризации.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .data import make_dataset


def make_polynomial_model(degree: int, alpha: float = 0.0) -> Pipeline:
    """Полиномиальная регрессия заданной степени.

    alpha = 0 -> обычный метод наименьших квадратов (без регуляризации),
    alpha > 0 -> Ridge (L2-регуляризация, штраф за большие коэффициенты).
    """
    if degree < 1:
        raise ValueError("degree должен быть не меньше 1")
    if alpha < 0:
        raise ValueError("alpha не может быть отрицательным")

    regressor = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("reg", regressor),
        ]
    )


def _check_folds(n_samples: int, folds: int) -> None:
    if folds < 2:
        raise ValueError("folds должен быть не меньше 2")
    if folds > n_samples:
        raise ValueError("folds не может превышать число объектов")


def cv_errors(
    X: np.ndarray,
    y: np.ndarray,
    degrees: list[int],
    alpha: float = 0.0,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, list[float]]:
    """MSE на обучении и на валидации (среднее и разброс по фолдам) для каждой степени."""
    _check_folds(len(y), folds)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    out: dict[str, list[float]] = {
        "degrees": list(degrees),
        "train": [],
        "val": [],
        "val_std": [],
    }
    for d in degrees:
        res = cross_validate(
            make_polynomial_model(d, alpha),
            X,
            y,
            cv=kf,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )
        out["train"].append(float(-res["train_score"].mean()))
        out["val"].append(float(-res["test_score"].mean()))
        out["val_std"].append(float(res["test_score"].std()))
    return out


def best_degree(errors: dict[str, list[float]], one_se_rule: bool = True) -> int:
    """Выбирает степень по ошибке на валидации.

    one_se_rule=False: степень с минимальной ошибкой.
    one_se_rule=True (по умолчанию): самая простая степень, чья ошибка не хуже
    минимальной больше чем на одно стандартное отклонение по фолдам. Так мы не
    гонимся за случайными колебаниями и берём простую модель, если она
    практически не уступает сложной.
    """
    val = np.asarray(errors["val"])
    i_min = int(np.argmin(val))
    if not one_se_rule:
        return errors["degrees"][i_min]
    threshold = val[i_min] + errors["val_std"][i_min]
    for degree, v in zip(errors["degrees"], val):  # степени идут по возрастанию сложности
        if v <= threshold:
            return degree
    return errors["degrees"][i_min]


def learning_curve_errors(
    degree: int,
    sizes: list[int],
    alpha: float = 0.0,
    noise: float = 0.25,
    seed: int = 42,
    n_test: int = 500,
    repeats: int = 30,
) -> dict[str, list[float]]:
    """Кривая обучения: ошибка на обучении и на новых данных в зависимости от размера выборки.

    Для каждого размера выборки эксперимент повторяется `repeats` раз на
    независимых данных, берётся медиана. Тестовая выборка большая и общая,
    поэтому оценка на ней почти не шумит.
    """
    X_test, y_test = make_dataset(n_test, noise, seed + 999_983)
    train_med, test_med = [], []
    for n in sizes:
        tr, te = [], []
        for r in range(repeats):
            X, y = make_dataset(n, noise, seed * 100_003 + r * 1_009 + n)
            model = make_polynomial_model(degree, alpha).fit(X, y)
            tr.append(mean_squared_error(y, model.predict(X)))
            te.append(mean_squared_error(y_test, model.predict(X_test)))
        train_med.append(float(np.median(tr)))
        test_med.append(float(np.median(te)))
    return {"sizes": list(sizes), "train": train_med, "test": test_med}


def alpha_sweep(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    alphas: list[float],
    folds: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Ошибка на валидации для разных alpha (Ridge) и для модели без регуляризации."""
    baseline = cv_errors(X, y, [degree], alpha=0.0, folds=folds, seed=seed)["val"][0]
    val = [cv_errors(X, y, [degree], alpha=a, folds=folds, seed=seed)["val"][0] for a in alphas]
    return {"degree": degree, "alphas": list(alphas), "val": val, "baseline": baseline}
