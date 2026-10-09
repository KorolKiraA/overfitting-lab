"""Построение графиков (сохраняются в PNG, окно не открывается)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без GUI: работает и на сервере, и в CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import true_function  # noqa: E402
from .models import make_polynomial_model  # noqa: E402


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_fits(X, y, degrees: list[int], alpha: float, out: Path) -> None:
    """Подгонки разной сложности на одних и тех же данных."""
    grid = np.linspace(0, 1, 400).reshape(-1, 1)
    fig, axes = plt.subplots(1, len(degrees), figsize=(5 * len(degrees), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, d in zip(axes, degrees):
        model = make_polynomial_model(d, alpha).fit(X, y)
        ax.scatter(X, y, color="tab:blue", s=25, label="данные")
        ax.plot(grid, true_function(grid.ravel()), "g--", label="истина")
        ax.plot(grid, model.predict(grid), "r-", label=f"полином {d}")
        ax.set_ylim(-2, 2)
        ax.set_title(f"Степень {d}")
        ax.legend(loc="upper right", fontsize=8)

    title = "Недообучение, норма, переобучение"
    if alpha > 0:
        title += f" (Ridge, alpha={alpha:g})"
    fig.suptitle(title)
    _save(fig, out)


def plot_error_curves(errors: dict, out: Path) -> None:
    """Ошибка на обучении и на валидации (k-fold) в зависимости от сложности модели."""
    d = np.array(errors["degrees"])
    val = np.array(errors["val"])
    std = np.array(errors["val_std"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d, errors["train"], "o-", label="обучение")
    ax.plot(d, val, "s-", label="валидация (среднее по фолдам)")
    # на лог-шкале нижняя граница не может уйти в минус: ограничиваем её одной декадой ниже линии
    lower = np.maximum(val - std, val / 10)
    ax.fill_between(d, lower, val + std, color="tab:orange", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Степень полинома (сложность модели)")
    ax.set_ylabel("MSE (логарифмическая шкала)")
    ax.set_title("Главный признак переобучения: ошибки расходятся")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_learning_curves(curves: dict[int, dict], noise: float, out: Path) -> None:
    """Кривые обучения для нескольких степеней: ошибка против размера выборки."""
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (degree, c) in zip(axes, curves.items()):
        ax.plot(c["sizes"], c["train"], "o-", label="обучение")
        ax.plot(c["sizes"], c["test"], "s-", label="новые данные")
        ax.axhline(noise**2, color="gray", ls="--", label="уровень шума")
        ax.set_yscale("log")
        ax.set_title(f"Степень {degree}")
        ax.set_xlabel("Размер обучающей выборки")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MSE (медиана, лог. шкала)")
    fig.suptitle("Кривые обучения: больше данных сужает разрыв")
    _save(fig, out)


def plot_alpha_sweep(sweep: dict, out: Path) -> None:
    """Ошибка на валидации в зависимости от силы регуляризации."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep["alphas"], sweep["val"], "o-", label=f"Ridge, степень {sweep['degree']}")
    ax.axhline(sweep["baseline"], color="red", ls="--", label="без регуляризации")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("alpha (сила регуляризации)")
    ax.set_ylabel("MSE на валидации (лог. шкала)")
    ax.set_title("Регуляризация лечит переобучение")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)
