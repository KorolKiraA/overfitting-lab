"""Генерация синтетических данных."""

from __future__ import annotations

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """Истинная зависимость, которую модель пытается восстановить."""
    return np.sin(2 * np.pi * x)


def make_dataset(
    n_samples: int = 30,
    noise: float = 0.25,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (X, y): X формы (n, 1), y формы (n,).

    y = sin(2*pi*x) + гауссов шум со стандартным отклонением `noise`.
    """
    if n_samples < 2:
        raise ValueError("n_samples должен быть не меньше 2")
    if noise < 0:
        raise ValueError("noise не может быть отрицательным")

    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n_samples))
    y = true_function(x) + rng.normal(0, noise, n_samples)
    return x.reshape(-1, 1), y
