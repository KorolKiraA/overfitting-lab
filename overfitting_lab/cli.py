"""Командная строка: python -m overfitting_lab  (или команда overfitting-lab)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__
from .data import make_dataset
from .models import alpha_sweep, best_degree, cv_errors, learning_curve_errors
from .plots import plot_alpha_sweep, plot_error_curves, plot_fits, plot_learning_curves


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overfitting_lab",
        description="Наглядная лаборатория переобучения на полиномиальной регрессии.",
    )
    p.add_argument("--samples", type=int, default=30, help="число точек в данных (по умолчанию 30)")
    p.add_argument("--noise", type=float, default=0.25, help="стандартное отклонение шума (0.25)")
    p.add_argument("--max-degree", type=int, default=15, help="максимальная степень полинома (15)")
    p.add_argument("--complex-degree", type=int, default=12,
                   help="'слишком сложная' степень для кривых обучения и регуляризации (12)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="сила Ridge-регуляризации для основных графиков (0 = выключена)")
    p.add_argument("--folds", type=int, default=5, help="число фолдов кросс-валидации (5)")
    p.add_argument("--seed", type=int, default=42, help="seed для воспроизводимости (42)")
    p.add_argument("--out", type=Path, default=Path("results"), help="папка для результатов (results)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_degree < 2:
        raise SystemExit("--max-degree должен быть не меньше 2")
    if args.complex_degree < 1:
        raise SystemExit("--complex-degree должен быть не меньше 1")
    args.out.mkdir(parents=True, exist_ok=True)

    X, y = make_dataset(args.samples, args.noise, args.seed)

    # 1. Кросс-валидация по степеням
    degrees = list(range(1, args.max_degree + 1))
    errors = cv_errors(X, y, degrees, alpha=args.alpha, folds=args.folds)
    best = best_degree(errors)

    print(f"Данные: {args.samples} точек, шум {args.noise}, {args.folds}-fold CV, alpha={args.alpha:g}\n")
    print(f"{'степень':>8} {'train MSE':>12} {'val MSE':>14}")
    for d, tr, va in zip(errors["degrees"], errors["train"], errors["val"]):
        mark = "  <- лучшая" if d == best else ""
        print(f"{d:>8} {tr:>12.4f} {va:>14.4f}{mark}")

    # 2. Кривые обучения: простая (лучшая) и слишком сложная модель
    # размеры выборки: плотнее в начале, где разрыв между кривыми виден лучше всего
    start = max(2 * args.complex_degree, 20)
    sizes = sorted({int(s) for s in np.geomspace(start, 200, 8)})
    curves = {
        degree: learning_curve_errors(degree, sizes, alpha=args.alpha, noise=args.noise, seed=args.seed)
        for degree in dict.fromkeys([best, args.complex_degree])  # без дублей, порядок сохраняется
    }

    # 3. Влияние силы регуляризации
    alphas = [float(a) for a in np.logspace(-6, 2, 17)]
    sweep = alpha_sweep(X, y, args.complex_degree, alphas, folds=args.folds)
    best_alpha = alphas[int(np.argmin(sweep["val"]))]

    # 4. Графики и сводка
    plot_fits(X, y, sorted({1, best, args.max_degree}), args.alpha, args.out / "fits.png")
    plot_error_curves(errors, args.out / "error_curves.png")
    plot_learning_curves(curves, args.noise, args.out / "learning_curves.png")
    plot_alpha_sweep(sweep, args.out / "regularization.png")

    summary = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "best_degree": best,
        "best_alpha_for_complex_model": best_alpha,
        "cv_errors": errors,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nЛучшая степень по валидации: {best}")
    print(f"Для степени {args.complex_degree} лучшая сила регуляризации: alpha ≈ {best_alpha:.2g}")
    print(f"  ошибка без регуляризации: {sweep['baseline']:.4f}")
    print(f"  ошибка с лучшей alpha:    {min(sweep['val']):.4f}")
    print(f"\nРезультаты сохранены в: {args.out.resolve()}")
    return 0
