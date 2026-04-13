from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc


plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35


# =========================
# Теоретические плотности
# =========================

GAMMA_13 = gamma(1.0 / 3.0)


def lower_incomplete_gamma_two_thirds(x):
    """
    γ(2/3, x) = Γ(2/3) * gammainc(2/3, x)
    """
    if np.isscalar(x):
        if x <= 0.0:
            return 0.0
        return gamma(2.0 / 3.0) * gammainc(2.0 / 3.0, x)

    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0.0
    out[mask] = gamma(2.0 / 3.0) * gammainc(2.0 / 3.0, x[mask])
    return out


def ugg_density_scalar(x, shift, scale, form):
    """
    Плотность Shifted U-GG из твоей формулы.
    """
    z = (x - shift) / scale
    abs_z = abs(z)

    if form == 0.0:
        base = 3.0 * math.exp(-(abs_z ** 3.0)) / (2.0 * GAMMA_13)
    elif abs_z == 0.0:
        base = 3.0 * (2.0 - form) / (4.0 * GAMMA_13)
    else:
        x_cube = abs_z ** 3.0
        lower = ((1.0 - form) * abs_z) ** 3.0
        numerator = (
            lower_incomplete_gamma_two_thirds(x_cube)
            - lower_incomplete_gamma_two_thirds(lower)
        )
        base = numerator / (2.0 * GAMMA_13 * form * abs_z * abs_z)

    return base / scale


def ugg_density(x, shift, scale, form):
    x = np.asarray(x, dtype=float)
    return np.array([ugg_density_scalar(value, shift, scale, form) for value in x])


def normal_density(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def uniform_density(x, a, b):
    x = np.asarray(x, dtype=float)
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)


def build_theoretical_density(distribution_name, x):
    if distribution_name == "Normal":
        return normal_density(x, 0.0, 1.0)

    if distribution_name == "Uniform":
        return uniform_density(x, -1.0, 3.0)

    if distribution_name == "ShiftedUGG":
        return ugg_density(x, shift=0.5, scale=1.2, form=0.5)

    raise ValueError(f"Unknown distribution: {distribution_name}")


# =========================
# Генерация выборок
# =========================

def sample_generalized_gaussian_exp_abs_cube(size, shift=0.0, scale=1.0):
    """
    Генерация из плотности:
        f(z) = 3 * exp(-|z|^3) / (2 * Gamma(1/3))
    затем affine-преобразование x = shift + scale * z

    Идея:
    Если T ~ Gamma(shape=1/3, scale=1), то |Z| = T^(1/3),
    а знак берётся равновероятно ±1.
    """
    t = np.random.gamma(shape=1.0 / 3.0, scale=1.0, size=size)
    r = np.cbrt(t)
    signs = np.random.choice([-1.0, 1.0], size=size)
    z = signs * r
    return shift + scale * z


def proposal_density_shifted_base(x, shift, scale):
    """
    Плотность базового proposal-распределения:
        Z ~ exp(-|z|^3)-семейство
        X = shift + scale * Z
    """
    z = (np.asarray(x) - shift) / scale
    base = 3.0 * np.exp(-(np.abs(z) ** 3.0)) / (2.0 * GAMMA_13)
    return base / scale


def estimate_rejection_M(shift, scale, form, z_max=12.0, grid_size=40000, safety=1.10):
    """
    Оцениваем константу M для rejection sampling:
        target(x) <= M * proposal(x)

    Делаем плотную сетку по z и берём запас safety.
    """
    z_grid = np.linspace(-z_max, z_max, grid_size)
    x_grid = shift + scale * z_grid

    target = ugg_density(x_grid, shift=shift, scale=scale, form=form)
    proposal = proposal_density_shifted_base(x_grid, shift=shift, scale=scale)

    ratio = np.divide(
        target,
        proposal,
        out=np.zeros_like(target),
        where=proposal > 0
    )

    m = np.max(ratio)
    return float(max(1.0, m * safety))


def sample_shifted_ugg(size, shift=0.5, scale=1.2, form=0.5, batch_size=4096):
    """
    Более корректная генерация ShiftedUGG:
    rejection sampling из базового exp(-|z|^3)-proposal.

    Это не аппроксимация через linspace — здесь реально генерируются случайные числа
    из целевой плотности, заданной формулой ugg_density_scalar.
    """
    if size <= 0:
        return np.array([], dtype=float)

    M = estimate_rejection_M(shift=shift, scale=scale, form=form)

    accepted = []
    total = 0

    while total < size:
        current_batch = min(batch_size, size - total + batch_size)

        candidates = sample_generalized_gaussian_exp_abs_cube(
            current_batch, shift=shift, scale=scale
        )

        target_vals = ugg_density(candidates, shift=shift, scale=scale, form=form)
        proposal_vals = proposal_density_shifted_base(candidates, shift=shift, scale=scale)

        acceptance_prob = target_vals / (M * proposal_vals)
        acceptance_prob = np.clip(acceptance_prob, 0.0, 1.0)

        uniforms = np.random.random(size=current_batch)
        chosen = candidates[uniforms <= acceptance_prob]

        if len(chosen) > 0:
            accepted.append(chosen)
            total += len(chosen)

    return np.concatenate(accepted)[:size]


def generate_sample(distribution_name, size):
    if distribution_name == "Normal":
        return np.array([random.normalvariate(0.0, 1.0) for _ in range(size)], dtype=float)

    if distribution_name == "Uniform":
        return np.array([random.uniform(-1.0, 3.0) for _ in range(size)], dtype=float)

    if distribution_name == "ShiftedUGG":
        return sample_shifted_ugg(size=size, shift=0.5, scale=1.2, form=0.5)

    raise ValueError(f"Unknown distribution: {distribution_name}")


# =========================
# Эмпирическая гистограмма
# =========================

def build_histogram_dataframe(sample, bins=30):
    density, edges = np.histogram(sample, bins=bins, density=True)

    data = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "emp_density": density,
    })
    data["center"] = (data["bin_left"] + data["bin_right"]) / 2.0
    data["width"] = data["bin_right"] - data["bin_left"]
    return data


# =========================
# Отрисовка
# =========================

def plot_distribution_panel(ax, distribution_name, title_name, sample_size, bins=30):
    sample = generate_sample(distribution_name, sample_size)
    data = build_histogram_dataframe(sample, bins=bins)

    x_min = data["bin_left"].min()
    x_max = data["bin_right"].max()
    span = x_max - x_min
    padding = 0.25 * span if span > 0.0 else 1.0

    curve_x = np.linspace(x_min - padding, x_max + padding, 600)
    curve_y = build_theoretical_density(distribution_name, curve_x)

    theory_point_count = 120 if sample_size >= 500 else 70
    theory_idx = np.linspace(0, len(curve_x) - 1, theory_point_count, dtype=int)
    theory_x = curve_x[theory_idx]
    theory_y = curve_y[theory_idx]

    empirical_x = data["center"].to_numpy()
    empirical_y = data["emp_density"].to_numpy()

    ax.plot(
        curve_x,
        curve_y,
        color="#26d7e6",
        linewidth=1.8,
        zorder=2,
        label="Теоретическая плотность",
    )
    ax.scatter(
        theory_x,
        theory_y,
        s=12,
        color="#1f2cff",
        zorder=3,
        label="Точки теории",
    )
    ax.scatter(
        empirical_x,
        empirical_y,
        s=18,
        color="#ff2020",
        zorder=4,
        label="Точки эмпирики",
    )

    ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#d9d9d9", linewidth=0.8, zorder=1)
    ax.set_title(f"{title_name}, n = {sample_size}")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность")


def plot_distribution_figure(distribution_name, title_name, sample_sizes, bins=30):
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(8 * len(sample_sizes), 6), squeeze=False)
    axes = axes[0]

    for ax, n in zip(axes, sample_sizes):
        plot_distribution_panel(ax, distribution_name, title_name, n, bins=bins)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        f"{title_name}: эмпирические точки и теоретическая кривая",
        y=1.08,
        fontsize=15,
    )
    fig.tight_layout()

    output_name = f"{distribution_name.lower()}_points_vs_theory.png"
    fig.savefig(output_name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Сохранен график: {output_name}")


# =========================
# Main
# =========================

def main():
    np.random.seed(42)
    random.seed(42)

    configs = [
        ("Normal", "Normal(0, 1)", [100, 500, 1000]),
        ("Uniform", "Uniform(-1, 3)", [100, 500, 1000]),
        ("ShiftedUGG", "ShiftedUGG(shift=0.5, scale=1.2, form=0.5)", [100, 500, 1000]),
    ]

    for distribution_name, title_name, sample_sizes in configs:
        plot_distribution_figure(
            distribution_name=distribution_name,
            title_name=title_name,
            sample_sizes=sample_sizes,
            bins=30,
        )


if __name__ == "__main__":
    main()