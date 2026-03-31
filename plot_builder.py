from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35


def lower_incomplete_gamma_two_thirds(x, steps=2000):
    if x <= 0.0:
        return 0.0

    upper = x ** (1.0 / 3.0)
    h = upper / steps

    def integrand(u):
        return 3.0 * u * math.exp(-(u ** 3))

    total = integrand(0.0) + integrand(upper)
    for i in range(1, steps):
        u = i * h
        total += (4.0 if i % 2 else 2.0) * integrand(u)

    return total * h / 3.0


def ugg_density_scalar(x, shift, scale, form):
    z = (x - shift) / scale
    gamma13 = math.gamma(1.0 / 3.0)
    abs_z = abs(z)

    if form == 0.0:
        base = 3.0 * math.exp(-(abs_z ** 3.0)) / (2.0 * gamma13)
    elif abs_z == 0.0:
        base = 3.0 * (2.0 - form) / (4.0 * gamma13)
    else:
        x_cube = abs_z ** 3.0
        lower = ((1.0 - form) * abs_z) ** 3.0
        numerator = lower_incomplete_gamma_two_thirds(x_cube) - lower_incomplete_gamma_two_thirds(lower)
        base = numerator / (2.0 * gamma13 * form * abs_z * abs_z)

    return base / scale


def normal_density(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def uniform_density(x, a, b):
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)


def ugg_density(x, shift, scale, form):
    return np.array([ugg_density_scalar(value, shift, scale, form) for value in x])


def load_histogram(path):
    data = pd.read_csv(path)
    data["center"] = (data["bin_left"] + data["bin_right"]) / 2.0
    data["width"] = data["bin_right"] - data["bin_left"]
    return data


def load_params(path):
    return [float(value) for value in path.read_text(encoding="utf-8").split()]


def build_theoretical_density(distribution_name, x):
    if distribution_name == "Normal":
        mu, sigma = load_params(Path("normal_distribution_params.txt"))
        return normal_density(x, mu, sigma)

    if distribution_name == "Uniform":
        a, b = load_params(Path("uniform_distribution_params.txt"))
        return uniform_density(x, a, b)

    if distribution_name == "ShiftedUGG":
        shift, scale, form = load_params(Path("shifted_ugg_distribution_params.txt"))
        return ugg_density(x, shift, scale, form)

    raise ValueError(f"Unknown distribution: {distribution_name}")


def parse_distribution_info(path):
    name = path.stem
    sample_match = re.search(r"_n(\d+)$", name)
    sample_size = int(sample_match.group(1)) if sample_match else 0

    if name.startswith("Normal"):
        return "Normal", "Normal(0, 1)", sample_size
    if name.startswith("Uniform"):
        return "Uniform", "Uniform(-1, 3)", sample_size
    if name.startswith("ShiftedUGG"):
        return "ShiftedUGG", "ShiftedUGG(shift=0.5, scale=1.2, form=0.5)", sample_size

    return None, name, sample_size


def build_empirical_points(data, points_per_bin):
    x_parts = []
    y_parts = []

    for _, row in data.iterrows():
        x_parts.append(np.linspace(row["bin_left"], row["bin_right"], points_per_bin))
        y_parts.append(np.full(points_per_bin, row["emp_density"]))

    return np.concatenate(x_parts), np.concatenate(y_parts)


def plot_distribution_panel(ax, path):
    distribution_name, title_name, sample_size = parse_distribution_info(path)
    if distribution_name is None:
        return

    data = load_histogram(path)
    x_min = data["bin_left"].min()
    x_max = data["bin_right"].max()
    span = x_max - x_min
    padding = 0.25 * span if span > 0.0 else 1.0

    curve_x = np.linspace(x_min - padding, x_max + padding, 500)
    curve_y = build_theoretical_density(distribution_name, curve_x)

    theory_point_count = 120 if sample_size >= 500 else 70
    theory_idx = np.linspace(0, len(curve_x) - 1, theory_point_count, dtype=int)
    theory_x = curve_x[theory_idx]
    theory_y = curve_y[theory_idx]

    empirical_x, empirical_y = build_empirical_points(data, 24 if sample_size >= 500 else 12)

    ax.plot(curve_x, curve_y, color="#26d7e6", linewidth=1.8, zorder=2, label="Теоретическая плотность")
    ax.scatter(theory_x, theory_y, s=12, color="#1f2cff", zorder=3, label="Точки теории")
    ax.scatter(empirical_x, empirical_y, s=10, color="#ff2020", zorder=4, label="Точки эмпирики")

    ax.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
    ax.axvline(0.0, color="#d9d9d9", linewidth=0.8, zorder=1)
    ax.set_title(f"{title_name}, n = {sample_size}")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность")


def plot_distribution_figure(paths):
    ordered_paths = sorted(paths, key=lambda item: parse_distribution_info(item)[2])
    distribution_name, title_name, _ = parse_distribution_info(ordered_paths[0])
    if distribution_name is None:
        return

    fig, axes = plt.subplots(1, len(ordered_paths), figsize=(8 * len(ordered_paths), 6), squeeze=False)
    axes = axes[0]

    for ax, path in zip(axes, ordered_paths):
        plot_distribution_panel(ax, path)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{title_name}: точки по бинам и теоретическая кривая", y=1.08, fontsize=15)
    fig.tight_layout()

    output_name = f"{distribution_name.lower()}_points_vs_theory.png"
    fig.savefig(output_name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Сохранен график: {output_name}")


def main():
    csv_files = sorted(Path(".").glob("*_histogram_data_n*.csv"))
    if not csv_files:
        print("Не найдены файлы гистограмм.")
        return

    grouped = {}
    for path in csv_files:
        if path.stem.startswith("ShiftedExponentialLaplace"):
            continue

        distribution_name, _, _ = parse_distribution_info(path)
        if distribution_name is None:
            continue
        grouped.setdefault(distribution_name, []).append(path)

    for paths in grouped.values():
        plot_distribution_figure(paths)


if __name__ == "__main__":
    main()
