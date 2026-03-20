import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZES = [50, 500]


def normal_density(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def uniform_density(x, a, b):
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)


def sel_base_density(x, form):
    return (
        form * (form + 1.0 + np.abs(x))
        / (2.0 * (form + np.abs(x)) ** 2)
        * np.exp(-np.abs(x))
    )


def sel_density(x, shift, scale, form):
    return sel_base_density((x - shift) / scale, form) / scale


def load_histogram_csv(filename):
    bin_left = []
    bin_right = []
    emp_density = []

    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            bin_left.append(float(row["bin_left"]))
            bin_right.append(float(row["bin_right"]))
            emp_density.append(float(row["emp_density"]))

    return np.array(bin_left), np.array(bin_right), np.array(emp_density)


def save_plot(filename, csv_path, density_fn, x_min, x_max, title):
    bin_left, bin_right, emp_density = load_histogram_csv(csv_path)

    x = np.linspace(x_min, x_max, 1000)
    y = density_fn(x)
    widths = bin_right - bin_left

    plt.figure(figsize=(9, 5))
    plt.bar(
        bin_left,
        emp_density,
        width=widths,
        align="edge",
        alpha=0.55,
        color="#7aa6c2",
        edgecolor="black",
        label="Эмпирическая плотность",
    )
    plt.plot(x, y, color="#b03a2e", linewidth=2.0, label="Теоретическая плотность")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()


def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    for size in SIZES:
        save_plot(
            out_dir / f"normal_hist_{size}.png",
            Path(f"Normal(0,1)_histogram_data_n{size}.csv"),
            lambda x: normal_density(x, 0.0, 1.0),
            -4.0,
            4.0,
            f"Normal(0, 1): гистограмма и теоретическая плотность, n = {size}",
        )

        save_plot(
            out_dir / f"uniform_hist_{size}.png",
            Path(f"Uniform(-1,3)_histogram_data_n{size}.csv"),
            lambda x: uniform_density(x, -1.0, 3.0),
            -1.5,
            3.5,
            f"Uniform(-1, 3): гистограмма и теоретическая плотность, n = {size}",
        )

        save_plot(
            out_dir / f"sel_hist_{size}.png",
            Path(
                f"ShiftedExponentialLaplace(shift=0.5, scale=1.2, form=2.0)_histogram_data_n{size}.csv"
            ),
            lambda x: sel_density(x, 0.5, 1.2, 2.0),
            -3.0,
            4.0,
            f"ShiftedExponentialLaplace: гистограмма и теоретическая плотность, n = {size}",
        )


if __name__ == "__main__":
    main()
