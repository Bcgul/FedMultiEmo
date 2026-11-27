"""Figures for the README from the outputs of `fedmultiemo experiments`.

python scripts/make_figures.py --results results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
REGIME = {
    "centralized": "Centralised",
    "local": "Local only",
    "fedavg": "FedAvg",
    "federated_forest": "Federated forest",
}
RULE = {
    "face_only": "Face\nonly",
    "physio_only": "Physiology\nonly",
    "vote": "Vote",
    "average": "Average",
    "product": "Product",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _bars(ax, labels, values, errors, title, color) -> None:
    _style(ax)
    x = range(len(labels))
    ax.bar(x, values, 0.6, color=color, zorder=2)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor=INK_2, elinewidth=1, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(x), labels, rotation=0, fontsize=8, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, color=INK, fontsize=10, loc="left")


def overview(results: Path, out: Path) -> None:
    face = pd.DataFrame([json.loads(p.read_text()) for p in sorted(results.glob("face__*.json"))])
    physio = pd.read_csv(results / "physio_runs.csv")
    fusion = pd.read_csv(results / "fusion_runs.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), dpi=150, facecolor=SURFACE, sharey=True)

    f = (
        face.groupby("regime", sort=False)["accuracy"]
        .agg(["mean", "std"])
        .reindex([r for r in REGIME if r in set(face.regime)])
    )
    _bars(axes[0], [REGIME[r] for r in f.index], f["mean"], f["std"].fillna(0), "Facial CNN, FER+ test set", SERIES[0])
    p = physio.groupby("regime", sort=False)["accuracy"].agg(["mean", "std"])
    _bars(axes[1], [REGIME[r] for r in p.index], p["mean"], p["std"], "Physiological RF, random split", SERIES[1])
    fu = fusion[fusion.face_regime == "fedavg"].groupby("rule", sort=False)["accuracy"].agg(["mean", "std"])
    _bars(
        axes[2],
        [RULE[r] for r in fu.index],
        fu["mean"],
        fu["std"],
        "Fusion (FedAvg CNN + federated forest),\nlabel-matched pairs",
        SERIES[2],
    )
    axes[0].set_ylabel("Accuracy", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "accuracy_overview.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def rounds(results: Path, out: Path) -> None:
    fed = results / "face__fedavg__seed0_history.csv"
    cen = results / "face__centralized__seed0_history.csv"
    if not fed.exists():
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    h = pd.read_csv(fed)
    ax.plot(
        h["round"] * 2,
        h["val_loss"],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=4,
        label="FedAvg, 3 clients (2 local epochs per round)",
    )
    if cen.exists():
        c = pd.read_csv(cen)
        ax.plot(c["epoch"], c["val_loss"], color=SERIES[1], linewidth=2, label="Centralised")
    loc = results / "face__local__seed0_history.csv"
    if loc.exists():
        lh = pd.read_csv(loc)
        ax.plot(lh["round"] * 2, lh["val_loss"], color=SERIES[2], linewidth=2, label="Local only (mean of clients)")
    ax.set_xlabel("Training epochs (federated: rounds x 2 local epochs)", color=INK_2, fontsize=9)
    ax.set_ylabel("Validation loss", color=INK_2, fontsize=9)
    ax.set_title("Facial CNN: validation loss during training", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "validation_loss.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    overview(args.results, out)
    rounds(args.results, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
