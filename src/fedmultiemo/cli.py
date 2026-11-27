"""Command-line interface.

    fedmultiemo face        train the facial CNN (centralized | fedavg | local)
    fedmultiemo physio      Random Forest regimes on the physiological data
    fedmultiemo experiments everything above plus decision-level fusion

Common options: ``--config file.yaml`` and ``--set key.subkey=value``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import load_config
from .data import client_image_sets, client_physio_sets, load_fer
from .fusion import fuse, label_matched_pairs
from .train import classification_metrics, cnn_centralized, cnn_federated, rf_regimes

FACE_REGIMES = ("centralized", "fedavg", "local")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML file merged onto the packaged defaults")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--output", help="output directory (default: experiment.output_dir)")


def _out(cfg: dict, args) -> Path:
    out = Path(args.output or cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def _environment() -> dict:
    import flwr
    import sklearn
    import tensorflow as tf

    return {
        "fedmultiemo": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "flwr": flwr.__version__,
        "scikit-learn": sklearn.__version__,
    }


def run_face(cfg: dict, regime: str, seed: int, out: Path, images=None, verbose: int = 2):
    images = images if images is not None else load_fer(cfg)
    if regime == "centralized":
        result = cnn_centralized(images, cfg, seed, verbose=verbose)
    else:
        clients = client_image_sets(images, cfg, seed)
        result = cnn_federated(images, clients, cfg, seed, local_only=(regime == "local"), verbose=verbose)
    tag = f"face__{regime}__seed{seed}"
    np.savez_compressed(out / f"{tag}.npz", proba=result.proba, y=images.part("test").y)
    if result.weights is not None:
        np.savez_compressed(out / f"{tag}_weights.npz", **{k.replace("/", "__"): v for k, v in result.weights.items()})
    pd.DataFrame(result.history).to_csv(out / f"{tag}_history.csv", index=False)
    (out / f"{tag}.json").write_text(json.dumps({**result.row(), "confusion": result.metrics["confusion"]}, indent=2))
    print(
        f"{tag}: accuracy {result.metrics['accuracy']:.4f} macro-F1 {result.metrics['macro_f1']:.4f} "
        f"(best step {result.best_step}, {result.train_seconds:.0f}s)",
        flush=True,
    )
    return result


def run_physio(cfg: dict, seeds: list[int], out: Path):
    rows, probas = [], {}
    for seed in seeds:
        results, test = rf_regimes(client_physio_sets(cfg, seed), cfg, seed)
        for r in results:
            rows.append(r.row())
            probas[(r.regime, seed)] = (r.proba, test.y)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "physio_runs.csv", index=False)
    for (regime, seed), (proba, y) in probas.items():
        np.savez_compressed(out / f"physio__{regime}__seed{seed}.npz", proba=proba, y=y)
    print(frame.groupby("regime")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4), flush=True)
    return frame


def run_fusion(cfg: dict, out: Path, face_files: list[Path], physio_regime: str = "federated_forest") -> pd.DataFrame:
    rows = []
    for face_file in face_files:
        face = np.load(face_file)
        face_regime, face_seed = face_file.stem.split("__")[1], int(face_file.stem.split("seed")[-1])
        for physio_file in sorted(out.glob(f"physio__{physio_regime}__seed*.npz")):
            physio = np.load(physio_file)
            seed = int(physio_file.stem.split("seed")[-1])
            fi, pi = label_matched_pairs(face["y"], physio["y"], seed)
            y = face["y"][fi]
            pf, pp = face["proba"][fi], physio["proba"][pi]
            base = {
                "face_regime": face_regime,
                "face_seed": face_seed,
                "physio_regime": physio_regime,
                "pair_seed": seed,
            }
            rows.append({**base, "rule": "face_only", **_flat(classification_metrics(y, pf))})
            rows.append({**base, "rule": "physio_only", **_flat(classification_metrics(y, pp))})
            for rule in cfg["fusion"]["rules"]:
                fused = fuse(pf, pp, rule, cfg["fusion"]["visual_weight"])
                rows.append({**base, "rule": rule, **_flat(classification_metrics(y, fused))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "fusion_runs.csv", index=False)
    if len(frame):
        print(
            frame.groupby(["face_regime", "rule"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4), flush=True
        )
    return frame


def _flat(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "confusion"}


def summary_markdown(out: Path) -> str:
    lines = [
        "## Facial CNN (FER test set)",
        "",
        "| Regime | Seeds | Accuracy | Macro-F1 | Best round/epoch | Train time (min) |",
        "|---|---|---|---|---|---|",
    ]
    face = [json.loads(p.read_text()) for p in sorted(out.glob("face__*.json"))]
    if face:
        f = pd.DataFrame(face)
        for regime, g in f.groupby("regime", sort=False):
            lines.append(
                f"| {regime} | {len(g)} | {_ms(g.accuracy)} | {_ms(g.macro_f1)} | "
                f"{g.best_step.mean():.0f} | {g.train_seconds.mean() / 60:.0f} |"
            )
    p = out / "physio_runs.csv"
    if p.exists():
        f = pd.read_csv(p)
        lines += [
            "",
            "## Physiological Random Forest (union of the clients' test splits)",
            "",
            "| Regime | Seeds | Accuracy | Macro-F1 |",
            "|---|---|---|---|",
        ]
        for regime, g in f.groupby("regime", sort=False):
            lines.append(f"| {regime} | {len(g)} | {_ms(g.accuracy)} | {_ms(g.macro_f1)} |")
    p = out / "fusion_runs.csv"
    if p.exists():
        f = pd.read_csv(p)
        lines += [
            "",
            "## Decision-level fusion on label-matched pairs",
            "",
            "| Facial model | Rule | Accuracy | Macro-F1 |",
            "|---|---|---|---|",
        ]
        for (fr, rule), g in f.groupby(["face_regime", "rule"], sort=False):
            lines.append(f"| {fr} | {rule} | {_ms(g.accuracy)} | {_ms(g.macro_f1)} |")
    return "\n".join(lines) + "\n"


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3f}" if len(s) < 2 else f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"


# --------------------------------------------------------------------------- #
def cmd_face(args):
    cfg = load_config(args.config, args.overrides)
    out = _out(cfg, args)
    run_face(cfg, args.regime, cfg["seed"] if args.seed is None else args.seed, out)


def cmd_physio(args):
    cfg = load_config(args.config, args.overrides)
    run_physio(cfg, args.seeds, _out(cfg, args))


def cmd_fusion(args):
    cfg = load_config(args.config, args.overrides)
    out = _out(cfg, args)
    run_fusion(cfg, out, sorted(p for p in out.glob("face__*__seed*.npz") if not p.stem.endswith("_weights")))
    (out / "summary.md").write_text(summary_markdown(out), encoding="utf-8")


def cmd_experiments(args):
    cfg = load_config(args.config, args.overrides)
    out = _out(cfg, args)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
    start = time.perf_counter()
    run_physio(cfg, args.physio_seeds, out)
    images = load_fer(cfg)
    for regime in args.face_regimes:
        for seed in args.face_seeds:
            run_face(cfg, regime, seed, out, images=images)
    run_fusion(cfg, out, sorted(p for p in out.glob("face__*__seed*.npz") if not p.stem.endswith("_weights")))
    (out / "summary.md").write_text(summary_markdown(out), encoding="utf-8")
    print(f"done in {(time.perf_counter() - start) / 60:.0f} min\n")
    print(summary_markdown(out))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fedmultiemo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"fedmultiemo {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("face", help="train and test the facial CNN")
    _add_common(p)
    p.add_argument("--regime", choices=FACE_REGIMES, default="fedavg")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_face)

    p = sub.add_parser("physio", help="Random Forest regimes on the physiological data")
    _add_common(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.set_defaults(func=cmd_physio)

    p = sub.add_parser("fusion", help="fuse saved test predictions of both modalities")
    _add_common(p)
    p.set_defaults(func=cmd_fusion)

    p = sub.add_parser("experiments", help="physiology, face and fusion in one run")
    _add_common(p)
    p.add_argument("--face-regimes", nargs="+", choices=FACE_REGIMES, default=list(FACE_REGIMES))
    p.add_argument("--face-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--physio-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.set_defaults(func=cmd_experiments)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
