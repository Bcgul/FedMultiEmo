"""Data loading and federated partitioning for both modalities.

Class indices follow FER2013: 0 angry, 1 disgust, 2 fear, 3 happy, 4 sad,
5 surprise, 6 neutral. The physiological files carry their own numeric
mapping (``Emotion_Mapped``); they are mapped by class *name* instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
# FER+ vote columns -> our class names (contempt, unknown and NF are dropped).
FERPLUS_COLUMNS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
FERPLUS_ALL = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]
SPLITS = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


@dataclass
class ImageSet:
    """48x48 grayscale faces as uint8 plus integer labels."""

    x: np.ndarray  # (N, 48, 48, 1) uint8
    y: np.ndarray  # (N,) int64
    split: np.ndarray  # (N,) 'train' | 'val' | 'test'

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx) -> ImageSet:
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return ImageSet(self.x[idx], self.y[idx], self.split[idx])

    def part(self, split: str) -> ImageSet:
        return self.subset(self.split == split)


# --------------------------------------------------------------------------- #
# FER2013 / FER+
# --------------------------------------------------------------------------- #
def ferplus_majority(votes: pd.DataFrame) -> np.ndarray:
    """FER+ 'majority' labels: the top category must hold more than half of
    the votes and be one of the seven basic emotions; otherwise -1 (dropped)."""
    counts = votes[FERPLUS_ALL].to_numpy(dtype=float)
    total = counts.sum(axis=1)
    top = counts.argmax(axis=1)
    keep = counts.max(axis=1) > 0.5 * total
    labels = np.full(len(votes), -1, dtype=np.int64)
    for j, col in enumerate(FERPLUS_ALL):
        if col in FERPLUS_COLUMNS:
            labels[keep & (top == j)] = CLASS_INDEX[FERPLUS_COLUMNS[col]]
    return labels


def load_fer(cfg: dict, root: str | Path | None = None) -> ImageSet:
    fcfg = cfg["fer"]
    frame = pd.read_csv(resolve_path(fcfg["csv"], root))
    missing = {"emotion", "pixels", "Usage"} - set(frame.columns)
    if missing:
        raise ValueError(f"{fcfg['csv']}: missing columns {sorted(missing)}")
    if fcfg["labels"] == "ferplus":
        votes = pd.read_csv(resolve_path(fcfg["ferplus_csv"], root))
        if len(votes) != len(frame):
            raise ValueError("fer2013new.csv must have one row per FER2013 image")
        labels = ferplus_majority(votes)
    else:
        labels = frame["emotion"].to_numpy(dtype=np.int64)
    keep = labels >= 0
    pixels = frame.loc[keep, "pixels"].to_numpy()
    x = np.stack([np.asarray(p.split(), dtype=np.uint8) for p in pixels]).reshape(-1, 48, 48, 1)
    split = frame.loc[keep, "Usage"].map(SPLITS).to_numpy()
    images = ImageSet(x=x, y=labels[keep], split=split)
    if fcfg.get("subsample"):  # quick runs and tests only
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(images), int(len(images) * float(fcfg["subsample"])), replace=False))
        images = images.subset(idx)
    return images


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def partition(y: np.ndarray, num_clients: int, scheme: str, alpha: float, seed: int) -> np.ndarray:
    """Client index for every sample.

    ``iid``: stratified round-robin, so every client gets the same label mix.
    ``dirichlet``: label skew, client shares per class ~ Dir(alpha)
    (Hsu et al., 2019); smaller alpha = more heterogeneous clients.
    """
    rng = np.random.default_rng(seed)
    owner = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        if scheme == "iid":
            owner[idx] = (np.arange(len(idx)) + rng.integers(num_clients)) % num_clients
        elif scheme == "dirichlet":
            shares = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                owner[chunk] = k
        else:
            raise ValueError(f"unknown partition scheme {scheme!r}")
    return owner


def client_image_sets(images: ImageSet, cfg: dict, seed: int) -> list[dict[str, ImageSet]]:
    """Training and validation images split across clients; the test set is
    kept whole and only used for the final evaluation."""
    fl = cfg["federated"]
    n = int(fl["num_clients"])
    out: list[dict[str, ImageSet]] = [{} for _ in range(n)]
    for split, offset in (("train", 0), ("val", 1)):
        part = images.part(split)
        owner = partition(part.y, n, fl["partition"], float(fl["dirichlet_alpha"]), seed + offset)
        for k in range(n):
            out[k][split] = part.subset(owner == k)
    return out


# --------------------------------------------------------------------------- #
# Physiological data
# --------------------------------------------------------------------------- #
@dataclass
class PhysioSet:
    x: np.ndarray  # (N, n_features) float
    y: np.ndarray  # (N,) int64, FER2013 class order

    def __len__(self) -> int:
        return len(self.y)


def load_physio_file(path: str | Path, features: list[str]) -> PhysioSet:
    frame = pd.read_csv(path)
    missing = [c for c in [*features, "Emotion_Detected"] if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    names = frame["Emotion_Detected"].str.strip().str.lower()
    unknown = sorted(set(names) - set(CLASSES))
    if unknown:
        raise ValueError(f"{path}: unknown emotion labels {unknown}")
    return PhysioSet(x=frame[features].to_numpy(dtype=float), y=names.map(CLASS_INDEX).to_numpy(dtype=np.int64))


def split_physio(data: PhysioSet, test_fraction: float, seed: int) -> tuple[PhysioSet, PhysioSet]:
    """Stratified random train/test split (the files carry no time stamps or
    participant IDs, so a time- or subject-aware split is not possible)."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(data), dtype=bool)
    for c in np.unique(data.y):
        idx = rng.permutation(np.flatnonzero(data.y == c))
        test[idx[: int(round(test_fraction * len(idx)))]] = True
    return PhysioSet(data.x[~test], data.y[~test]), PhysioSet(data.x[test], data.y[test])


def client_physio_sets(cfg: dict, seed: int, root: str | Path | None = None) -> list[dict[str, PhysioSet]]:
    pcfg = cfg["physio"]
    out = []
    for k, path in enumerate(pcfg["client_files"]):
        data = load_physio_file(resolve_path(path, root), list(pcfg["features"]))
        train, test = split_physio(data, float(pcfg["test_fraction"]), seed + k)
        out.append({"train": train, "test": test})
    return out
