"""Decision-level fusion of the facial and physiological classifiers.

No recording in this release contains a face image and a physiological sample
of the same person at the same moment. Fusion is therefore evaluated on
*label-matched pairs*: every facial test image is paired with a randomly drawn
physiological test sample of the same emotion. This measures how two
independent classifiers combine; it is not a measurement on real multimodal
data, and it favours fusion because the two modalities' errors are
independent by construction.
"""

from __future__ import annotations

import numpy as np

from .data import CLASSES


def label_matched_pairs(y_face: np.ndarray, y_physio: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices (face_idx, physio_idx) pairing every face sample with a
    physiological sample of the same class (drawn with replacement). Face
    samples of classes absent from the physiological set are skipped."""
    rng = np.random.default_rng(seed)
    face_idx, physio_idx = [], []
    for c in range(len(CLASSES)):
        f = np.flatnonzero(y_face == c)
        p = np.flatnonzero(y_physio == c)
        if len(f) == 0 or len(p) == 0:
            continue
        face_idx.append(f)
        physio_idx.append(rng.choice(p, size=len(f), replace=True))
    return np.concatenate(face_idx), np.concatenate(physio_idx)


def fuse(p_face: np.ndarray, p_physio: np.ndarray, rule: str, visual_weight: float = 0.5) -> np.ndarray:
    """Fused class probabilities (rows sum to 1).

    ``vote``   : majority vote of the two hard decisions; a disagreement is a
                 tie between two voters and goes to the modality with the
                 higher (weighted) confidence.
    ``average``: w * p_face + (1 - w) * p_physio.
    ``product``: p_face^w * p_physio^(1-w), renormalised.
    """
    w = float(visual_weight)
    if rule == "average":
        return w * p_face + (1 - w) * p_physio
    if rule == "product":
        logp = w * np.log(np.clip(p_face, 1e-12, None)) + (1 - w) * np.log(np.clip(p_physio, 1e-12, None))
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)
    if rule == "vote":
        face_wins = w * p_face.max(axis=1) >= (1 - w) * p_physio.max(axis=1)
        winner = np.where(face_wins, p_face.argmax(axis=1), p_physio.argmax(axis=1))
        out = np.zeros_like(p_face)
        out[np.arange(len(winner)), winner] = 1.0
        return out
    raise ValueError(f"unknown fusion rule {rule!r}")
