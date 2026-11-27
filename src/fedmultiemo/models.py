"""Facial CNN, physiological Random Forest and their federated aggregation."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from .data import CLASSES  # noqa: E402

N_CLASSES = len(CLASSES)


# --------------------------------------------------------------------------- #
# Facial CNN (architecture of the original client code)
# --------------------------------------------------------------------------- #
def build_cnn(cfg: dict):
    ccfg = cfg["cnn"]
    inputs = keras.Input(shape=(48, 48, 1), name="face")
    x = inputs
    if ccfg["augment"]:  # active only during training
        x = layers.RandomFlip("horizontal", name="aug_flip")(x)
        x = layers.RandomRotation(10.0 / 360.0, fill_mode="nearest", name="aug_rotate")(x)
    x = layers.Conv2D(32, 3, activation="relu", name="conv1")(x)
    x = layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(name="pool2")(x)
    x = layers.Conv2D(128, 3, activation="relu", name="conv4")(x)
    x = layers.MaxPooling2D(name="pool3")(x)
    x = layers.Dropout(0.25, name="drop2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(1024, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="drop3")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="head")(x)
    model = keras.Model(inputs, outputs, name="fedmultiemo_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(float(ccfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_named_weights(model) -> dict[str, np.ndarray]:
    """Trainable weights only (augmentation and dropout seed states excluded)."""
    return {f"{layer.name}/{i}": np.asarray(v) for layer in model.layers for i, v in enumerate(layer.trainable_weights)}


def set_named_weights(model, named: dict[str, np.ndarray]) -> None:
    for layer in model.layers:
        for i, var in enumerate(layer.trainable_weights):
            key = f"{layer.name}/{i}"
            if key not in named:
                raise KeyError(f"missing weights {key!r}")
            var.assign(named[key])


def reset_optimizer(model) -> None:
    """Fresh optimiser state (step counter and Adam moments), as a stateless
    Flower client has at the start of every round."""
    optimizer = model.optimizer
    if not getattr(optimizer, "built", False):
        return
    for var in optimizer.variables:
        if not var.path.endswith("learning_rate"):
            var.assign(np.zeros(var.shape, dtype=var.dtype))


def predict_proba(model, x_uint8: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = []
    for start in range(0, len(x_uint8), batch_size):
        batch = x_uint8[start : start + batch_size].astype(np.float32) / 255.0
        out.append(np.asarray(model(batch, training=False)))
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES))


# --------------------------------------------------------------------------- #
# Physiological Random Forest
# --------------------------------------------------------------------------- #
def build_forest(cfg: dict, seed: int) -> RandomForestClassifier:
    pcfg = cfg["physio"]
    return RandomForestClassifier(
        n_estimators=int(pcfg["n_estimators"]),
        criterion="gini",
        max_depth=pcfg["max_depth"],
        random_state=seed,
        n_jobs=-1,
    )


def forest_proba(forest: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    """Class probabilities in the global 7-class order, whatever classes the
    forest saw during training."""
    proba = np.zeros((len(x), N_CLASSES))
    proba[:, forest.classes_.astype(int)] = forest.predict_proba(x)
    return proba


class FederatedForest:
    """Union of the clients' forests (no raw data leaves a client).

    The prediction is the average of the client forests' class probabilities,
    weighted by the number of training samples of each client, i.e. FedAvg's
    weighting applied to tree ensembles. Gradient-based FedAvg does not apply
    to Random Forests."""

    def __init__(self, members: list, sizes: list[int]):
        if len(members) != len(sizes) or not members:
            raise ValueError("need one size per member forest")
        self.members = members
        self.weights = np.asarray(sizes, dtype=float) / float(sum(sizes))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sum(w * _member_proba(m, x) for m, w in zip(self.members, self.weights, strict=True))


def _member_proba(member, x: np.ndarray) -> np.ndarray:
    return forest_proba(member, x) if isinstance(member, RandomForestClassifier) else member.predict_proba(x)


# --------------------------------------------------------------------------- #
# Forests as plain arrays (safe to send through Flower: no pickling)
# --------------------------------------------------------------------------- #
def forest_to_arrays(forest: RandomForestClassifier) -> dict[str, np.ndarray]:
    """Flatten every tree into arrays; class distributions in the global order."""
    out: dict[str, np.ndarray] = {}
    classes = forest.classes_.astype(int)
    for t, est in enumerate(forest.estimators_):
        tree = est.tree_
        value = np.zeros((tree.node_count, N_CLASSES))
        raw = tree.value[:, 0, :]
        value[:, classes] = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-12, None)
        out[f"tree{t}/left"] = tree.children_left.astype(np.int64)
        out[f"tree{t}/right"] = tree.children_right.astype(np.int64)
        out[f"tree{t}/feature"] = tree.feature.astype(np.int64)
        out[f"tree{t}/threshold"] = tree.threshold.astype(np.float64)
        out[f"tree{t}/value"] = value
    return out


class ArrayForest:
    """Numpy re-implementation of a forest's ``predict_proba`` from arrays."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        ids = sorted({int(k.split("/")[0][4:]) for k in arrays})
        self.trees = [
            {f: arrays[f"tree{t}/{f}"] for f in ("left", "right", "feature", "threshold", "value")} for t in ids
        ]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        total = np.zeros((len(x), N_CLASSES))
        rows = np.arange(len(x))
        for tree in self.trees:
            node = np.zeros(len(x), dtype=np.int64)
            while True:
                left = tree["left"][node]
                internal = left != -1
                if not internal.any():
                    break
                go_left = x[rows, np.maximum(tree["feature"][node], 0)] <= tree["threshold"][node]
                node = np.where(internal, np.where(go_left, left, tree["right"][node]), node)
            total += tree["value"][node]
        return total / len(self.trees)


def federated_forest_from_arrays(arrays: dict[str, np.ndarray]) -> FederatedForest:
    """Rebuild a federated forest from ``c<k>/tree<t>/...`` and ``c<k>/n_train`` arrays."""
    clients = sorted({k.split("/")[0] for k in arrays})
    members, sizes = [], []
    for c in clients:
        prefix = f"{c}/"
        sub = {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix) and not k.endswith("n_train")}
        members.append(ArrayForest(sub))
        sizes.append(int(np.asarray(arrays[f"{c}/n_train"]).ravel()[0]))
    return FederatedForest(members, sizes)
