"""Training regimes for both modalities and the evaluation metrics.

Facial CNN: centralised, local-only and FedAvg (in-process simulation of
Flower's FedAvg; ``fedmultiemo.fl`` runs the same client code with Flower).
Physiological RF: centralised, local-only and federated forest.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402

from .data import CLASSES, ImageSet, PhysioSet, to_float  # noqa: E402
from .models import (  # noqa: E402
    FederatedForest,
    build_cnn,
    build_forest,
    forest_proba,
    get_named_weights,
    predict_proba,
    reset_optimizer,
    set_named_weights,
)


def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(int(seed))
    try:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):  # pragma: no cover
        pass


def round_seed(seed: int, server_round: int, client: int) -> int:
    """Seed of one client's local training in one round (shared with Flower)."""
    return 1000 * int(seed) + 10 * int(server_round) + int(client)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }
    out.update({f"f1/{c}": float(v) for c, v in zip(CLASSES, per_class, strict=True)})
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


@dataclass
class Result:
    modality: str
    regime: str
    seed: int
    metrics: dict
    train_seconds: float
    best_step: int = 0
    history: list[dict] = field(default_factory=list)
    proba: np.ndarray | None = None  # test-set class probabilities
    weights: dict | None = None  # selected CNN weights (facial models)

    def row(self) -> dict:
        flat = {k: v for k, v in self.metrics.items() if k != "confusion"}
        return {
            "modality": self.modality,
            "regime": self.regime,
            "seed": self.seed,
            "train_seconds": self.train_seconds,
            "best_step": self.best_step,
            **flat,
        }


# --------------------------------------------------------------------------- #
# Facial CNN
# --------------------------------------------------------------------------- #
def _loss(model, data: ImageSet) -> float:
    proba = predict_proba(model, data.x)
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(data)), data.y], 1e-12, None))))


class _KeepBest(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.epoch, self.weights = np.inf, 0, None

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get("val_loss")
        if value is not None and value < self.best:
            self.best, self.epoch, self.weights = float(value), epoch + 1, self.model.get_weights()


def cnn_centralized(images: ImageSet, cfg: dict, seed: int, verbose: int = 0) -> Result:
    set_seed(seed)
    train, val, test = images.part("train"), images.part("val"), images.part("test")
    model = build_cnn(cfg)
    keep = _KeepBest()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(cfg["centralized"]["patience"]))
    start = time.perf_counter()
    hist = model.fit(
        to_float(train.x),
        train.y,
        validation_data=(to_float(val.x), val.y),
        epochs=int(cfg["centralized"]["epochs"]),
        batch_size=int(cfg["cnn"]["batch_size"]),
        shuffle=True,
        callbacks=[keep, stop],
        verbose=verbose,
    )
    seconds = time.perf_counter() - start
    model.set_weights(keep.weights)
    proba = predict_proba(model, test.x)
    history = [
        {"epoch": i + 1, **{k: float(v[i]) for k, v in hist.history.items()}} for i in range(len(hist.history["loss"]))
    ]
    return Result(
        "face",
        "centralized",
        seed,
        classification_metrics(test.y, proba),
        seconds,
        keep.epoch,
        history,
        proba,
        get_named_weights(model),
    )


def cnn_federated(
    images: ImageSet,
    clients: list[dict[str, ImageSet]],
    cfg: dict,
    seed: int,
    local_only: bool = False,
    verbose: int = 0,
) -> Result:
    """FedAvg over the clients (or, with ``local_only``, no aggregation).

    Every round each client starts from the global weights with a fresh
    optimiser and trains ``local_epochs`` epochs; the server averages the
    weights by number of training images (McMahan et al., 2017). The round
    with the lowest validation loss (weighted over the clients' local
    validation shares) is kept and evaluated once on the test set."""
    fl = cfg["federated"]
    set_seed(seed)
    model = build_cnn(cfg)
    init = get_named_weights(model)
    weights = {k: copy.deepcopy(init) for k in range(len(clients))} if local_only else None
    global_w = init
    best = {"val": np.inf, "round": 0, "weights": copy.deepcopy(init if not local_only else weights)}
    history = []
    start = time.perf_counter()
    for rnd in range(1, int(fl["rounds"]) + 1):
        tick = time.perf_counter()
        updates, sizes, losses = [], [], []
        for k, data in enumerate(clients):
            train = data["train"]
            if len(train) == 0:
                continue
            set_named_weights(model, weights[k] if local_only else global_w)
            reset_optimizer(model)
            set_seed(round_seed(seed, rnd, k))
            hist = model.fit(
                to_float(train.x),
                train.y,
                epochs=int(fl["local_epochs"]),
                batch_size=int(cfg["cnn"]["batch_size"]),
                shuffle=True,
                verbose=0,
            )
            new = get_named_weights(model)
            if local_only:
                weights[k] = new
            updates.append(new)
            sizes.append(len(train))
            losses.append(hist.history["loss"][-1])
        if not local_only:
            total = float(sum(sizes))
            global_w = {
                key: sum(u[key] * (n / total) for u, n in zip(updates, sizes, strict=True)) for key in updates[0]
            }
        val_losses, val_sizes = [], []
        for k, data in enumerate(clients):
            if len(data["val"]) == 0:
                continue
            set_named_weights(model, weights[k] if local_only else global_w)
            val_losses.append(_loss(model, data["val"]))
            val_sizes.append(len(data["val"]))
        val_loss = float(np.average(val_losses, weights=val_sizes))
        history.append(
            {
                "round": rnd,
                "train_loss": float(np.average(losses, weights=sizes)),
                "val_loss": val_loss,
                "seconds": time.perf_counter() - tick,
            }
        )
        if verbose:
            print(
                f"round {rnd:3d} train {history[-1]['train_loss']:.4f} val {val_loss:.4f} "
                f"({history[-1]['seconds']:.0f}s)",
                flush=True,
            )
        if val_loss < best["val"]:
            best = {"val": val_loss, "round": rnd, "weights": copy.deepcopy(weights if local_only else global_w)}
    seconds = time.perf_counter() - start

    test = images.part("test")
    if local_only:  # every vehicle uses its own model; report the client average
        per_client = []
        for k in range(len(clients)):
            set_named_weights(model, best["weights"][k])
            per_client.append(predict_proba(model, test.x))
        metrics = [classification_metrics(test.y, p) for p in per_client]
        avg = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key not in ("confusion", "n")}
        avg["n"] = int(len(test))
        avg["confusion"] = metrics[0]["confusion"]
        return Result("face", "local", seed, avg, seconds, best["round"], history, per_client[0])
    set_named_weights(model, best["weights"])
    proba = predict_proba(model, test.x)
    return Result(
        "face",
        "fedavg",
        seed,
        classification_metrics(test.y, proba),
        seconds,
        best["round"],
        history,
        proba,
        best["weights"],
    )


# --------------------------------------------------------------------------- #
# Physiological Random Forest
# --------------------------------------------------------------------------- #
def _stack(sets: list[PhysioSet]) -> PhysioSet:
    return PhysioSet(np.concatenate([s.x for s in sets]), np.concatenate([s.y for s in sets]))


def rf_regimes(clients: list[dict[str, PhysioSet]], cfg: dict, seed: int) -> tuple[list[Result], PhysioSet]:
    """Centralised RF, local RFs and the federated forest, all evaluated on
    the union of the clients' test splits."""
    test = _stack([c["test"] for c in clients])
    results = []

    start = time.perf_counter()
    central = build_forest(cfg, seed).fit(*_xy(_stack([c["train"] for c in clients])))
    results.append(
        Result(
            "physio",
            "centralized",
            seed,
            classification_metrics(test.y, forest_proba(central, test.x)),
            time.perf_counter() - start,
            proba=forest_proba(central, test.x),
        )
    )

    start = time.perf_counter()
    local = [build_forest(cfg, seed + k).fit(*_xy(c["train"])) for k, c in enumerate(clients)]
    seconds = time.perf_counter() - start
    per_client = [classification_metrics(test.y, forest_proba(f, test.x)) for f in local]
    avg = {k: float(np.mean([m[k] for m in per_client])) for k in per_client[0] if k not in ("confusion", "n")}
    avg.update(n=int(len(test)), confusion=per_client[0]["confusion"])
    results.append(Result("physio", "local", seed, avg, seconds, proba=forest_proba(local[0], test.x)))

    fed = FederatedForest(local, [len(c["train"]) for c in clients])
    proba = fed.predict_proba(test.x)
    results.append(
        Result("physio", "federated_forest", seed, classification_metrics(test.y, proba), seconds, proba=proba)
    )
    return results, test


def _xy(data: PhysioSet):
    return data.x, data.y
