"""Flower ClientApp: one vehicle (e.g. a Raspberry Pi with camera and wearable).

``train`` / ``evaluate`` messages carry a ``config`` record whose ``modality``
is ``face`` (CNN, FedAvg) or ``physio`` (Random Forest, federated forest).
Raw images and physiological samples never leave the client.
"""

from __future__ import annotations

import numpy as np
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..models import (
    build_cnn,
    build_forest,
    federated_forest_from_arrays,
    forest_to_arrays,
    get_named_weights,
    predict_proba,
)
from ..models import set_named_weights as set_cnn_weights
from ..train import round_seed, set_seed
from .task import config_from_run, face_partition, from_record, physio_partition, to_record

app = ClientApp()


def _ids(context: Context) -> tuple[int, int]:
    return int(context.node_config["partition-id"]), int(context.node_config["num-partitions"])


@app.train()
def train(msg: Message, context: Context) -> Message:
    pid, n = _ids(context)
    cfg = config_from_run(context.run_config, num_clients=n)
    conf = msg.content["config"]
    if conf["modality"] == "physio":
        data = physio_partition(cfg, pid, n)
        forest = build_forest(cfg, int(cfg["seed"]) + pid).fit(data["train"].x, data["train"].y)
        arrays = {f"c{pid}/{k}": v for k, v in forest_to_arrays(forest).items()}
        arrays[f"c{pid}/n_train"] = np.asarray([len(data["train"])], dtype=np.int64)
        metrics = MetricRecord({"num-examples": len(data["train"])})
        return Message(content=RecordDict({"arrays": to_record(arrays), "metrics": metrics}), reply_to=msg)

    data = face_partition(cfg, pid)
    set_seed(int(cfg["seed"]))
    model = build_cnn(cfg)
    set_cnn_weights(model, from_record(msg.content["arrays"]))
    set_seed(round_seed(cfg["seed"], int(conf["server-round"]), pid))
    train_set = data["train"]
    hist = model.fit(
        train_set.x.astype(np.float32) / 255.0,
        train_set.y,
        epochs=int(cfg["federated"]["local_epochs"]),
        batch_size=int(cfg["cnn"]["batch_size"]),
        shuffle=True,
        verbose=0,
    )
    metrics = MetricRecord({"train_loss": float(hist.history["loss"][-1]), "num-examples": len(train_set)})
    return Message(
        content=RecordDict({"arrays": to_record(get_named_weights(model)), "metrics": metrics}), reply_to=msg
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    pid, n = _ids(context)
    cfg = config_from_run(context.run_config, num_clients=n)
    conf = msg.content["config"]
    out: dict[str, float] = {}
    if conf["modality"] == "physio":
        data = physio_partition(cfg, pid, n)["test"]
        forest = federated_forest_from_arrays(from_record(msg.content["arrays"]))
        out["test_accuracy"] = float(np.mean(forest.predict_proba(data.x).argmax(axis=1) == data.y))
        out["num-examples"] = len(data)
        return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)

    data = face_partition(cfg, pid)
    model = build_cnn(cfg)
    set_cnn_weights(model, from_record(msg.content["arrays"]))
    for split in ("val", "test"):
        part = data[split]
        proba = predict_proba(model, part.x)
        out[f"{split}_loss"] = float(-np.mean(np.log(np.clip(proba[np.arange(len(part)), part.y], 1e-12, None))))
        out[f"{split}_accuracy"] = float(np.mean(proba.argmax(axis=1) == part.y))
    out["num-examples"] = len(data["val"])
    return Message(content=RecordDict({"metrics": MetricRecord(out)}), reply_to=msg)
