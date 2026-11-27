"""Flower ServerApp.

``modality = "face"``  : FedAvg of the facial CNN over ``num-server-rounds``
                         rounds, weighted by each client's number of images.
``modality = "physio"``: one round of federated-forest training: every client
                         fits a Random Forest on its own data and sends the
                         trees as plain arrays; the server merges them and
                         sends the merged forest back for local evaluation.
Works with any number of SuperNodes (3 in the paper).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from flwr.app import ConfigRecord, Context, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..models import build_cnn, get_named_weights
from ..train import set_seed
from .task import config_from_run, from_record, to_record

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    n_clients = int(rc["num-clients"])
    cfg = config_from_run(rc, num_clients=n_clients)
    modality = str(rc["modality"])
    start = time.perf_counter()
    history: dict = {}
    if modality == "face":
        set_seed(int(cfg["seed"]))
        initial = to_record(get_named_weights(build_cnn(cfg)))
        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=n_clients,
            min_evaluate_nodes=n_clients,
            min_available_nodes=n_clients,
        )
        result = strategy.start(
            grid=grid,
            initial_arrays=initial,
            num_rounds=int(cfg["federated"]["rounds"]),
            train_config=ConfigRecord({"modality": "face"}),
            evaluate_config=ConfigRecord({"modality": "face"}),
        )
        final = from_record(result.arrays)
        history = {
            "train": {int(r): dict(m) for r, m in result.train_metrics_clientapp.items()},
            "evaluate": {int(r): dict(m) for r, m in result.evaluate_metrics_clientapp.items()},
        }
    elif modality == "physio":
        final, history = _federated_forest(grid, n_clients)
    else:
        raise ValueError(f"run-config modality must be 'face' or 'physio', got {modality!r}")
    seconds = time.perf_counter() - start
    print(f"[{modality}] finished in {seconds:.1f}s")

    if rc.get("output-dir"):
        out = Path(str(rc["output-dir"])).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{modality}_global.npz", **{k.replace("/", "__"): v for k, v in final.items()})
        (out / f"{modality}_history.json").write_text(json.dumps({"seconds": seconds, **history}, indent=2))
        print(f"saved global model and metrics to {out}")


def _wait_for_nodes(grid: Grid, n: int, timeout: float = 600.0) -> list[int]:
    deadline = time.time() + timeout
    while True:
        nodes = list(grid.get_node_ids())
        if len(nodes) >= n:
            return nodes[:n]
        if time.time() > deadline:
            raise TimeoutError(f"only {len(nodes)} of {n} SuperNodes connected")
        time.sleep(1.0)


def _federated_forest(grid: Grid, n: int):
    nodes = _wait_for_nodes(grid, n)
    config = ConfigRecord({"modality": "physio", "server-round": 1})
    replies = grid.send_and_receive(
        [grid.create_message(RecordDict({"config": config}), MessageType.TRAIN, node, "1") for node in nodes]
    )
    merged: dict[str, np.ndarray] = {}
    for reply in replies:
        if reply.has_error():
            raise RuntimeError(f"client error: {reply.error}")
        merged.update(from_record(reply.content["arrays"]))
    record = to_record(merged)
    replies = grid.send_and_receive(
        [
            grid.create_message(RecordDict({"arrays": record, "config": config}), MessageType.EVALUATE, node, "1")
            for node in nodes
        ]
    )
    metrics = [dict(r.content["metrics"]) for r in replies if not r.has_error()]
    total = sum(m["num-examples"] for m in metrics)
    accuracy = sum(m["test_accuracy"] * m["num-examples"] for m in metrics) / total
    print(f"[physio] federated forest from {n} clients: test accuracy {accuracy:.4f} ({total} samples)")
    return merged, {"evaluate": {"clients": metrics, "weighted_test_accuracy": accuracy}}
