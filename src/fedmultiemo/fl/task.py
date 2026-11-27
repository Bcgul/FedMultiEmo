"""Helpers shared by the Flower ServerApp and ClientApp."""

from __future__ import annotations

import functools
import json

import numpy as np
from flwr.app import Array, ArrayRecord

from ..config import load_config
from ..data import PhysioSet, client_image_sets, client_physio_sets, load_fer, partition, split_physio


def config_from_run(run_config: dict, num_clients: int | None = None) -> dict:
    """Packaged defaults + optional YAML + the Flower run config."""
    cfg = load_config(run_config.get("config-file") or None)
    if run_config.get("fer-csv"):
        cfg["fer"]["csv"] = str(run_config["fer-csv"])
    if run_config.get("ferplus-csv"):
        cfg["fer"]["ferplus_csv"] = str(run_config["ferplus-csv"])
    if run_config.get("physio-dir"):
        base = str(run_config["physio-dir"]).rstrip("/\\")
        cfg["physio"]["client_files"] = [
            f"{base}/client_{k + 1}.csv" for k in range(len(cfg["physio"]["client_files"]))
        ]
    cfg["federated"]["rounds"] = int(run_config["num-server-rounds"])
    cfg["federated"]["local_epochs"] = int(run_config["local-epochs"])
    cfg["seed"] = int(run_config["seed"])
    if num_clients is not None:
        cfg["federated"]["num_clients"] = int(num_clients)
    return cfg


def _key(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True)


@functools.lru_cache(maxsize=1)
def _images(cfg_json: str):
    return load_fer(json.loads(cfg_json))


def face_partition(cfg: dict, partition_id: int):
    """This client's share of the FER training/validation images, plus the
    full test set (identical for every client)."""
    images = _images(_key({k: cfg[k] for k in ("fer",)}))
    share = client_image_sets(images, cfg, int(cfg["seed"]))[partition_id]
    share["test"] = images.part("test")
    return share


def physio_partition(cfg: dict, partition_id: int, num_partitions: int) -> dict[str, PhysioSet]:
    """One client file per SuperNode when the counts match; otherwise all
    files are pooled and split evenly (stratified) over the SuperNodes."""
    seed = int(cfg["seed"])
    files = cfg["physio"]["client_files"]
    if num_partitions == len(files):
        return client_physio_sets(cfg, seed)[partition_id]
    sets = client_physio_sets(cfg, seed)
    pooled = {
        s: PhysioSet(np.concatenate([c[s].x for c in sets]), np.concatenate([c[s].y for c in sets]))
        for s in ("train", "test")
    }
    out = {}
    for split, data in pooled.items():
        owner = partition(data.y, num_partitions, "iid", 0.0, seed + (split == "test"))
        out[split] = PhysioSet(data.x[owner == partition_id], data.y[owner == partition_id])
    return out


def to_record(named: dict[str, np.ndarray]) -> ArrayRecord:
    return ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in named.items()})


def from_record(record: ArrayRecord) -> dict[str, np.ndarray]:
    return {k: v.numpy() for k, v in record.items()}


__all__ = ["config_from_run", "face_partition", "physio_partition", "to_record", "from_record", "split_physio"]
