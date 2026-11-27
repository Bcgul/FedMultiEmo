"""Subject-aware evaluation of the physiological Random Forest on the full lab
recordings (not part of this release because they contain participant data).

The released client files have no participant IDs or time stamps, so only a
random split is possible there. With the lab table (one row every ~5 s, with
``User_id``, ``Timestamp``, ``Heart_Rate_(bpm)``, ``Skin_Conductance`` and
``Emotion_Detected``) this script reports:

* random 80/20 split of rows (protocol of the original code),
* within-subject, time-ordered split (last 20 % of every participant x emotion block),
* leave-one-participant-out (generalisation to a new person),
* the share of rows at the skin-conductance sensor floor.

Usage:
    python scripts/physio_subject_analysis.py path/to/final_data1.csv --out results/physio_subject_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURES = ["Heart_Rate_(bpm)", "Skin_Conductance"]


def forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--floor", type=float, default=23.0, help="skin conductance below this is the sensor floor")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["t"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y-%H:%M:%S")
    df = df.sort_values(["User_id", "t"]).reset_index(drop=True)
    y = df["Emotion_Detected"].to_numpy()
    report: dict = {
        "rows": len(df),
        "participants": int(df["User_id"].nunique()),
        "majority_class_share": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "sensor_floor_share": float((df["Skin_Conductance"] < args.floor).mean()),
    }

    rnd, chrono, loso = [], [], []
    block_pos = df.groupby(["User_id", "Emotion_Detected"]).cumcount()
    block_len = df.groupby(["User_id", "Emotion_Detected"])["t"].transform("size")
    late = (block_pos >= 0.8 * block_len).to_numpy()
    for seed in args.seeds:
        xtr, xte, ytr, yte = train_test_split(df[FEATURES], y, test_size=0.2, random_state=seed, stratify=y)
        rnd.append(scores(yte, forest(seed).fit(xtr, ytr).predict(xte)))
        chrono.append(
            scores(y[late], forest(seed).fit(df.loc[~late, FEATURES], y[~late]).predict(df.loc[late, FEATURES]))
        )
        yt, yp = [], []
        for user in df["User_id"].unique():
            test = (df["User_id"] == user).to_numpy()
            yp.extend(forest(seed).fit(df.loc[~test, FEATURES], y[~test]).predict(df.loc[test, FEATURES]))
            yt.extend(y[test])
        loso.append(scores(yt, yp))

    def agg(runs):
        return {
            k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs], ddof=1))}
            for k in runs[0]
        }

    report.update(random_split=agg(rnd), within_subject_time_split=agg(chrono), leave_one_participant_out=agg(loso))
    floor = df["Skin_Conductance"] < args.floor
    xtr, xte, ytr, yte = train_test_split(
        df.loc[floor, ["Skin_Conductance"]],
        y[floor.to_numpy()],
        test_size=0.2,
        random_state=0,
        stratify=y[floor.to_numpy()],
    )
    report["floor_rows_skin_conductance_only_random_split"] = scores(yte, forest(0).fit(xtr, ytr).predict(xte))
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
