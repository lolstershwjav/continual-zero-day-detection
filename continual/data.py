"""Disjoint UNSW-NB15 family episodes with held-out evaluation sets."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURES = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
            "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean"]


@dataclass(frozen=True)
class Split:
    x: torch.Tensor
    y: torch.Tensor

    def subset(self, mask):
        return Split(self.x[mask], self.y[mask])


def join(*parts):
    return Split(torch.cat([p.x for p in parts]), torch.cat([p.y for p in parts]))


def raw(directory: Path):
    columns = [*FEATURES, "attack_cat"]
    train = pd.read_csv(directory / "UNSW_NB15_training-set.csv", usecols=columns)
    test = pd.read_csv(directory / "UNSW_NB15_testing-set.csv", usecols=columns)
    train = train.loc[np.isfinite(train[FEATURES].to_numpy()).all(axis=1)].drop_duplicates(FEATURES)
    test = test.loc[np.isfinite(test[FEATURES].to_numpy()).all(axis=1)].drop_duplicates(FEATURES)
    indexed = pd.MultiIndex.from_frame(train[FEATURES])
    mask = indexed.get_indexer(pd.MultiIndex.from_frame(test[FEATURES])) < 0
    test = test.loc[mask]
    return train, test


def episodes(dataset, order, cfg, seed):
    train, test = dataset
    classes = ["Normal", *order, *cfg["unknown"]]
    fit, cal, final = {}, {}, {}
    for k, label in enumerate(classes):
        rng = np.random.default_rng(seed + sum(ord(c) * (i+1) for i, c in enumerate(label)))
        test_x = test.loc[test.attack_cat.eq(label), FEATURES].to_numpy()
        x = train.loc[train.attack_cat.eq(label), FEATURES].to_numpy()
        x = x[rng.permutation(len(x))]
        test_x = test_x[rng.permutation(len(test_x))[:cfg["test_rows"]]]
        if len(x) < 2 or len(test_x) < 2:
            raise ValueError(f"Insufficient distinct observations for {label}")
        if label not in cfg["unknown"]:
            n = min(cfg["calibration_rows"], len(x)//3)
            want = cfg["benign_per_episode"] * len(order) if k == 0 else cfg["episode_rows"]
            cal[k], fit[k] = x[:n], x[n:n+want]
        final[k] = test_x
    ref = np.concatenate([fit[0][:cfg["benign_per_episode"]], fit[1]])
    ref = np.log1p(ref)
    median = np.median(ref, axis=0)
    width = np.maximum(ref.std(axis=0), .01)

    def create(x, k):
        values = ((np.log1p(x) - median) / width).clip(-12, 12)
        return Split(torch.tensor(values, dtype=torch.float32),
                     torch.full((len(x),), k, dtype=torch.long))

    fit_tasks = []
    minor = cfg["benign_per_episode"]
    for i in range(len(order)):
        normals = fit[0][minor*i:minor*(i+1)]
        fit_tasks.append(join(create(normals, 0), create(fit[i+1], i+1)))
    cals = [create(cal[k], k) for k in sorted(cal)]
    tests = [create(final[k], k) for k in range(len(classes))]
    return classes, fit_tasks, cals, tests


def config(path):
    cfg = json.loads(path.read_text())
    if set(cfg["unknown"]) & set().union(*map(set,cfg["orders"])):
        raise ValueError("Unknown families must never appear in supervised episodes")
    base = set(cfg["orders"][0])
    if any(set(order) != base or len(order) != len(base) for order in cfg["orders"]):
        raise ValueError("Every episode order must contain the same distinct families")
    return cfg
