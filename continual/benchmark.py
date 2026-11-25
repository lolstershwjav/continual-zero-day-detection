"""Measure adaptation, forgetting and open-set rejection across family episodes."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch

from continual.data import config, episodes, join, raw
from continual.model import Centroids, FisherPenalty, Memory, Network, train

MODES = {
    "frozen": (0, False, False, False),
    "finetune": (0, False, False, False),
    "replay_128": (128, False, False, False),
    "replay_512": (512, False, False, False),
    "ewc": (0, True, False, False),
    "distill": (0, False, True, False),
    "hybrid_512": (512, True, True, False),
    "gated_512": (512, False, False, True),
    "joint": (0, False, False, False),
}


def threshold(values, alpha):
    return float(np.quantile(values, 1-alpha, method="higher"))


@torch.inference_mode()
def family_predictions(model, tests, active):
    return [(model(r.x, active).argmax(1).numpy(), r.y.numpy()) for r in tests]


def openness(model, summaries, cal, tests, seen, alpha, permanent):
    x_cal = join(*cal[:seen]).x
    known = join(*tests[:seen])
    unknown = join(*tests[permanent:])
    both = join(known, unknown)
    target = both.y.numpy() >= permanent
    with torch.inference_mode():
        low_conf_cal = 1 - model(x_cal, seen).softmax(1).amax(1).numpy()
        low_confidence = 1 - model(both.x, seen).softmax(1).amax(1).numpy()
    results = {}
    for name, cal_scores, values in (
            ("confidence", low_conf_cal, low_confidence),
            ("centroid", summaries.novelty(x_cal), summaries.novelty(both.x))):
        alarm = values > threshold(cal_scores, alpha)
        results[name] = dict(auc=float(roc_auc_score(target, values)),
                             ap=float(average_precision_score(target, values)),
                             recall=float(alarm[target].mean()),
                             fpr=float(alarm[~target].mean()))
    return results


def probe(model, initial, task, normal, future, cls):
    # Independent few-shot diagnostic fits heads without changing the actual policy.
    sample = join(task.subset(task.y == 0).subset(slice(0, 24)),
                  task.subset(task.y == cls).subset(slice(0, 24)))
    evaluation = join(normal, future)
    y = (sample.y == cls).numpy()
    y_test = (evaluation.y == cls).numpy()
    scores = []
    for network in (model, initial):
        with torch.inference_mode():
            a, b = network.encoder(sample.x).numpy(), network.encoder(evaluation.x).numpy()
        m = LogisticRegression(C=.5, max_iter=200).fit(a, y)
        scores.append(float(average_precision_score(y_test, m.predict_proba(b)[:, 1])))
    return scores[0] - scores[1]


def measure(model, classes, tests, seen, announced):
    predictions = family_predictions(model, tests[:announced], seen)
    y = np.concatenate([target for _, target in predictions])
    predicted = np.concatenate([values for values, _ in predictions])
    return dict(macro_f1=float(f1_score(y, predicted, average="macro", zero_division=0)),
                by_class={classes[i]: float((p == i).mean())
                          for i, (p, _) in enumerate(predictions)})


def evaluate(mode, seed, classes, tasks, calibration, tests, cfg):
    memory_limit, use_fisher, use_distill, use_gate = MODES[mode]
    torch.manual_seed(seed)
    initial = Network(tasks[0].x.shape[1], len(tasks)+1)
    model = deepcopy(initial)
    memory = Memory(memory_limit, seed)
    regularizer = FisherPenalty()
    summaries = Centroids()
    history = []
    transfer = []
    permanent = len(tasks)+1
    seen = 0
    for stage, current in enumerate(tasks):
        announced = stage+2
        changed = mode != "frozen" or stage == 0
        epoch_budget = cfg["epochs"] if changed else 0
        novelty_fraction = None
        immediate = None
        if stage and changed:
            current_gate = threshold(summaries.novelty(join(*calibration[:seen]).x),
                                     cfg["novelty_fpr"])
            novelty_fraction = float((summaries.novelty(current.x) > current_gate).mean())
            transfer.append(probe(model, initial, current, tests[0], tests[announced-1], announced-1))
            if use_gate and novelty_fraction < .24:
                epoch_budget = max(2, epoch_budget//3)
        start = perf_counter()
        if changed:
            previous = deepcopy(model) if use_distill and stage else None
            if mode == "joint":
                model = deepcopy(initial)
                training = join(*tasks[:stage+1])
            else:
                mem = memory.read()
                training = join(current, mem) if mem is not None else current
            train(model, training, announced, epochs=1, seed=seed+stage, previous=previous,
                  fisher=regularizer if use_fisher else None, strength=cfg["ewc_lambda"])
            immediate = measure(model, classes, tests, announced, announced)["by_class"][classes[announced-1]]
            train(model, training, announced, epochs=epoch_budget-1, seed=seed+200+stage,
                  previous=previous, fisher=regularizer if use_fisher else None,
                  strength=cfg["ewc_lambda"])
            if use_fisher:
                regularizer.update(model, current, announced, seed+stage+500)
            seen = announced
            if memory_limit:
                memory.store(current)
            summaries.update(current)
        duration = perf_counter() - start
        stats = measure(model, classes, tests, seen, announced)
        ood = openness(model, summaries, calibration, tests, seen, cfg["novelty_fpr"], permanent)
        history.append(dict(family=classes[announced-1], epoch_budget=epoch_budget, **stats,
                            one_epoch_recall=immediate, open_set=ood,
                            buffered_rows=sum(len(m.y) for m in memory.groups.values()),
                            update_seconds=duration, novelty_fraction=novelty_fraction))
    final = history[-1]["by_class"]
    earlier = list(classes[:len(tasks)])
    introduction = {c: next(r["by_class"][c] for r in history if c in r["by_class"]) for c in earlier}
    forgetting = [max(r["by_class"][c] for r in history if c in r["by_class"])-final[c]
                  for c in earlier]
    bwt = [final[c]-introduction[c] for c in earlier]
    return dict(model=mode, seed=seed, stages=history, final_macro_f1=history[-1]["macro_f1"],
                forgetting=float(np.mean(forgetting)), backward_transfer=float(np.mean(bwt)),
                forward_transfer_probe=float(np.mean(transfer)) if transfer else None,
                mean_update_seconds=float(np.mean([r["update_seconds"] for r in history])),
                memory_limit=memory_limit, observed_memory=max(r["buffered_rows"] for r in history),
                training_count=[len(t.y) for t in tasks], testing_count=[len(t.y) for t in tests])


def summarize(rows):
    rng = np.random.default_rng(888)
    result = []
    for order in sorted({r["order"] for r in rows}):
        for mode in MODES:
            runs = [r for r in rows if r["order"] == order and r["model"] == mode]
            fields = ("final_macro_f1", "forgetting", "backward_transfer",
                      "forward_transfer_probe", "mean_update_seconds", "observed_memory")
            stats = {k: [r[k] for r in runs if r[k] is not None] for k in fields}
            for detector in ("centroid", "confidence"):
                for metric in ("auc", "ap", "recall", "fpr"):
                    stats[f"{detector}_{metric}"] = [
                        r["stages"][-1]["open_set"][detector][metric] for r in runs]
            compact = {}
            for key, values in stats.items():
                if values:
                    samples = rng.choice(values, (3000, len(values))).mean(1)
                    lo, hi = np.quantile(samples, [.025,.975])
                    compact[key] = dict(mean=float(np.mean(values)), low=float(lo), high=float(hi))
            result.append(dict(order=order, model=mode, metrics=compact))
    return result


def run(directory, cfg):
    torch.set_num_threads(1)
    dataset = raw(directory)
    rows = []
    for i, order in enumerate(cfg["orders"]):
        for seed in cfg["seeds"]:
            classes, tasks, cal, tests = episodes(dataset, order, cfg, seed)
            for mode in MODES:
                rows.append(dict(order=i, classes=classes,
                                 **evaluate(mode, seed, classes, tasks, cal, tests, cfg)))
            print(f"Finished order {i}, seed {seed}", flush=True)
    return dict(config=cfg, retained_train_rows=len(dataset[0]),
                retained_test_rows=len(dataset[1]), summary=summarize(rows), runs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/study.json"))
    args = parser.parse_args()
    output = run(args.data, config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False))
    temp.replace(args.output)


if __name__ == "__main__":
    main()
