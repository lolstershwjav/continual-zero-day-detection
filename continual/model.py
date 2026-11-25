"""Class expansion, bounded replay, Fisher regularization and distillation."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from continual.data import join


class Network(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 48), nn.SiLU(),
                                     nn.Linear(48, 28), nn.SiLU())
        self.head = nn.Linear(28, classes)

    def forward(self, features, active):
        return self.head(self.encoder(features))[:, :active]


class Memory:
    def __init__(self, budget, seed):
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.groups = {}

    def store(self, incoming):
        for k in torch.unique(incoming.y).tolist():
            part = incoming.subset(incoming.y == k)
            if k in self.groups:
                part = join(self.groups[k], part)
            self.groups[k] = part
        quota = self.budget // len(self.groups)
        for k, part in self.groups.items():
            indices = self.rng.permutation(len(part.y))[:quota]
            self.groups[k] = part.subset(indices)

    def read(self):
        return join(*self.groups.values()) if self.groups else None


class FisherPenalty:
    def __init__(self):
        self.weight = None
        self.anchor = None

    def update(self, model, batch, active, seed):
        params = list(model.parameters())
        curvature = [torch.zeros_like(p) for p in params]
        n = min(len(batch.y), 44)
        selected = np.random.default_rng(seed).choice(len(batch.y), n, replace=False)
        for i in selected:
            loss = nn.functional.cross_entropy(model(batch.x[i:i+1], active), batch.y[i:i+1])
            for target, g in zip(curvature, torch.autograd.grad(loss, params), strict=True):
                target.add_(g.detach().square(), alpha=1/n)
        self.weight = curvature if self.weight is None else [
            old * .8 + fresh for old, fresh in zip(self.weight, curvature, strict=True)]
        self.anchor = [p.detach().clone() for p in params]

    def value(self, model):
        if self.weight is None:
            return 0.
        return sum((w * (p-a).square()).sum()
                   for w, p, a in zip(self.weight, model.parameters(), self.anchor, strict=True))


class Centroids:
    """Streaming class means and diagonal second moments with no stored feature rows."""

    def __init__(self):
        self.stats = {}

    def update(self, rows):
        for k in torch.unique(rows.y).tolist():
            x = rows.x[rows.y == k]
            n, sx, ss = self.stats.get(k, (0, 0., 0.))
            self.stats[k] = n+len(x), sx+x.sum(0), ss+x.square().sum(0)

    def novelty(self, x):
        d = []
        for n, sx, ss in self.stats.values():
            center = sx/n
            variance = (ss/n-center.square()).clamp(min=.07)
            d.append(((x-center).square()/variance).mean(1))
        return torch.stack(d, dim=1).min(1).values.numpy()


def train(model, data, active, *, epochs, seed, previous=None, fisher=None, strength=0.):
    gen = torch.Generator().manual_seed(seed)
    optim = torch.optim.AdamW(model.parameters(), lr=.006, weight_decay=.0005)
    count = torch.bincount(data.y, minlength=active).clamp(min=1)
    weights = (count.float().mean() / count).sqrt()
    teacher = deepcopy(previous).eval() if previous is not None else None
    for _ in range(epochs):
        for idx in torch.randperm(len(data.y), generator=gen).split(128):
            x, y = data.x[idx], data.y[idx]
            prediction = model(x, active)
            loss = nn.functional.cross_entropy(prediction, y, weight=weights)
            if teacher is not None:
                temp = 2.
                with torch.no_grad():
                    soft = (teacher(x, active-1)/temp).softmax(1)
                condensed = (prediction[:, :active-1]/temp).log_softmax(1)
                loss = loss + .65 * temp**2 * nn.functional.kl_div(
                    condensed, soft, reduction="batchmean")
            if fisher is not None:
                loss = loss + strength/2 * fisher.value(model)
            optim.zero_grad()
            loss.backward()
            optim.step()
