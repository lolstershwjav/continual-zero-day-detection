"""Check class expansion, memory limits, uncertainty and policy invariants."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from continual.benchmark import evaluate, measure, threshold
from continual.data import Split, config
from continual.model import Centroids, FisherPenalty, Memory, Network, train


def rows(label, offset=0., n=60):
    rng = torch.Generator().manual_seed(label + 917)
    return Split(torch.randn(n, 6, generator=rng) + offset, torch.full((n,), label))


class ContinualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_memory_stays_under_budget_and_represents_seen_classes(self):
        m = Memory(40, 3)
        for i in range(5):
            m.store(rows(i))
            past = m.read()
            self.assertLessEqual(len(past.y), 40)
            self.assertEqual(torch.unique(past.y).tolist(), list(range(i+1)))

    def test_streaming_centroids_preserve_old_classes(self):
        obj = Centroids()
        obj.update(rows(0))
        weights = obj.stats[0][1].clone()
        obj.update(rows(1, 4.))
        torch.testing.assert_close(weights, obj.stats[0][1])
        self.assertGreater(obj.novelty(rows(2, 11.).x).mean(), obj.novelty(rows(1, 4.).x).mean())

    def test_fisher_penalizes_changed_parameter(self):
        torch.manual_seed(17)
        x = rows(0)
        m = Network(6, 2)
        f = FisherPenalty()
        f.update(m, x, 2, 11)
        self.assertEqual(float(f.value(m).detach()), 0.)
        with torch.no_grad():
            m.head.bias.add_(1.)
        self.assertGreater(float(f.value(m).detach()), 0.)

    def test_old_slice_excludes_unannounced_families(self):
        m = Network(6, 3)
        x = rows(0)
        train(m, x, 2, epochs=1, seed=11)
        self.assertEqual(m(x.x, 2).shape, (60,2))
        report = measure(m, ["a","b","future"], [x, rows(1), rows(2)], 2, 2)
        self.assertNotIn("future", report["by_class"])

    def test_both_policies_and_open_set_evaluation(self):
        from continual.data import join

        classes = ["Normal", "First", "Next", "Unseen"]
        c = [rows(i, i*3) for i in range(3)]
        tasks = [join(c[0], c[1]), join(c[0], c[2])]
        test = [*c, rows(3, 9.)]
        cfg = {"epochs":2,"ewc_lambda":24,"novelty_fpr":.05}
        freeze = evaluate("frozen", 17, classes, tasks, c, test, cfg)
        hybrid = evaluate("hybrid_512", 17, classes, tasks, c, test, cfg)
        self.assertEqual(freeze["stages"][1]["epoch_budget"], 0)
        self.assertEqual(freeze["stages"][1]["by_class"]["Next"], 0.)
        self.assertGreaterEqual(hybrid["stages"][1]["one_epoch_recall"], 0.)
        self.assertLessEqual(hybrid["observed_memory"], 512)
        self.assertGreater(hybrid["stages"][1]["open_set"]["centroid"]["auc"], .8)

    def test_no_unknown_family_can_enter_training_orders(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/"cfg.json"
            p.write_text('{"orders":[["First","Unseen"]],"unknown":["Unseen"]}')
            with self.assertRaisesRegex(ValueError, "Unknown families"):
                config(p)
        self.assertEqual(threshold(np.array([0.,1.,2.,3.]),.25),3.)


if __name__ == "__main__":
    unittest.main()
