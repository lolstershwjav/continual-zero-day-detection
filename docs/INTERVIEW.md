# Technical interview preparation

1. **What is the class-incremental setting?** The detector learns to classify examples from a growing set of classes, with no task identity supplied at inference.
2. **What makes the split defensible?** The official holdout is untouched by fitting. Exact duplicate modeled vectors are excluded across partitions, and fit and calibration rows are disjoint.
3. **What is forgetting?** For each earlier class, take the best observed recall after introduction and subtract its final recall. Report backward transfer separately because improvement relative to introduction can occur even after a peak.
4. **Why do replay quotas change?** A fixed memory budget divided across more classes leaves fewer rows per class. The study measures how that constraint changes retained performance.
5. **How does EWC protect weights?** It uses squared likelihood gradients as a diagonal importance estimate and penalizes deviations from the preceding parameter values in proportion to importance.
6. **What does distillation preserve?** The KL term encourages agreement on relative probabilities among earlier classes. It constrains old logits compared with each other and can still allow new-class domination.
7. **Why mask the output head?** The fixed-size architecture allows fair parameter counts. Before introduction, unseen class outputs are unreachable in closed-set scoring.
8. **What does forward transfer mean here?** A small separate classifier assesses whether a previously trained encoder helps 24-shot discrimination of the next family compared with random encoder features.
9. **How does the drift gate work?** It checks novelty on the incoming unlabeled feature batch and reduces the update budget for batches that stay near previous distributions. This can preserve compute at the expense of learning.
10. **What can held-out families establish?** They proxy recognition of omitted families in one benchmark. A retrospective family split says little about detection of actual zero-day exploits in production.

## Résumé drafts

- Ran 90 continual-learning experiments across two orders of six UNSW-NB15 families, measuring retention, transfer, update time, replay memory and out-of-distribution detection.
- Reduced average forgetting from 82.42 to 30.44 points on the primary order with a replay buffer capped at 512 rows, and verified the trade-off against cumulative-data and no-replay baselines.
- Benchmarked maximum-softmax uncertainty and streaming centroid distance on three permanently held-out families, documenting weak open-set discrimination and avoiding unsupported zero-day claims.

CrowdStrike's research role values mathematically grounded methods and clear writing. Practice deriving the diagonal Fisher approximation and explaining when the open-set thresholds fail. Update these bullets to reflect the work you independently reproduce.
