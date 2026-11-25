# Methodology

## Questions and hypotheses

Can a detector add newly labeled attack families without forgetting earlier ones? Compare bounded replay, elastic weight consolidation (EWC), distillation, drift-gated training, a static baseline, and a cumulative-data reference. We expect replay to retain earlier performance, with a possible cost in new-class learning or training time. We also test whether maximum-softmax uncertainty and diagonal class-centroid distance generalize to permanently held-out families.

## Data and chronology

Use the published [UNSW-NB15 partitions](https://research.unsw.edu.au/projects/unsw-nb15-dataset), pinned in the download script by SHA-256. The corpus comes from a controlled cyber-range and the partitions lack timestamps. The six learning episodes are constructed from attack-family labels. They are proxies for the order in which defenders receive new class annotations. They do not establish detection of real zero-day exploits.

Whitelist 16 numeric per-flow measurements and omit IDs, labels and cross-flow connection counts. Remove non-finite rows, exact within-partition duplicates and test vectors equal to any training vector. Inside the original train partition, sample disjoint fit and calibration observations for each retained family. Each episode contains a new family and fresh normal flows. The six families first appear in two fixed orders, which share normal plus Fuzzers at warmup. Backdoor, Shellcode and Worms never enter training or calibration. Different seeds change the selected distinct rows. A train-only log1p and frozen warmup standardization are shared across policies.

## Policy design

The classifier uses a two-layer encoder and a fixed-capacity output head. A mask exposes only classes announced so far. The frozen model sees only Normal and Fuzzers. New training uses capped class-frequency weights, the same architecture and a fixed epoch budget, with a separate one-epoch measurement to characterize fast adaptation.

- **Fine-tuning:** continue optimizing on the current family plus fresh normal flows.
- **Replay:** retain up to 128 or 512 past rows. Allocate an equal integer share per seen class, re-sample within each class and join the buffer with the current episode.
- **EWC:** estimate squared per-row log-likelihood gradients on up to 44 samples from the current episode, update an exponentially discounted diagonal Fisher, and penalize movements from the previous parameter estimate.
- **Distillation:** copy the prior model and add temperature-2 KL on old-class logits, using current and any replay rows.
- **Hybrid:** combine replay, EWC and distillation at the 512-row budget.
- **Drift-gated replay:** compare the stream of incoming observations to class-centroid distances. Reduce from seven to two epochs when fewer than 24% exceed the known calibration cutoff.
- **Joint reference:** refit from the same initialization with all seen observations. Its larger past-data and compute budget makes it a reference, outside the bounded-memory comparison.

The known-only calibration scores set a nominal 5% novelty threshold after each stage. Compare `1 − max(p_class)` with the minimum class-standardized squared distance. The latter uses per-class streaming counts, means and second moments. It stores summary statistics independently of the replay memory.

## Metrics

Final macro-F1 includes all seven announced classes. For each nonterminal class, compute forgetting as peak recall minus final recall. Backward transfer is final recall minus recall immediately after introduction. The few-shot forward diagnostic compares logistic probes trained with 24 new-class and 24 benign examples on the current encoder and on the untrained encoder. It is reported separately and is never used for policy updates. The joint reference can improve past classes, so backward transfer and peak forgetting are complementary.

Report mean and bootstrap intervals over five seeds within each of two orders. The intervals measure selected-row and optimizer variation on one corpus. Compare one-epoch recall, replay memory and wall time. Held-out novelty metrics depend on the mixture of unknown families and on the known class inventory. Frozen-model novelty scores use its smaller initial inventory and should be read with that distinction.

## References

- Kirkpatrick et al. [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114), PNAS 2017.
- Li and Hoiem. [Learning without Forgetting](https://doi.org/10.1007/978-3-319-46493-0_37), ECCV 2016.
- Chaudhry et al. [On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486), 2019.
- Bendale and Boult. [Towards Open Set Deep Networks](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), CVPR 2016.

The UNSW dataset terms request these five publications for academic use.

1. Moustafa and Slay. [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://doi.org/10.1109/MilCIS.2015.7348942), MilCIS 2015.
2. Moustafa and Slay. [The evaluation of Network Anomaly Detection Systems](https://doi.org/10.1080/19393555.2015.1125974), Information Security Journal 2016.
3. Moustafa et al. [Novel geometric area analysis technique for anomaly detection](https://ieeexplore.ieee.org/document/7948715), IEEE Transactions on Big Data 2017.
4. Moustafa et al. [Big data analytics for intrusion detection system using finite Dirichlet mixture models](https://link.springer.com/chapter/10.1007/978-3-319-59439-2_5), 2017.
5. Sarhan et al. [NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems](https://arxiv.org/abs/2011.09144), BDTA 2020.
