# Cross-Network Federated Learning for Mobile Money Fraud Detection

Code, experimental results and figures accompanying the MSc thesis *Federated
Learning for Cross-Network Mobile Money Fraud Detection: A Privacy-Preserving
Framework for Ghana's Interoperable Telecom Ecosystem*.

**Author:** Gertrude Kaneah Abagale
**Institution:** Kwame Nkrumah University of Science and Technology, Kumasi, Ghana
**Programme:** MSc Cybersecurity and Digital Forensics

## What this repository contains

Fraud in Ghana's mobile money market crosses operator boundaries, but fraud
detection does not: each operator sees only its own transactions. Pooling the
data into a single detection system is constrained by the Data Protection Act,
2012 (Act 843) and by the privacy reasoning in *Edem Adinyira vs Scancom Ltd*
(2016). This study asks whether federated learning can close that gap, training
a shared model across three simulated operators without any transaction record
leaving its node.

The repository holds the experiment notebook, the results of five independent
runs, and the figures reported in the thesis. It does not hold the dataset,
which is public and linked below.

## Repository layout

```
notebooks/
  cnfl_experiment.ipynb        Full pipeline: partitioning, training, evaluation, SHAP
results/
  all_runs_optimal.csv         Per-run metrics at each model's MCC-optimal threshold
  all_runs_fixed.csv           Per-run metrics at the fixed threshold of 0.5
  thesis_numbers.json          Aggregated values used in the thesis tables
  run1/ ... run5/
    comparison_optimal.csv     Per-configuration metrics, tuned threshold
    comparison_fixed.csv       Per-configuration metrics, threshold 0.5
    federated_history.csv      Per-round validation loss, AUROC and MCC
    paired_ttests.json         Paired t-tests on per-fold MCC
    shap_importance.csv        Mean absolute SHAP value per feature
    summary.json               Run configuration and headline metrics
figures/
  figure_4_1_mcc_both_thresholds.png
  figure_4_2_federation_benefit.png
  figure_4_3_shap_run1.png
```

## Dataset

The experiments use **PaySim**, a public synthetic mobile money dataset of
6,362,620 transactions, of which 8,213 are labelled fraudulent (0.1291%). Fraud
is labelled only in the TRANSFER and CASH_OUT transaction types.

- Source paper: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A
  financial mobile money simulator for fraud detection. *Proceedings of the 28th
  European Modelling and Simulation Symposium*, 249-255.
- Download: https://www.kaggle.com/datasets/ealaxi/paysim1

The dataset is not committed here. Download it and place `PS_20174392719_1491204439457_log.csv`
in the notebook's working directory before running.

## Method

**Partition.** The training pool is divided across three nodes standing for
Ghana's three mobile money operators, in the ratio **72 / 21 / 7**. These shares
are derived from the National Communications Authority's published mobile
industry trends dashboard (Q1 2026 subscription shares of 72.0%, 20.9% and 7.1%,
rounded to whole percentages so the three sum to 100). No regulator publishes an
operator-level breakdown of mobile money accounts or transactions, so subscriber
share is used as the documented proxy. This assumption is **not** tested in the
study; see Limitations.

| Node | Records | Share | Fraud cases | Training rows after SMOTE-ENN |
| --- | --- | --- | --- | --- |
| MTN | 3,664,869 | 72.0% | 4,730 | 3,610,030 |
| Telecel | 1,068,920 | 21.0% | 1,380 | 1,053,156 |
| AirtelTigo | 356,307 | 7.0% | 460 | 351,011 |
| Pooled (centralised) | 5,090,096 | 100% | 6,570 | 5,013,704 |

Resampling is applied to each node's 90% local training split, not to its full
partition, which is why the resampled counts are lower than the record counts.

The split is stratified on the fraud label, so all three nodes carry the same
fraud prevalence (0.1291%) and differ only in volume, by a factor of ten between
largest and smallest. The heterogeneity studied here is therefore **quantity
skew rather than label skew**, which isolates data volume as the variable behind
any difference in federated benefit between participants.

**Preprocessing.** A 20% stratified global test set (1,272,524 transactions) is
held out before any resampling, so synthetic samples never reach evaluation
data. Identifier fields and `isFlaggedFraud` are dropped. Four features are
engineered: `amount_log`, `hour_of_day`, and two balance-reconciliation features,
`errorBalanceOrig` (newbalanceOrig + amount − oldbalanceOrg) and
`errorBalanceDest` (oldbalanceDest + amount − newbalanceDest). With five one-hot
transaction-type indicators the final vector has 13 features. Each node fits its
own scaler on its own training split and applies SMOTE-ENN to that split only.

**Models.** Five configurations are compared on the same held-out test set: the
CNFL global model, a centralised model trained on pooled data, and three
single-network models trained on one operator's data each.

| Parameter | Value |
| --- | --- |
| Architecture | Input (13) → Dense 64 ReLU → Dense 32 ReLU → Dense 1 sigmoid |
| Optimiser / loss | Adam (lr 0.001) / binary cross-entropy |
| Batch size | 512 |
| Federated rounds (max) | 50 |
| Local epochs per round | 3 |
| Early stopping | Patience 5 rounds, min delta 0.0005 on pooled validation loss |
| Aggregation | Data-size-weighted FedAvg |
| Baseline epoch budget | 150, same early stopping rule |
| Test / node validation split | 20% stratified / 10% stratified |
| SMOTE-ENN strategy / ENN neighbours | 0.10 / 3 |
| Random seed | 22108479 |

**Evaluation.** Accuracy is not reported: at 0.1291% prevalence a model that
flags nothing scores above 99.8%. Matthews Correlation Coefficient is the
primary metric, with precision, recall, F1-score and AUROC alongside. Because
MCC, precision, recall and F1 all depend on where the decision boundary is
placed, **every threshold-dependent metric is reported twice**: at each model's
MCC-optimal threshold, found by an exact sweep over the validation
probabilities, and at a fixed operating threshold of 0.5. AUROC, which is
threshold-independent, is reported alongside both.

## Results

Mean ± standard deviation over five runs.

**At each model's MCC-optimal threshold** (the primary regime)

| Configuration | MCC | Precision | Recall | F1 | AUROC |
| --- | --- | --- | --- | --- | --- |
| CNFL global | 0.8477 ± 0.0148 | 0.9074 ± 0.0335 | 0.7927 ± 0.0131 | 0.8458 ± 0.0139 | 0.9974 ± 0.0009 |
| Centralised | 0.8509 ± 0.0056 | 0.9313 ± 0.0122 | 0.7778 ± 0.0124 | 0.8476 ± 0.0059 | 0.9982 ± 0.0002 |
| MTN only | 0.8627 ± 0.0147 | 0.9204 ± 0.0213 | 0.8089 ± 0.0090 | 0.8610 ± 0.0144 | 0.9982 ± 0.0003 |
| Telecel only | 0.8479 ± 0.0086 | 0.9269 ± 0.0256 | 0.7763 ± 0.0162 | 0.8446 ± 0.0082 | 0.9972 ± 0.0002 |
| AirtelTigo only | 0.8070 ± 0.0197 | 0.8790 ± 0.0470 | 0.7422 ± 0.0264 | 0.8039 ± 0.0187 | 0.9920 ± 0.0006 |

**At the fixed threshold of 0.5** (deployment without model-specific tuning)

| Configuration | MCC | Precision | Recall | F1 | AUROC |
| --- | --- | --- | --- | --- | --- |
| CNFL global | 0.5510 ± 0.0504 | 0.3300 ± 0.0691 | 0.9328 ± 0.0252 | 0.4834 ± 0.0706 | 0.9974 ± 0.0009 |
| Centralised | 0.5014 ± 0.0157 | 0.2654 ± 0.0159 | 0.9517 ± 0.0058 | 0.4149 ± 0.0197 | 0.9982 ± 0.0002 |
| MTN only | 0.5049 ± 0.0354 | 0.2682 ± 0.0387 | 0.9584 ± 0.0146 | 0.4178 ± 0.0465 | 0.9982 ± 0.0003 |
| Telecel only | 0.6390 ± 0.0528 | 0.4551 ± 0.0730 | 0.9036 ± 0.0051 | 0.6026 ± 0.0657 | 0.9972 ± 0.0002 |
| AirtelTigo only | 0.4602 ± 0.0090 | 0.2372 ± 0.0090 | 0.8975 ± 0.0090 | 0.3751 ± 0.0113 | 0.9920 ± 0.0006 |

AUROC is identical across the two regimes because it does not depend on a
threshold. Every other difference between the tables is calibration, not
detection.

### Three findings

**Federation costs nothing against pooling.** At the tuned threshold the
federated model reaches 0.8477 MCC against the centralised model's 0.8509, a
difference of 0.0032, while no transaction record ever crosses an operator
boundary.

**The benefit is not shared equally.** It decreases with the volume of data a
participant holds: AirtelTigo gains 0.0407 MCC in all five runs, Telecel is
unaffected at −0.0002, and MTN loses 0.0149. The participant with the most to
contribute has the least to gain.

**The threshold regime changes the answer.** AUROC varies by only 0.0062 across
the five configurations, yet four of them change rank on MCC between the two
regimes. Telecel-only looks like the strongest configuration at a cut-off of
0.5 and is indistinguishable from the federated model once each is allowed its
own operating point.

## Reproducing the results

1. Download the PaySim CSV from the Kaggle link above into the working directory.
2. Open `notebooks/cnfl_experiment.ipynb` in Google Colab or Jupyter.
3. Run all cells. The original runs used Colab with CPU compute, 12.7 GB RAM and
   107.7 GB disk, on Python 3.12 with TensorFlow, scikit-learn, imbalanced-learn
   and SHAP.
4. Outputs are written to `results/runN/`.

**On exact reproduction.** The data pipeline is deterministic: a single fixed
seed governs the global split, the node partitioning, the node-local validation
splits and SMOTE-ENN, and all five runs produced byte-identical resampled
training sets. Model training is not. Even with determinism environment
variables set before TensorFlow is imported, deterministic operations enabled
and both thread pools pinned to a single thread, parallel floating-point
reductions do not reproduce exactly. Across five runs of identical code on
identical data, MCC at the fixed threshold ranged from 0.5048 to 0.6166. This is
why results are reported as distributions with consistency counts rather than as
single point estimates.

**On the five runs.** The data pipeline was executed once and the resampled
training sets reused across the five training repetitions, so every run trained
on byte-identical data. This isolates the observed variation to model training,
which is what the repetition protocol sets out to measure.

## Limitations

- **The partition is an assumption.** The sensitivity analysis over alternative
  partition ratios was designed but not completed, so the 72/21/7 split and the
  findings that rest on it remain untested against other ratios.
- **PaySim is synthetic.** It does not reproduce Ghanaian agent-network dynamics,
  SIM-swap patterns or the social-engineering typologies documented locally.
- **The leading feature may be a simulator artefact.** `errorBalanceOrig` carries
  the largest mean absolute SHAP value (0.1328 ± 0.0365, ranked first in four
  runs of five), but its discriminating power comes from fraudulent records
  reconciling exactly on the sender's side while legitimate records frequently do
  not. That reflects how the simulator maintains balances rather than an economic
  property of fraud, and should not be carried over to real data without
  re-testing.
- **The nodes are simulated on one platform.** Network latency, node dropout and
  inter-institutional trust are not modelled.
- **The operating points are not cost-derived.** The tuned thresholds fall
  between 0.945 and 0.998 and were selected to maximise MCC on validation data.
  A deployment would need to re-derive them against an explicit cost function
  weighing undetected fraud against blocked legitimate customers.

## Citing this work

```bibtex
@mastersthesis{abagale2026cnfl,
  author  = {Abagale, Gertrude Kaneah},
  title   = {Federated Learning for Cross-Network Mobile Money Fraud Detection:
             A Privacy-Preserving Framework for Ghana's Interoperable Telecom Ecosystem},
  school  = {Kwame Nkrumah University of Science and Technology},
  year    = {November 2026},
  type    = {{MSc} thesis}
}
```

## Licence
