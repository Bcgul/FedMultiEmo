## Facial CNN (FER test set)

| Regime | Seeds | Accuracy | Macro-F1 | Best round/epoch | Train time (min) |
|---|---|---|---|---|---|
| centralized | 1 | 0.864 | 0.703 | 30 | 37 |
| fedavg | 1 | 0.843 | 0.694 | 15 | 26 |
| local | 1 | 0.803 | 0.631 | 15 | 27 |

## Physiological Random Forest (union of the clients' test splits)

| Regime | Seeds | Accuracy | Macro-F1 |
|---|---|---|---|
| centralized | 5 | 0.759 ± 0.013 | 0.760 ± 0.011 |
| local | 5 | 0.736 ± 0.005 | 0.738 ± 0.005 |
| federated_forest | 5 | 0.771 ± 0.009 | 0.771 ± 0.011 |

## Decision-level fusion on label-matched pairs

| Facial model | Rule | Accuracy | Macro-F1 |
|---|---|---|---|
| centralized | face_only | 0.864 ± 0.000 | 0.703 ± 0.000 |
| centralized | physio_only | 0.756 ± 0.015 | 0.655 ± 0.023 |
| centralized | vote | 0.931 ± 0.005 | 0.829 ± 0.012 |
| centralized | average | 0.946 ± 0.005 | 0.858 ± 0.007 |
| centralized | product | 0.961 ± 0.003 | 0.911 ± 0.014 |
| fedavg | face_only | 0.843 ± 0.000 | 0.694 ± 0.000 |
| fedavg | physio_only | 0.756 ± 0.015 | 0.655 ± 0.023 |
| fedavg | vote | 0.923 ± 0.004 | 0.825 ± 0.010 |
| fedavg | average | 0.942 ± 0.003 | 0.862 ± 0.013 |
| fedavg | product | 0.961 ± 0.003 | 0.916 ± 0.015 |
| local | face_only | 0.800 ± 0.000 | 0.630 ± 0.000 |
| local | physio_only | 0.756 ± 0.015 | 0.655 ± 0.023 |
| local | vote | 0.894 ± 0.005 | 0.776 ± 0.014 |
| local | average | 0.922 ± 0.006 | 0.816 ± 0.021 |
| local | product | 0.944 ± 0.005 | 0.856 ± 0.034 |
