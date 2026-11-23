BraTS 2021, 189 held-out test cases. Best per column in bold.

| Model | WT dice | WT hd95 | TC dice | TC hd95 | ET dice | ET hd95 |
|---|---|---|---|---|---|---|
| 3D U-Net | 0.9354 ± 0.0586 (0.9570) | 5.36 ± 11.34 (2.00) | 0.9157 ± 0.1406 (0.9637) | 5.21 ± 12.05 (1.41) | 0.8587 ± 0.2019 (0.9289) | 4.02 ± 9.17 (1.41) |

Reported as mean +/- std over the held-out cases, median in parentheses. Cases where the metric was undefined are excluded from the mean rather than given an arbitrary penalty (3D U-Net ET hd95: 6/189). Cases with an empty ground-truth region: ET 2.6%. Metrics use ignore_empty=False (the BraTS convention): a region absent from the ground truth scores Dice 1.0 if the prediction is also empty.
