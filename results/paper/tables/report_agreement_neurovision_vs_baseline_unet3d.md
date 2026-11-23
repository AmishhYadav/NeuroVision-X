Report-agreement metrics, neurovision vs baseline_unet3d, held-out test split. Positive improvement means the report generated from this model's prediction agrees more closely with the report generated from ground truth.

| Metric | neurovision | baseline_unet3d | Improvement | 95% CI | p (Holm) | Effect (g) | n | n missing | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| abserr_cortical_frac_of_tumour | 0.008 | 0.008 | 0.000 | [-0.001, 0.001] | 1.000 | -0.03 | 189 | 0 | inconclusive |
| abserr_distance_mm | 0.003 | 0.000 | -0.003 | [-0.009, -0.000] | 1.000 | 0.07 | 189 | 0 | inconclusive |
| abserr_epicentre_distance_mm | 0.279 | 0.292 | 0.012 | [-0.045, 0.079] | 1.000 | -0.03 | 189 | 0 | inconclusive |
| abserr_frac_enhancing_of_wt | 0.031 | 0.031 | 0.001 | [-0.003, 0.005] | 1.000 | -0.04 | 189 | 0 | inconclusive |
| abserr_n_components_WT | 0.624 | 0.619 | -0.005 | [-0.127, 0.122] | 1.000 | 0.01 | 189 | 0 | inconclusive |
| abserr_n_structures | 1.810 | 1.847 | 0.037 | [-0.175, 0.249] | 1.000 | -0.02 | 189 | 0 | inconclusive |
| abserr_ventricle_frac_of_group | 0.010 | 0.011 | 0.000 | [-0.001, 0.002] | 1.000 | -0.04 | 189 | 0 | inconclusive |
| abserr_ventricle_frac_of_tumour | 0.002 | 0.002 | 0.000 | [0.000, 0.001] | 1.000 | -0.15 | 189 | 0 | inconclusive |
| abserr_white_matter_frac_of_tumour | 0.013 | 0.013 | -0.001 | [-0.002, 0.001] | 1.000 | 0.06 | 189 | 0 | inconclusive |
| agree_deep_wm_contact | 0.963 | 0.958 | 0.005 | [-0.021, 0.032] | 1.000 | 0.03 | 189 | 0 | inconclusive |
| agree_dominant_side_WT | 1.000 | 0.995 | 0.005 | [0.000, 0.016] | 1.000 | 0.07 | 189 | 0 | inconclusive |
| agree_eloquent_any | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_epicentre_side | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_multifocal_WT | 0.730 | 0.704 | 0.026 | [-0.037, 0.090] | 1.000 | 0.06 | 189 | 0 | inconclusive |
| agree_near_eloquent | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_ventricle_contact | 0.984 | 0.979 | 0.005 | [-0.011, 0.026] | 1.000 | 0.04 | 189 | 0 | inconclusive |
| jaccard_eloquent_structures | 0.936 | 0.934 | 0.002 | [-0.009, 0.013] | 1.000 | 0.03 | 189 | 0 | inconclusive |
| jaccard_structures | 0.907 | 0.911 | -0.003 | [-0.018, 0.010] | 1.000 | -0.03 | 189 | 0 | inconclusive |
| match_epicentre_structure | 0.857 | 0.878 | -0.021 | [-0.058, 0.016] | 1.000 | -0.08 | 189 | 0 | inconclusive |
| match_top_structure | 0.857 | 0.878 | -0.021 | [-0.058, 0.016] | 1.000 | -0.08 | 189 | 0 | inconclusive |
| precision_structures | 0.943 | 0.946 | -0.003 | [-0.015, 0.008] | 1.000 | -0.04 | 189 | 0 | inconclusive |
| recall_structures | 0.950 | 0.949 | 0.001 | [-0.007, 0.009] | 1.000 | 0.02 | 189 | 0 | inconclusive |
| relerr_vol_ET | 0.423 | 0.682 | 0.259 | [0.013, 0.732] | 0.093 | -0.09 | 184 | 5 | inconclusive |
| relerr_vol_TC | 0.780 | 1.165 | 0.384 | [0.006, 1.120] | 0.022 | -0.08 | 189 | 0 | better |
| relerr_vol_WT | 0.076 | 0.075 | -0.001 | [-0.011, 0.008] | 1.000 | 0.02 | 189 | 0 | inconclusive |

Positive improvement always means **neurovision** is better, regardless of whether the metric itself is higher- or lower-is-better. Neither an inconclusive verdict (the bootstrap CI contains zero, or the Holm-adjusted p exceeds alpha) nor a negligible one (conclusive, but the entire CI sits inside the practical-equivalence margin) may be claimed as a difference. n is the paired-complete case count and n_missing the cases dropped for a NaN on either side. The Holm family is the whole table -- re-running on a subset after seeing the p-values would destroy the error-rate guarantee.
