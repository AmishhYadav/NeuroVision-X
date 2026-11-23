Report-agreement metrics, capacity_control_unet3d vs baseline_unet3d, held-out test split. Positive improvement means the report generated from this model's prediction agrees more closely with the report generated from ground truth.

| Metric | capacity_control_unet3d | baseline_unet3d | Improvement | 95% CI | p (Holm) | Effect (g) | n | n missing | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| abserr_cortical_frac_of_tumour | 0.008 | 0.008 | 0.000 | [-0.000, 0.001] | 1.000 | -0.08 | 189 | 0 | inconclusive |
| abserr_distance_mm | 0.003 | 0.000 | -0.003 | [-0.009, -0.000] | 1.000 | 0.07 | 189 | 0 | inconclusive |
| abserr_epicentre_distance_mm | 0.317 | 0.292 | -0.025 | [-0.095, 0.030] | 1.000 | 0.06 | 189 | 0 | inconclusive |
| abserr_frac_enhancing_of_wt | 0.031 | 0.031 | 0.001 | [-0.001, 0.002] | 0.827 | -0.06 | 189 | 0 | inconclusive |
| abserr_n_components_WT | 0.593 | 0.619 | 0.026 | [-0.079, 0.138] | 1.000 | -0.03 | 189 | 0 | inconclusive |
| abserr_n_structures | 1.741 | 1.847 | 0.106 | [-0.063, 0.286] | 1.000 | -0.09 | 189 | 0 | inconclusive |
| abserr_ventricle_frac_of_group | 0.011 | 0.011 | -0.000 | [-0.001, 0.001] | 1.000 | 0.01 | 189 | 0 | inconclusive |
| abserr_ventricle_frac_of_tumour | 0.002 | 0.002 | 0.000 | [-0.000, 0.000] | 1.000 | -0.01 | 189 | 0 | inconclusive |
| abserr_white_matter_frac_of_tumour | 0.013 | 0.013 | -0.000 | [-0.001, 0.001] | 1.000 | 0.06 | 189 | 0 | inconclusive |
| agree_deep_wm_contact | 0.942 | 0.958 | -0.016 | [-0.037, 0.000] | 1.000 | -0.13 | 189 | 0 | inconclusive |
| agree_dominant_side_WT | 1.000 | 0.995 | 0.005 | [0.000, 0.016] | 1.000 | 0.07 | 189 | 0 | inconclusive |
| agree_eloquent_any | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_epicentre_side | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_multifocal_WT | 0.735 | 0.704 | 0.032 | [-0.021, 0.085] | 1.000 | 0.09 | 189 | 0 | inconclusive |
| agree_near_eloquent | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | 0.00 | 189 | 0 | inconclusive |
| agree_ventricle_contact | 0.974 | 0.979 | -0.005 | [-0.016, 0.000] | 1.000 | -0.07 | 189 | 0 | inconclusive |
| jaccard_eloquent_structures | 0.937 | 0.934 | 0.003 | [-0.006, 0.012] | 1.000 | 0.05 | 189 | 0 | inconclusive |
| jaccard_structures | 0.912 | 0.911 | 0.001 | [-0.009, 0.012] | 1.000 | 0.02 | 189 | 0 | inconclusive |
| match_epicentre_structure | 0.847 | 0.878 | -0.032 | [-0.074, 0.005] | 1.000 | -0.12 | 189 | 0 | inconclusive |
| match_top_structure | 0.862 | 0.878 | -0.016 | [-0.053, 0.021] | 1.000 | -0.06 | 189 | 0 | inconclusive |
| precision_structures | 0.946 | 0.946 | 0.000 | [-0.007, 0.007] | 1.000 | 0.01 | 189 | 0 | inconclusive |
| recall_structures | 0.950 | 0.949 | 0.001 | [-0.005, 0.007] | 1.000 | 0.02 | 189 | 0 | inconclusive |
| relerr_vol_ET | 0.610 | 0.682 | 0.072 | [0.005, 0.190] | 0.067 | -0.10 | 184 | 5 | inconclusive |
| relerr_vol_TC | 0.886 | 1.165 | 0.279 | [-0.030, 0.867] | 1.000 | -0.07 | 189 | 0 | inconclusive |
| relerr_vol_WT | 0.076 | 0.075 | -0.001 | [-0.009, 0.006] | 1.000 | 0.02 | 189 | 0 | inconclusive |

Positive improvement always means **capacity_control_unet3d** is better, regardless of whether the metric itself is higher- or lower-is-better. Neither an inconclusive verdict (the bootstrap CI contains zero, or the Holm-adjusted p exceeds alpha) nor a negligible one (conclusive, but the entire CI sits inside the practical-equivalence margin) may be claimed as a difference. n is the paired-complete case count and n_missing the cases dropped for a NaN on either side. The Holm family is the whole table -- re-running on a subset after seeing the p-values would destroy the error-rate guarantee.
