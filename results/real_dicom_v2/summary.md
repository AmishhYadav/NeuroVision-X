# Real-DICOM v2 (W3.4) -- summary

**NOT external validation.** test-split BraTS 2021 patients via RSNA DICOM -- carried-reference front-end cost and thick-slice refusal check, NOT external validation.

## Counts

- n_studies: 90
- n_done: 52
- n_refused: 38
- n_failed: 0
- n_thick_slice_refusals: 35
- n_override_runs: 35
- n_deployed_with_mask: 55
- n_carry_qc_failed: 18
- n_mirrored: 0
- n_bridge_stored_masks: 22

## E1 -- front-end cost after removing registration disagreement (NOT external validation)

**Verdict: INCONCLUSIVE** (margin 0.05, n = 37, carry-QC excluded 18).

| region | n | median_diff | median_diff_lo | median_diff_hi | mean_diff | wilcoxon_pvalue |
|---|---|---|---|---|---|---|
| ET | 37 | -0.0584 | -0.0737 | -0.0458 | -0.1077 | 0.0000 |
| TC | 37 | -0.0281 | -0.0465 | -0.0224 | -0.0776 | 0.0000 |
| WT | 37 | -0.0342 | -0.0565 | -0.0250 | -0.0651 | 0.0000 |

## E2 -- was the thick-slice refusal right? (NOT external validation)

**Verdict: REFUSES USABLE STUDIES**. Usable on the carried reference: 18/20 = 0.900 (95% CI 0.683-0.988); 35 thick-slice refusals, 15 excluded (carry QC / unscored).

## E3 -- Table XV v2 (NOT external validation)

| row | column | value | ci_lo | ci_hi | n |
|---|---|---|---|---|---|
| completion: done | atlas | 0.5778 | 0.4778 | 0.6778 | 90 |
| completion: refused (gatekeeper) | atlas | 0.0333 | 0.0000 | 0.0778 | 90 |
| completion: refused (input_qc (pre-preprocessing)) | atlas | 0.3889 | 0.2889 | 0.4889 | 90 |
| carry_qc_excluded | atlas | 0.0000 | nan | nan | 90 |
| p_accepted_and_usable | atlas | 0.2333 | 0.1444 | 0.3222 | 90 |
| p_usable_given_accepted | atlas | 0.4038 | 0.2692 | 0.5385 | 52 |
| p_silent_failure | atlas | 0.3444 | 0.2444 | 0.4444 | 90 |
| front_end_cost_median_ET | atlas | -0.4935 | -0.5680 | -0.3826 | 55 |
| front_end_cost_median_TC | atlas | -0.2527 | -0.4015 | -0.2358 | 55 |
| front_end_cost_median_WT | atlas | -0.2391 | -0.3308 | -0.1791 | 55 |
| completion: done | carried | 0.5778 | 0.4778 | 0.6778 | 90 |
| completion: refused (gatekeeper) | carried | 0.0333 | 0.0000 | 0.0778 | 90 |
| completion: refused (input_qc (pre-preprocessing)) | carried | 0.3889 | 0.2889 | 0.4889 | 90 |
| carry_qc_excluded | carried | 18.0000 | nan | nan | 90 |
| p_accepted_and_usable | carried | 0.4028 | 0.2917 | 0.5139 | 72 |
| p_usable_given_accepted | carried | 0.8286 | 0.6857 | 0.9429 | 35 |
| p_silent_failure | carried | 0.0833 | 0.0278 | 0.1528 | 72 |
| front_end_cost_median_ET | carried | -0.0584 | -0.0737 | -0.0458 | 37 |
| front_end_cost_median_TC | carried | -0.0281 | -0.0465 | -0.0224 | 37 |
| front_end_cost_median_WT | carried | -0.0342 | -0.0565 | -0.0250 | 37 |
| completion: done | overridden | 0.9222 | 0.8667 | 0.9778 | 90 |
| completion: refused (after override) | overridden | 0.0444 | 0.0111 | 0.0889 | 90 |
| completion: refused (gatekeeper) | overridden | 0.0333 | 0.0000 | 0.0778 | 90 |
| carry_qc_excluded | overridden | 33.0000 | nan | nan | 90 |
| p_accepted_and_usable | overridden | 0.7895 | 0.6842 | 0.8947 | 57 |
| p_usable_given_accepted | overridden | 0.8824 | 0.7843 | 0.9608 | 51 |
| p_silent_failure | overridden | 0.1053 | 0.0351 | 0.1930 | 57 |
| front_end_cost_median_ET | overridden | -0.0566 | -0.0673 | -0.0449 | 57 |
| front_end_cost_median_TC | overridden | -0.0273 | -0.0440 | -0.0225 | 57 |
| front_end_cost_median_WT | overridden | -0.0304 | -0.0481 | -0.0250 | 57 |

## E4 -- CAUTION instead of REFUSE for thick slices (NOT external validation)

**post-hoc variant** (not the deployed rule). 35 thick-slice studies, 31 accepted when overridden.

```json
{
  "endpoint": "E4",
  "label": "post-hoc variant",
  "reference": "carried",
  "n_thick_slice": 35,
  "n_thick_slice_accepted": 31,
  "n": 57,
  "n_accepted": 51,
  "n_carry_qc_excluded": 33,
  "p_accepted_and_usable": [
    0.7894736842105263,
    0.6842105263157895,
    0.8947368421052632
  ],
  "p_usable_given_accepted": [
    0.8823529411764706,
    0.7843137254901961,
    0.9607843137254902
  ],
  "p_silent_failure": [
    0.10526315789473684,
    0.03508771929824561,
    0.19298245614035087
  ]
}
```

## Bridge to P1.2

22 stored masks compared, bar WT Dice >= 0.99: all pass = False; below the bar: ['BraTS2021_00185', 'BraTS2021_00230', 'BraTS2021_00247', 'BraTS2021_00271', 'BraTS2021_00283', 'BraTS2021_00296', 'BraTS2021_00320', 'BraTS2021_00364', 'BraTS2021_00464', 'BraTS2021_00478', 'BraTS2021_00504', 'BraTS2021_00561', 'BraTS2021_00579', 'BraTS2021_00610', 'BraTS2021_00620', 'BraTS2021_00628', 'BraTS2021_00638', 'BraTS2021_00679', 'BraTS2021_00694', 'BraTS2021_00703', 'BraTS2021_00724', 'BraTS2021_01008'].
