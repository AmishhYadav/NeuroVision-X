# Severity ladder: endpoint tables

## Primary (per cell)

| kind | model | region | n_cases | L | p | p_holm | diff | ci_lo | ci_hi | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bias | neurovision | WT | 189 | 5079 | 2.332e-19 | 4.198e-18 | 0.1477 | 0.1174 | 0.1791 | GRADED |
| bias | neurovision | TC | 189 | 5130 | 1.072e-24 | 2.036e-23 | 0.1485 | 0.1149 | 0.1838 | GRADED |
| bias | nnunet_gatea | WT | 189 | 5179 | 1.324e-30 | 2.647e-29 | 0.1305 | 0.1068 | 0.1556 | GRADED |
| bias | nnunet_gatea | TC | 189 | 5210 | 1.203e-34 | 2.526e-33 | 0.1153 | 0.0904 | 0.1425 | GRADED |
| gamma | neurovision | WT | 189 | 5067 | 3.419e-18 | 5.47e-17 | 0.02766 | 0.02134 | 0.03456 | GRADED |
| gamma | neurovision | TC | 189 | 4854 | 0.000576 | 0.005184 | 0.02066 | 0.0112 | 0.0306 | GRADED |
| gamma | nnunet_gatea | WT | 189 | 5054 | 5.092e-17 | 7.638e-16 | 0.01797 | 0.01341 | 0.02267 | GRADED |
| gamma | nnunet_gatea | TC | 189 | 5023 | 2.981e-14 | 4.174e-13 | 0.01479 | 0.007499 | 0.02327 | GRADED |
| motion | baseline_unet3d | WT | 189 | 4858 | 0.0004209 | 0.005051 | 0.005435 | 0.001131 | 0.009941 | GRADED |
| motion | baseline_unet3d | TC | 189 | 4858 | 0.0004209 | 0.005051 | 0.009113 | 0.001444 | 0.01723 | GRADED |
| motion | neurovision | WT | 189 | 4857 | 0.0004404 | 0.005051 | 0.00524 | 0.0004557 | 0.01013 | GRADED |
| motion | neurovision | TC | 189 | 4798 | 0.03201 | 0.2561 | 0.007015 | -0.0004105 | 0.01506 | NOT GRADED |
| motion | nnunet_gatea | WT | 189 | 5072 | 1.01e-18 | 1.717e-17 | 0.01718 | 0.01249 | 0.02206 | GRADED |
| motion | nnunet_gatea | TC | 189 | 4913 | 1.084e-06 | 1.409e-05 | 0.01941 | 0.01154 | 0.02875 | GRADED |
| rician | baseline_unet3d | WT | 189 | 5667 | 7.634e-125 | 1.908e-123 | 0.0394 | 0.0333 | 0.04635 | GRADED |
| rician | baseline_unet3d | TC | 189 | 5593 | 2.424e-106 | 5.819e-105 | 0.01662 | 0.01272 | 0.0211 | GRADED |
| rician | neurovision | WT | 189 | 5670 | 1.261e-125 | 3.28e-124 | 0.05652 | 0.04757 | 0.06677 | GRADED |
| rician | neurovision | TC | 189 | 5590 | 1.267e-105 | 2.915e-104 | 0.01905 | 0.01377 | 0.02538 | GRADED |
| rician | nnunet_gatea | WT | 189 | 5536 | 4.061e-93 | 8.935e-92 | 0.009029 | 0.007243 | 0.01102 | GRADED |
| rician | nnunet_gatea | TC | 189 | 4638 | 0.9863 | 1 | 0.00262 | 0.0003341 | 0.005783 | NOT GRADED |
| thick_slice | baseline_unet3d | WT | 189 | 4334 | 1 | 1 | -0.01942 | -0.0246 | -0.01459 | NOT GRADED |
| thick_slice | baseline_unet3d | TC | 189 | 4588 | 0.9997 | 1 | -0.006294 | -0.01228 | -3.994e-05 | NOT GRADED |
| thick_slice | neurovision | WT | 189 | 4419 | 1 | 1 | -0.01819 | -0.0236 | -0.01329 | NOT GRADED |
| thick_slice | neurovision | TC | 189 | 4728 | 0.4749 | 1 | -0.001891 | -0.009378 | 0.005609 | NOT GRADED |
| thick_slice | nnunet_gatea | WT | 189 | 4579 | 0.9999 | 1 | -0.007642 | -0.01331 | -0.002288 | NOT GRADED |
| thick_slice | nnunet_gatea | TC | 189 | 4707 | 0.6749 | 1 | 0.002629 | -0.005904 | 0.01183 | NOT GRADED |

## Kind verdicts

| kind | models_measured | graded_regions | verdict |
|---|---|---|---|
| bias | 2 | TC,WT | GRADED_ON_MEASURED_MODELS_ONLY |
| gamma | 2 | TC,WT | GRADED_ON_MEASURED_MODELS_ONLY |
| motion | 3 | WT | GRADED |
| rician | 3 | WT | GRADED |
| thick_slice | 3 |  | NOT GRADED |

## Secondary (slope pairs)

| kind | region | stronger | weaker | n_cases | slope_stronger | slope_weaker | diff | ci_lo | ci_hi | p | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bias | WT | nnunet_gatea | neurovision | 189 | 0.02958 | 0.033 | -0.003417 | -0.007636 | 0.0006203 | 0.1048 | 0.9431 | NOT FLATTER |
| bias | TC | nnunet_gatea | neurovision | 189 | 0.02843 | 0.03419 | -0.005758 | -0.01316 | 0.001567 | 0.1274 | 0.9431 | NOT FLATTER |
| gamma | WT | nnunet_gatea | neurovision | 189 | 0.004252 | 0.006291 | -0.002039 | -0.003003 | -0.00114 | 0.0002 | 0.0032 | FLATTER |
| gamma | TC | nnunet_gatea | neurovision | 189 | 0.003149 | 0.004325 | -0.001176 | -0.002871 | 0.0005143 | 0.164 | 0.9839 | NOT FLATTER |
| motion | WT | nnunet_gatea | neurovision | 189 | 0.004144 | 0.001137 | 0.003007 | 0.002021 | 0.004179 | 0.0002 | 0.0032 | NOT FLATTER |
| motion | TC | nnunet_gatea | neurovision | 189 | 0.004613 | 0.001426 | 0.003187 | 0.001836 | 0.004721 | 0.0002 | 0.0032 | NOT FLATTER |
| motion | WT | neurovision | baseline_unet3d | 189 | 0.001137 | 0.001123 | 1.403e-05 | -0.001029 | 0.0008912 | 0.9299 | 1 | NOT FLATTER |
| motion | TC | neurovision | baseline_unet3d | 189 | 0.001426 | 0.002105 | -0.0006789 | -0.001866 | 0.0004106 | 0.2352 | 0.9839 | NOT FLATTER |
| rician | WT | nnunet_gatea | neurovision | 189 | 0.002143 | 0.01339 | -0.01125 | -0.01335 | -0.009438 | 0.0002 | 0.0032 | FLATTER |
| rician | TC | nnunet_gatea | neurovision | 189 | 0.0006372 | 0.004457 | -0.00382 | -0.005195 | -0.002644 | 0.0002 | 0.0032 | FLATTER |
| rician | WT | neurovision | baseline_unet3d | 189 | 0.01339 | 0.009368 | 0.004023 | 0.003079 | 0.005115 | 0.0002 | 0.0032 | NOT FLATTER |
| rician | TC | neurovision | baseline_unet3d | 189 | 0.004457 | 0.004005 | 0.0004523 | -0.0006586 | 0.001627 | 0.439 | 1 | NOT FLATTER |
| thick_slice | WT | nnunet_gatea | neurovision | 189 | -0.002251 | -0.004953 | 0.002702 | 0.001748 | 0.003796 | 0.0002 | 0.0032 | NOT FLATTER |
| thick_slice | TC | nnunet_gatea | neurovision | 189 | 0.000397 | -0.0008948 | 0.001292 | -0.0005493 | 0.003374 | 0.1818 | 0.9839 | NOT FLATTER |
| thick_slice | WT | neurovision | baseline_unet3d | 189 | -0.004953 | -0.005201 | 0.0002478 | -0.0006439 | 0.001192 | 0.5973 | 1 | NOT FLATTER |
| thick_slice | TC | neurovision | baseline_unet3d | 189 | -0.0008948 | -0.001912 | 0.001017 | -0.0002506 | 0.002202 | 0.1156 | 0.9431 | NOT FLATTER |
