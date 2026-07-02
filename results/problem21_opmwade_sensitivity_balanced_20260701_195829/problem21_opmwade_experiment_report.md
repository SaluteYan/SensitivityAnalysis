# Problem 21 OPMWADE Sensitivity Experiments

- Damping mode: `none`
- Target angle: `1.05` rad
- Initial population file: `init_data/PrG21InitData-target_1_05-none.npz`
- Anchor repeats / NFEs: `8` / `6400`
- Sensitivity repeats / NFEs: `6` / `6400`
- Pareto repeats / NFEs: `6` / `6400`
- Population factors: initial `18.0`, minimum `5.0`
- Parallel workers: `16`

## Normalization Scales

| Objective | Unit | Min | Max | Span |
|---|---:|---:|---:|---:|
| J_y | m^2 s | 5.7446716e-05 | 0.00094145696 | 0.00088401025 |
| J_u | J | 0.029141892 | 0.1329337 | 0.10379181 |
| J_e | rad | 3.3639304e-05 | 0.0097716949 | 0.0097380556 |

## Objective Conflict Summary

| Case | Jy | Ju | Je | Jy_bar | Ju_bar | Je_bar |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 5.74467e-05 | 0.0614452 | 0.00977169 | 0.000 | 0.311 | 1.000 |
| W2 | 0.000475024 | 0.0291419 | 0.000175956 | 0.472 | 0.000 | 0.015 |
| W3 | 0.000941457 | 0.132934 | 3.36393e-05 | 1.000 | 1.000 | 0.000 |
| UNSCALED_W14 | 0.000484333 | 0.0288198 | 4.29229e-05 | 0.483 | -0.003 | 0.001 |
| W14 | 6.94045e-05 | 0.0407839 | 4.21264e-05 | 0.014 | 0.112 | 0.001 |

## Weight Sensitivity Summary

| Case | wy | wu | we | Jy_bar | Ju_bar | Je_bar | Score | Non-dominated | Dominated by |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W1 | 1 | 0 | 0 | 0.000 | 0.311 | 1.000 | 0.000 | True |  |
| W2 | 0 | 1 | 0 | 0.472 | 0.000 | 0.015 | 0.000 | True |  |
| W3 | 0 | 0 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | False | W12;W13;W4;W7;W8 |
| W4 | 0.6 | 0.2 | 0.2 | 0.012 | 0.116 | -0.000 | 0.031 | True |  |
| W5 | 0.2 | 0.6 | 0.2 | 0.116 | 0.051 | 0.001 | 0.054 | True |  |
| W6 | 0.2 | 0.2 | 0.6 | 0.009 | 0.127 | 0.001 | 0.028 | False | W13 |
| W7 | 0.5 | 0.3 | 0.2 | 0.024 | 0.093 | -0.000 | 0.040 | True |  |
| W8 | 0.5 | 0.2 | 0.3 | 0.014 | 0.107 | -0.000 | 0.029 | True |  |
| W9 | 0.3 | 0.5 | 0.2 | 0.041 | 0.077 | 0.000 | 0.051 | True |  |
| W10 | 0.2 | 0.5 | 0.3 | 0.063 | 0.076 | 0.000 | 0.051 | True |  |
| W11 | 0.3 | 0.2 | 0.5 | 0.024 | 0.100 | 0.001 | 0.028 | True |  |
| W12 | 0.2 | 0.3 | 0.5 | 0.026 | 0.098 | -0.000 | 0.035 | True |  |
| W13 | 0.333 | 0.333 | 0.333 | 0.007 | 0.120 | -0.000 | 0.042 | True |  |
| W14 | 0.5 | 0.25 | 0.25 | 0.014 | 0.112 | 0.001 | 0.035 | True |  |
| W15 | 0.4 | 0.3 | 0.3 | 0.017 | 0.141 | 0.001 | 0.049 | False | W13;W14;W4;W6;W8 |

## Pareto Verification

- epsilon-constraint cases: `16`
- epsilon-feasible and terminal-feasible cases: `0`
- In the Pareto figures, gray crosses are infeasible epsilon attempts; orange squares are feasible epsilon-constraint solutions.

| Case | epsilon feasible | overall feasible | Jy_bar | Ju_bar | Je_bar | epsilon violation |
|---|---:|---:|---:|---:|---:|---:|
| PARETO_E11 | False | False | 0.213 | 0.761 | 4.704 | 4.7 |
| PARETO_E12 | False | False | 0.238 | 0.582 | 4.694 | 4.69 |
| PARETO_E13 | False | False | 0.182 | 1.200 | 4.620 | 4.62 |
| PARETO_E14 | False | False | 0.221 | 1.593 | 7.884 | 7.88 |
| PARETO_E21 | False | False | 0.149 | 1.250 | 4.143 | 4.14 |
| PARETO_E22 | False | False | 0.205 | 1.156 | 4.679 | 4.68 |
| PARETO_E23 | False | False | 0.198 | 0.784 | 4.469 | 4.47 |
| PARETO_E24 | False | False | 0.221 | 0.529 | 5.553 | 5.55 |
| PARETO_E31 | False | False | 0.183 | 0.510 | 4.179 | 4.18 |
| PARETO_E32 | False | False | 0.182 | 0.821 | 4.322 | 4.32 |
| PARETO_E33 | False | False | 0.168 | 0.471 | 4.189 | 4.19 |
| PARETO_E34 | False | False | 0.235 | 0.900 | 5.763 | 5.76 |
| PARETO_E41 | False | False | 0.155 | 0.435 | 4.039 | 4.04 |
| PARETO_E42 | False | False | 0.147 | 0.539 | 4.331 | 4.33 |
| PARETO_E43 | False | False | 0.189 | 0.808 | 5.109 | 5.11 |
| PARETO_E44 | False | False | 0.140 | 0.541 | 4.264 | 4.26 |

## Non-Dominated Candidate Summary

| Case | Stage | Jy_bar | Ju_bar | Je_bar |
|---|---|---:|---:|---:|
| W1 | conflict_single_objective | 0.000 | 0.311 | 1.000 |
| W2 | conflict_single_objective | 0.472 | 0.000 | 0.015 |
| UNSCALED_W14 | normalization_comparison | 0.483 | -0.003 | 0.001 |
| W10 | weight_sensitivity | 0.063 | 0.076 | 0.000 |
| W11 | weight_sensitivity | 0.024 | 0.100 | 0.001 |
| W12 | weight_sensitivity | 0.026 | 0.098 | -0.000 |
| W13 | weight_sensitivity | 0.007 | 0.120 | -0.000 |
| W14 | weight_sensitivity | 0.014 | 0.112 | 0.001 |
| W4 | weight_sensitivity | 0.012 | 0.116 | -0.000 |
| W5 | weight_sensitivity | 0.116 | 0.051 | 0.001 |
| W7 | weight_sensitivity | 0.024 | 0.093 | -0.000 |
| W8 | weight_sensitivity | 0.014 | 0.107 | -0.000 |
| W9 | weight_sensitivity | 0.041 | 0.077 | 0.000 |