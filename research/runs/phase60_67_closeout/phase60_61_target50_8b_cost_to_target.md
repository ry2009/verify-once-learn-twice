# Direct Cost Comparability

- Methods: `adaptive_fwb` vs `resample_only`

## Success and Efficiency

| Method | Budget | Success | Avg feedback calls | Success / feedback call |
|---|---:|---:|---:|---:|
| adaptive_fwb | 1 | 0.480 | 1.000 | 0.480 |
| adaptive_fwb | 2 | 0.610 | 1.340 | 0.455 |
| adaptive_fwb | 4 | 0.470 | 1.850 | 0.254 |
| resample_only | 1 | 0.410 | 1.000 | 0.410 |
| resample_only | 2 | 0.440 | 1.580 | 0.278 |
| resample_only | 4 | 0.450 | 2.820 | 0.160 |

## Feedback Calls To Hit Target Success

| Target source (resample_only) | Target success | resample_only calls | adaptive_fwb min calls | Reduction (x) |
|---|---:|---:|---:|---:|
| resample_only@b1 | 0.410 | 1.000 | 1.000 | 1.00x |
| resample_only@b2 | 0.440 | 1.580 | 1.000 | 1.58x |
| resample_only@b4 | 0.450 | 2.820 | 1.000 | 2.82x |
