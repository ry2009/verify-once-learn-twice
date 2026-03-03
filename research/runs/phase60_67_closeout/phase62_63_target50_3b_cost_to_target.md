# Direct Cost Comparability

- Methods: `adaptive_fwb` vs `resample_only`

## Success and Efficiency

| Method | Budget | Success | Avg feedback calls | Success / feedback call |
|---|---:|---:|---:|---:|
| adaptive_fwb | 1 | 0.500 | 1.000 | 0.500 |
| adaptive_fwb | 2 | 0.490 | 1.360 | 0.360 |
| adaptive_fwb | 4 | 0.510 | 1.850 | 0.276 |
| resample_only | 1 | 0.100 | 1.000 | 0.100 |
| resample_only | 2 | 0.110 | 1.910 | 0.058 |
| resample_only | 4 | 0.130 | 3.670 | 0.035 |

## Feedback Calls To Hit Target Success

| Target source (resample_only) | Target success | resample_only calls | adaptive_fwb min calls | Reduction (x) |
|---|---:|---:|---:|---:|
| resample_only@b1 | 0.100 | 1.000 | 1.000 | 1.00x |
| resample_only@b2 | 0.110 | 1.910 | 1.000 | 1.91x |
| resample_only@b4 | 0.130 | 3.670 | 1.000 | 3.67x |
