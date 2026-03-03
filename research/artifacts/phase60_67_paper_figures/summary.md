# Phase60-67 KernelBench Figure Pack Summary

## Success snapshots (Adaptive / Resample / Inference)

- Target-50 8B: b1 0.480/0.410/0.380, b2 0.610/0.440, b4 0.470/0.450
- Target-50 3B: b1 0.500/0.100/0.090, b2 0.490/0.110, b4 0.510/0.130
- Target-100 8B: b1 0.530/0.410/0.405, b2 0.575/0.460, b4 0.440/0.525

## Phase66 tuned-adaptive vs phase60/61 base-adaptive

- b2 success: 0.610 -> 0.567
- b4 success: 0.470 -> 0.573
- b2 test_calls: 3.340 -> 2.340
- b4 test_calls: 6.080 -> 6.833

## Phase67 mode ablation (mean across seeds)

- b2 off: success=0.567, feedback=1.347, test_calls=3.447
- b2 always: success=0.553, feedback=1.340, test_calls=5.233
- b2 fail_only: success=0.533, feedback=1.347, test_calls=5.387
- b4 always: success=0.573, feedback=1.860, test_calls=7.453
- b4 off: success=0.507, feedback=1.867, test_calls=6.393
- b4 fail_only: success=0.447, feedback=1.933, test_calls=7.833
