# KernelLLM Comparison Card

## KernelLLM Published Reference

- Model: `ScalingIntelligence/KernelLLM-8B-Instruct`
- Source: https://huggingface.co/ScalingIntelligence/KernelLLM-8B-Instruct
- KernelBench L1 pass@1/10/20: 20.2/51.8/57.1
- Reported training footprint: 25000 examples, 192 GPU-hours

## Our k-Pass Proxy (KernelBench transfer target-12)

- Note: this is a different task/eval setup than KernelBench-L1 Triton pass@k, so use as cost/efficiency proxy only.

| Method | Budget | Success | Avg feedback calls | Success / feedback |
|---|---:|---:|---:|---:|
| resample_only | 1 | 0.410 | 1.000 | 0.410 |
| adaptive_fwb | 1 | 0.480 | 1.000 | 0.480 |
| adaptive_fwb | 2 | 0.610 | 1.340 | 0.455 |
| adaptive_fwb | 4 | 0.470 | 1.850 | 0.254 |
| inference_only | 1 | 0.380 | 1.000 | 0.380 |

### Feedback Calls To Hit Resample Targets

| Target source | Target success | Source feedback calls | Adaptive min calls to hit target | Call reduction (x) |
|---|---:|---:|---:|---:|
| resample_only@b1 | 0.410 | 1.000 | 1.000 | 1.00x |

## Win Conditions vs KernelLLM

- Cost win: match/exceed a resample pass@k point with lower feedback budget.
- Speed win: higher success per feedback call (`success_mean / feedback_mean`).
- Size win: use smaller base model while preserving acceptable success.
