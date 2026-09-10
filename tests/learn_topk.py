import argparse
import json
import statistics
from dataclasses import dataclass

import torch

import deep_select


@dataclass(frozen=True)
class Case:
    batch: int
    width: int
    k: int


STAGES = {
    "V0_full_sort": lambda x, k: tuple(t[:, :k] for t in torch.sort(x, dim=1, descending=True)),
    "V1_partial_select": lambda x, k: torch.topk(x, k, dim=1, sorted=True),
    "V2_skip_output_sort": lambda x, k: torch.topk(x, k, dim=1, sorted=False),
    "V3_deepselect": lambda x, k: deep_select.topk(
        x, k, sorted=False, indices_type=torch.int64, return_value=True, abort_when_nan_found=False
    ),
}


def percentile(samples, fraction):
    return sorted(samples)[min(int(len(samples) * fraction), len(samples) - 1)]


def benchmark(operation, warmup, repetitions):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    ends[-1].synchronize()
    samples_us = [start.elapsed_time(end) * 1_000 for start, end in zip(starts, ends)]
    return {
        "median_us": statistics.median(samples_us),
        "p90_us": percentile(samples_us, 0.9),
    }


def validate(stages, x, k):
    reference_values = torch.topk(x, k, dim=1, sorted=True).values
    for name, operation in stages.items():
        values, indices = operation(x, k)
        gathered = x.gather(1, indices.to(torch.int64))
        if values is not None and not torch.equal(values, gathered):
            raise AssertionError(f"{name}: values do not match indices")
        if not torch.equal(torch.sort(gathered, dim=1, descending=True).values, reference_values):
            raise AssertionError(f"{name}: incorrect Top-K set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    cases = [
        Case(6, 131072, 512),
        Case(256, 16384, 512),
        Case(512, 4096, 512),
        Case(6, 1048576, 1024),
    ]
    results = []

    dtype = torch.bfloat16
    for case in cases:
        x = torch.randn((case.batch, case.width), device="cuda", dtype=dtype)
        validate(STAGES, x, case.k)
        baseline_us = None
        previous_us = None
        for stage, operation in STAGES.items():
            timing = benchmark(lambda operation=operation: operation(x, case.k), args.warmup, args.repetitions)
            median_us = timing["median_us"]
            baseline_us = baseline_us or median_us
            previous_speedup = 1.0 if previous_us is None else previous_us / median_us
            result = {
                "dtype": str(dtype).removeprefix("torch."),
                "batch": case.batch,
                "width": case.width,
                "k": case.k,
                "stage": stage,
                **timing,
                "speedup_vs_v0": baseline_us / median_us,
                "speedup_vs_previous": previous_speedup,
            }
            results.append(result)
            previous_us = median_us

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("dtype    shape              k     stage                    p50(us)  p90(us)  vs V0  vs prev")
    for row in results:
        shape = f"{row['batch']}x{row['width']}"
        print(
            f"{row['dtype']:<8} {shape:<18} {row['k']:<5} {row['stage']:<24} "
            f"{row['median_us']:>8.2f} {row['p90_us']:>8.2f} "
            f"{row['speedup_vs_v0']:>6.2f}x {row['speedup_vs_previous']:>7.2f}x"
        )


if __name__ == "__main__":
    main()
