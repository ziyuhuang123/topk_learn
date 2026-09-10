import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import torch

from learn_topk import STAGES, benchmark, validate


LABELS = {
    "V0_full_sort": "V0 Full sort",
    "V1_partial_select": "V1 Partial select",
    "V2_skip_output_sort": "V2 Unsorted Top-K",
    "V3_deepselect": "V3 DeepSelect",
}
H20_HBM_PEAK_GBS = 4_000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("assets/topk_h20_stages.png"))
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    widths = [1024, 4096, 16384, 65536, 131072, 262144]
    results = []

    for width in widths:
        x = torch.randn((args.batch, width), device="cuda", dtype=torch.bfloat16)
        validate(STAGES, x, args.k)
        for stage, operation in STAGES.items():
            timing = benchmark(
                lambda operation=operation: operation(x, args.k),
                args.warmup,
                args.repetitions,
            )
            median_us = timing["median_us"]
            input_bandwidth_gbs = x.numel() * x.element_size() / median_us / 1_000
            results.append({
                "batch": args.batch,
                "width": width,
                "k": args.k,
                "dtype": "bfloat16",
                "stage": stage,
                **timing,
                "input_bandwidth_gbs": input_bandwidth_gbs,
            })
            print(width, stage, f"{median_us:.2f} us", f"{input_bandwidth_gbs:.1f} GB/s")

    for row in results:
        row["hbm_peak_gbs"] = H20_HBM_PEAK_GBS
        row["hbm_peak_fraction"] = row["input_bandwidth_gbs"] / H20_HBM_PEAK_GBS
        row["gap_to_hbm_peak"] = H20_HBM_PEAK_GBS / row["input_bandwidth_gbs"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_suffix(".json")
    data_path.write_text(json.dumps(results, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for stage in STAGES:
        rows = [row for row in results if row["stage"] == stage]
        axis.plot(
            [row["width"] for row in rows],
            [row["input_bandwidth_gbs"] for row in rows],
            marker="o",
            linewidth=2,
            label=LABELS[stage],
        )

    axis.axhline(
        H20_HBM_PEAK_GBS,
        color="black",
        linestyle="--",
        linewidth=2,
        label="H20 HBM peak (4.0 TB/s)",
    )
    deepselect_last = next(
        row
        for row in reversed(results)
        if row["stage"] == "V3_deepselect"
    )
    axis.annotate(
        f"{deepselect_last['input_bandwidth_gbs']:.0f} GB/s\n"
        f"{deepselect_last['hbm_peak_fraction']:.1%} of peak\n"
        f"{deepselect_last['gap_to_hbm_peak']:.2f}x gap",
        xy=(deepselect_last["width"], deepselect_last["input_bandwidth_gbs"]),
        xytext=(-95, -65),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
    )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_yticks([10, 30, 100, 300, 1_000, H20_HBM_PEAK_GBS])
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="both", alpha=0.25)
    axis.set_xlabel(f"Vocabulary size N (batch={args.batch}, K={args.k})")
    axis.set_ylabel("Effective input bandwidth (GB/s, higher is better)")
    axis.legend()
    axis.set_title("Absolute Top-K performance on NVIDIA H20 (BF16)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"plot: {args.output}")
    print(f"data: {data_path}")


if __name__ == "__main__":
    main()
