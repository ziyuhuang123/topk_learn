import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import torch

from learn_topk import (
    DEFAULT_WIDTHS,
    H20_HBM_PEAK_GBS,
    NORMAL_STAGES,
    build_report,
    make_stages,
    run_benchmark_matrix,
    run_cluster_preflight,
    run_correctness_suite,
    supported_cluster_sizes,
)


NORMAL_STYLES = {
    "V0_full_sort": {"color": "#bdbdbd", "alpha": 0.70},
    "V1_partial_select": {"color": "#969696", "alpha": 0.75},
    "V2_unsorted_topk": {"color": "#6baed6", "alpha": 0.90},
    "V3A_scan_filter_atomic": {"color": "#9ecae1", "alpha": 0.95},
    "V3B_ballot_compaction": {"color": "#4292c6", "alpha": 0.95},
    "V3C_tma_pipeline": {"color": "#2171b5", "alpha": 0.95},
    "V3D_adaptive_threshold": {"color": "#08519c", "alpha": 0.95},
    "V3E_adaptive_dispatch": {"color": "#d62728", "alpha": 1.00},
}
CLUSTER_COLORS = {2: "#31a354", 4: "#f28e2b", 8: "#9467bd"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("assets/topk_h20_stages.png"))
    args = parser.parse_args()

    if args.k != 512:
        raise ValueError("the teaching benchmark fixes K=512")

    preflight = run_cluster_preflight(args.device)
    stages = make_stages(supported_cluster_sizes(preflight))
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    correctness = (
        {"status": "skipped", "cases": [], "stages": []}
        if args.skip_validation
        else run_correctness_suite(stages, device, args.k)
    )
    results = run_benchmark_matrix(
        stages,
        args.widths,
        args.batch,
        args.k,
        args.warmup,
        args.repetitions,
        device,
    )
    report = build_report(
        results,
        args.widths,
        args.batch,
        args.k,
        args.warmup,
        args.repetitions,
        device,
        preflight,
        correctness,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_suffix(".json")
    data_path.write_text(json.dumps(report, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(11, 6.5))
    selected_stage = report["cluster_selection"]["selected_stage"]
    for stage in stages:
        stage_rows = {
            row["width"]: row
            for row in results
            if row["stage"] == stage.name and row["status"] == "passed"
        }
        bandwidths = [
            stage_rows[width]["input_bandwidth_gbs"] if width in stage_rows else math.nan
            for width in args.widths
        ]
        if stage.cluster_size is None:
            style = NORMAL_STYLES[stage.name]
            axis.plot(
                args.widths,
                bandwidths,
                marker="o",
                linewidth=2.0,
                label=stage.label,
                **style,
            )
        else:
            selected = stage.name == selected_stage
            label = f"{stage.label}{' (selected)' if selected else ''}"
            axis.plot(
                args.widths,
                bandwidths,
                marker="D" if selected else "o",
                linewidth=2.8 if selected else 1.6,
                linestyle="--",
                alpha=1.0 if selected else 0.45,
                color=CLUSTER_COLORS[stage.cluster_size],
                label=label,
            )

    axis.axhline(
        H20_HBM_PEAK_GBS,
        color="black",
        linestyle=":",
        linewidth=2,
        label="H20 HBM peak (4.0 TB/s)",
    )

    final_stage = selected_stage or NORMAL_STAGES[-1].name
    final_rows = [
        row
        for row in results
        if row["stage"] == final_stage and row["status"] == "passed"
    ]
    if final_rows:
        final_row = max(final_rows, key=lambda row: row["width"])
        axis.annotate(
            f"{final_row['input_bandwidth_gbs']:.0f} GB/s\n"
            f"{final_row['hbm_peak_fraction']:.1%} of peak",
            xy=(final_row["width"], final_row["input_bandwidth_gbs"]),
            xytext=(-105, -55),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#333333"},
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_yticks([10, 30, 100, 300, 1_000, H20_HBM_PEAK_GBS])
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="both", alpha=0.25)
    axis.set_xlabel(f"Vocabulary size N (batch={args.batch}, K={args.k})")
    axis.set_ylabel("Effective input bandwidth (GB/s, higher is better)")
    axis.legend(ncol=2, fontsize=8.5)
    axis.set_title("Cumulative BF16 Top-K optimizations on NVIDIA H20")
    figure.tight_layout()
    figure.savefig(args.output, dpi=180)
    print(f"selected cluster: {selected_stage or 'none'}")
    print(f"plot: {args.output}")
    print(f"data: {data_path}")


if __name__ == "__main__":
    main()
