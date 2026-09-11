import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

import deep_select


H20_HBM_PEAK_GBS = 4_000.0
DEFAULT_WIDTHS = [65_536, 131_072, 262_144, 524_288, 1_048_576, 2_097_152, 4_194_304]


@dataclass(frozen=True)
class Stage:
    name: str
    label: str
    operation: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
    cluster_size: Optional[int] = None


@dataclass(frozen=True)
class CorrectnessCase:
    batch: int
    width: int
    pattern: str
    seed: int


def _full_sort(x: torch.Tensor, k: int):
    values, indices = torch.sort(x, dim=1, descending=True)
    return values[:, :k], indices[:, :k]


def _partial_select(x: torch.Tensor, k: int):
    return torch.topk(x, k, dim=1, sorted=True)


def _unsorted_topk(x: torch.Tensor, k: int):
    return torch.topk(x, k, dim=1, sorted=False)


def _deepselect_variant(variant: str):
    def operation(x: torch.Tensor, k: int):
        return deep_select.benchmark_topk(
            x,
            k,
            variant,
            sorted=False,
            indices_type=torch.int64,
            return_value=True,
            abort_when_nan_found=False,
        )

    return operation


NORMAL_STAGES = (
    Stage("V0_full_sort", "V0 Full sort", _full_sort),
    Stage("V1_partial_select", "V1 Partial select", _partial_select),
    Stage("V2_unsorted_topk", "V2 Unsorted Top-K", _unsorted_topk),
    Stage(
        "V3A_scan_filter_atomic",
        "V3A Scan/filter + atomic",
        _deepselect_variant("v3a_scan_filter_atomic"),
    ),
    Stage(
        "V3B_ballot_compaction",
        "V3B + ballot compaction",
        _deepselect_variant("v3b_ballot_compaction"),
    ),
    Stage(
        "V3C_tma_pipeline",
        "V3C + TMA pipeline",
        _deepselect_variant("v3c_tma_pipeline"),
    ),
    Stage(
        "V3D_adaptive_threshold",
        "V3D + adaptive threshold",
        _deepselect_variant("v3d_adaptive_threshold"),
    ),
    Stage(
        "V3E_adaptive_dispatch",
        "V3E + adaptive dispatch",
        _deepselect_variant("v3e_deepselect"),
    ),
)

CLUSTER_STAGES = {
    cluster_size: Stage(
        f"V3F_cluster_C{cluster_size}",
        f"V3F Cluster C{cluster_size}",
        _deepselect_variant(f"v3f_cluster_c{cluster_size}"),
        cluster_size,
    )
    for cluster_size in (2, 4, 8)
}

STAGES = {stage.name: stage.operation for stage in NORMAL_STAGES}

CORRECTNESS_CASES = (
    CorrectnessCase(1, 512, "all_equal", 0),
    CorrectnessCase(2, 513, "heavy_tie", 1),
    CorrectnessCase(3, 4_095, "normal", 2),
    CorrectnessCase(6, 4_096, "uniform", 3),
    CorrectnessCase(3, 4_097, "ascending", 4),
    CorrectnessCase(2, 8_191, "descending", 5),
    CorrectnessCase(1, 8_192, "heavy_tie", 6),
    CorrectnessCase(2, 8_193, "normal", 7),
    CorrectnessCase(1, 65_536, "all_equal", 8),
    CorrectnessCase(1, 1_048_576, "uniform", 9),
    CorrectnessCase(1, 4_194_304, "normal", 10),
)


def make_stages(cluster_sizes: Iterable[int] = ()) -> tuple[Stage, ...]:
    return NORMAL_STAGES + tuple(CLUSTER_STAGES[size] for size in cluster_sizes)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)]


@torch.inference_mode()
def benchmark(operation, warmup: int, repetitions: int):
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
    samples_us = [start.elapsed_time(end) * 1_000.0 for start, end in zip(starts, ends)]
    return {
        "samples_us": samples_us,
        "median_us": statistics.median(samples_us),
        "p90_us": percentile(samples_us, 0.9),
    }


def make_input(case: CorrectnessCase, device: torch.device) -> torch.Tensor:
    torch.manual_seed(case.seed)
    alignment_bytes = deep_select.get_stride_requirement()[0]
    alignment_elements = alignment_bytes // torch.tensor([], dtype=torch.bfloat16).element_size()
    row_stride = math.ceil(case.width / alignment_elements) * alignment_elements
    storage = torch.empty(
        (case.batch, row_stride),
        device=device,
        dtype=torch.bfloat16,
    )
    x = storage[:, :case.width]
    if case.pattern == "normal":
        return x.normal_()
    if case.pattern == "uniform":
        return x.uniform_()
    if case.pattern == "all_equal":
        return x.fill_(0.5)
    if case.pattern == "heavy_tie":
        values = torch.randint(
            -8,
            9,
            (case.batch, case.width),
            device=device,
            dtype=torch.int32,
        )
        return x.copy_(values)
    ordered = torch.arange(case.width, device=device, dtype=torch.float32)
    ordered = (ordered / max(case.width - 1, 1)).to(torch.bfloat16)
    if case.pattern == "descending":
        ordered = ordered.flip(0)
    if case.pattern not in {"ascending", "descending"}:
        raise ValueError(f"unknown input pattern: {case.pattern}")
    return x.copy_(ordered.unsqueeze(0))


@torch.inference_mode()
def validate_stage(stage: Stage, x: torch.Tensor, k: int, reference_values=None):
    if reference_values is None:
        reference_values = torch.topk(x, k, dim=1, sorted=True).values
    values, indices = stage.operation(x, k)
    if values is None:
        raise AssertionError(f"{stage.name}: values were not returned")
    if values.shape != (x.shape[0], k) or indices.shape != (x.shape[0], k):
        raise AssertionError(f"{stage.name}: incorrect output shape")
    if values.dtype != torch.bfloat16 or indices.dtype != torch.int64:
        raise AssertionError(f"{stage.name}: incorrect output dtype")
    if not bool(((indices >= 0) & (indices < x.shape[1])).all()):
        raise AssertionError(f"{stage.name}: index out of range")
    sorted_indices = indices.sort(dim=1).values
    if k > 1 and not bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all()):
        raise AssertionError(f"{stage.name}: duplicate indices in a row")
    gathered = x.gather(1, indices)
    if not torch.equal(values, gathered):
        raise AssertionError(f"{stage.name}: values do not match indices")
    actual_values = torch.sort(values, dim=1, descending=True).values
    if not torch.equal(actual_values, reference_values):
        raise AssertionError(f"{stage.name}: incorrect Top-K value multiset")


@torch.inference_mode()
def validate(stages, x: torch.Tensor, k: int):
    stage_objects = (
        tuple(stages)
        if not isinstance(stages, dict)
        else tuple(Stage(name, name, operation) for name, operation in stages.items())
    )
    reference_values = torch.topk(x, k, dim=1, sorted=True).values
    for stage in stage_objects:
        validate_stage(stage, x, k, reference_values)


def run_input_contract_checks(device: torch.device):
    cases = []

    underbacked_storage = torch.empty(1_537, device=device, dtype=torch.bfloat16)
    underbacked = torch.as_strided(
        underbacked_storage,
        size=(2, 513),
        stride=(1_024, 1),
    )
    cases.append((
        "underbacked_aligned_rows",
        underbacked,
        "input storage must cover every complete aligned row",
    ))

    offset_storage = torch.empty(2_049, device=device, dtype=torch.bfloat16)
    misaligned = offset_storage[1:].view(2, 1_024)
    cases.append((
        "misaligned_tma_base",
        misaligned,
        "input data pointer must be aligned",
    ))

    completed = []
    for name, x, expected_message in cases:
        try:
            deep_select.topk(
                x,
                512,
                sorted=False,
                indices_type=torch.int64,
                return_value=True,
                abort_when_nan_found=False,
            )
        except RuntimeError as error:
            if expected_message not in str(error):
                raise AssertionError(f"{name}: unexpected rejection: {error}") from error
        else:
            raise AssertionError(f"{name}: unsafe TMA input was accepted")
        completed.append(name)
    return completed


def run_correctness_suite(stages: tuple[Stage, ...], device: torch.device, k: int = 512):
    completed = []
    for case in CORRECTNESS_CASES:
        x = make_input(case, device)
        validate(stages, x, k)
        torch.cuda.synchronize()
        completed.append({
            "batch": case.batch,
            "width": case.width,
            "pattern": case.pattern,
            "seed": case.seed,
        })
    return {
        "status": "passed",
        "cases": completed,
        "input_contract_checks": run_input_contract_checks(device),
        "stages": [stage.name for stage in stages],
    }


def run_cluster_canary(cluster_size: int, device_index: int) -> int:
    result = {
        "cluster_size": cluster_size,
        "status": "failed",
        "capability": None,
        "canary_cases": [],
        "repeat_launches": 0,
        "reason": "",
    }
    try:
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        capability = deep_select.get_cluster_capability(cluster_size)
        result["capability"] = capability
        if not capability["supported"]:
            result["status"] = "unsupported"
            result["reason"] = capability["reason"]
            print(json.dumps(result))
            return 0

        stage = CLUSTER_STAGES[cluster_size]
        canary_cases = (
            CorrectnessCase(1, 512, "all_equal", 100 + cluster_size),
            CorrectnessCase(2, 513, "heavy_tie", 200 + cluster_size),
            CorrectnessCase(3, 4_095, "normal", 300 + cluster_size),
            CorrectnessCase(6, 4_096, "uniform", 400 + cluster_size),
            CorrectnessCase(2, 4_097, "ascending", 500 + cluster_size),
            CorrectnessCase(1, 65_536, "descending", 600 + cluster_size),
            CorrectnessCase(1, 1_048_576, "normal", 700 + cluster_size),
        )
        for case in canary_cases:
            x = make_input(case, device)
            validate((stage,), x, 512)
            torch.cuda.synchronize()
            result["canary_cases"].append({
                "batch": case.batch,
                "width": case.width,
                "pattern": case.pattern,
                "seed": case.seed,
            })

        repeat_input = make_input(
            CorrectnessCase(3, 262_144, "heavy_tie", 800 + cluster_size),
            device,
        )
        reference_values = torch.topk(
            repeat_input,
            512,
            dim=1,
            sorted=True,
        ).values
        for _ in range(50):
            validate_stage(stage, repeat_input, 512, reference_values)
        torch.cuda.synchronize()
        result["repeat_launches"] = 50
        result["status"] = "passed"
        print(json.dumps(result))
        return 0
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result))
        return 1


def run_cluster_preflight(device_index: int, timeout_seconds: int = 180):
    results = {}
    script = str(Path(__file__).resolve())
    for cluster_size in (2, 4, 8):
        command = [
            sys.executable,
            script,
            "--device",
            str(device_index),
            "--cluster-canary-size",
            str(cluster_size),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            payload = json.loads(completed.stdout.strip())
            if completed.returncode != 0 and payload.get("status") == "passed":
                payload["status"] = "failed"
            if completed.stderr.strip():
                payload["stderr"] = completed.stderr.strip()[-2_000:]
            results[str(cluster_size)] = payload
        except subprocess.TimeoutExpired:
            results[str(cluster_size)] = {
                "cluster_size": cluster_size,
                "status": "failed",
                "reason": f"canary timed out after {timeout_seconds} seconds",
            }
        except (json.JSONDecodeError, OSError) as error:
            results[str(cluster_size)] = {
                "cluster_size": cluster_size,
                "status": "failed",
                "reason": f"canary process failed: {error}",
            }
    return results


def supported_cluster_sizes(preflight) -> tuple[int, ...]:
    return tuple(
        cluster_size
        for cluster_size in (2, 4, 8)
        if preflight.get(str(cluster_size), {}).get("status") == "passed"
    )


def run_benchmark_matrix(
    stages: tuple[Stage, ...],
    widths: list[int],
    batch: int,
    k: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
    progress: bool = True,
):
    results = []
    for width in widths:
        torch.manual_seed(width)
        x = torch.randn((batch, width), device=device, dtype=torch.bfloat16)
        reference_values = torch.topk(x, k, dim=1, sorted=True).values
        timings = {}
        for stage in stages:
            row = {
                "status": "failed",
                "reason": "",
                "dtype": "bfloat16",
                "batch": batch,
                "width": width,
                "k": k,
                "stage": stage.name,
                "label": stage.label,
                "cluster_size": stage.cluster_size,
            }
            try:
                validate_stage(stage, x, k, reference_values)
                timing = benchmark(
                    lambda stage=stage: stage.operation(x, k),
                    warmup,
                    repetitions,
                )
                median_us = timing["median_us"]
                bandwidth = x.numel() * x.element_size() / median_us / 1_000.0
                row.update({
                    "status": "passed",
                    **timing,
                    "input_bandwidth_gbs": bandwidth,
                    "hbm_peak_gbs": H20_HBM_PEAK_GBS,
                    "hbm_peak_fraction": bandwidth / H20_HBM_PEAK_GBS,
                    "gap_to_hbm_peak": H20_HBM_PEAK_GBS / bandwidth,
                })
                timings[stage.name] = median_us
                if progress:
                    print(
                        width,
                        stage.name,
                        f"{median_us:.2f} us",
                        f"{bandwidth:.1f} GB/s",
                    )
            except (RuntimeError, AssertionError) as error:
                row["reason"] = f"{type(error).__name__}: {error}"
                if progress:
                    print(width, stage.name, "FAILED", row["reason"])
                torch.cuda.empty_cache()
            results.append(row)

        for row in results[-len(stages):]:
            if row["status"] != "passed":
                continue
            stage_index = next(index for index, stage in enumerate(stages) if stage.name == row["stage"])
            previous_name = (
                "V3E_adaptive_dispatch"
                if row["cluster_size"] is not None
                else (stages[stage_index - 1].name if stage_index > 0 else None)
            )
            baseline_us = timings.get("V0_full_sort")
            previous_us = timings.get(previous_name) if previous_name else None
            v3e_us = timings.get("V3E_adaptive_dispatch")
            median_us = row["median_us"]
            row["speedup_vs_v0"] = baseline_us / median_us if baseline_us else None
            row["speedup_vs_previous"] = previous_us / median_us if previous_us else None
            row["relative_gain_vs_previous_pct"] = (
                (previous_us / median_us - 1.0) * 100.0 if previous_us else None
            )
            row["cluster_vs_normal_speedup"] = (
                v3e_us / median_us if row["cluster_size"] is not None and v3e_us else None
            )
    return results


def select_cluster_stage(results, widths: list[int]):
    candidates = []
    for cluster_size in (2, 4, 8):
        stage_name = f"V3F_cluster_C{cluster_size}"
        rows = [row for row in results if row["stage"] == stage_name]
        successful = [row for row in rows if row["status"] == "passed"]
        eligible = len(successful) == len(widths)
        geometric_mean_gbs = (
            math.exp(statistics.fmean(math.log(row["input_bandwidth_gbs"]) for row in successful))
            if eligible
            else None
        )
        candidates.append({
            "stage": stage_name,
            "cluster_size": cluster_size,
            "eligible": eligible,
            "successful_widths": [row["width"] for row in successful],
            "geometric_mean_input_bandwidth_gbs": geometric_mean_gbs,
        })
    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(eligible_candidates, key=lambda candidate: candidate["geometric_mean_input_bandwidth_gbs"])
        if eligible_candidates
        else None
    )
    return {
        "policy": "highest geometric-mean bandwidth across the complete width matrix",
        "selected_stage": selected["stage"] if selected else None,
        "selected_cluster_size": selected["cluster_size"] if selected else None,
        "candidates": candidates,
    }


def build_report(
    results,
    widths,
    batch,
    k,
    warmup,
    repetitions,
    device,
    preflight,
    correctness,
):
    properties = torch.cuda.get_device_properties(device)
    cluster_selection = select_cluster_stage(results, widths)
    selected_stage = cluster_selection["selected_stage"]
    for row in results:
        row["selected_cluster_stage"] = row["stage"] == selected_stage
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpu": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "benchmark": {
            "timing": "CUDA Events",
            "dtype": "bfloat16",
            "index_dtype": "int64",
            "sorted": False,
            "return_values": True,
            "batch": batch,
            "k": k,
            "widths": widths,
            "warmup": warmup,
            "repetitions": repetitions,
            "effective_bandwidth": "batch * width * sizeof(BF16) / median latency",
            "h20_hbm_peak_gbs": H20_HBM_PEAK_GBS,
        },
        "correctness": correctness,
        "cluster_preflight": preflight,
        "cluster_selection": cluster_selection,
        "results": results,
    }


def print_table(report):
    print("width     stage                       p50(us)   p90(us)      GB/s  vs prev")
    for row in report["results"]:
        if row["status"] != "passed":
            print(f"{row['width']:<9} {row['stage']:<27} FAILED: {row['reason']}")
            continue
        previous = row["speedup_vs_previous"]
        previous_text = "-" if previous is None else f"{previous:.2f}x"
        print(
            f"{row['width']:<9} {row['stage']:<27} "
            f"{row['median_us']:>9.2f} {row['p90_us']:>9.2f} "
            f"{row['input_bandwidth_gbs']:>9.1f} {previous_text:>8}"
        )
    selection = report["cluster_selection"]
    print(f"selected cluster: {selection['selected_stage'] or 'none'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cluster-canary-size", type=int, choices=(2, 4, 8))
    args = parser.parse_args()

    if args.cluster_canary_size is not None:
        raise SystemExit(run_cluster_canary(args.cluster_canary_size, args.device))
    if args.k != 512:
        raise ValueError("the teaching benchmark fixes K=512")

    preflight = run_cluster_preflight(args.device)
    cluster_sizes = supported_cluster_sizes(preflight)
    stages = make_stages(cluster_sizes)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    correctness = (
        {"status": "skipped", "cases": [], "stages": []}
        if args.skip_validation
        else run_correctness_suite(stages, device, args.k)
    )

    if args.validate_only:
        report = {
            "schema_version": 2,
            "correctness": correctness,
            "cluster_preflight": preflight,
        }
    else:
        results = run_benchmark_matrix(
            stages,
            args.widths,
            args.batch,
            args.k,
            args.warmup,
            args.repetitions,
            device,
            progress=not args.json,
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

    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    if args.json:
        print(serialized, end="")
    elif args.validate_only:
        print("correctness:", correctness["status"])
        for cluster_size, result in preflight.items():
            print(f"cluster C{cluster_size}: {result['status']}")
    else:
        print_table(report)


if __name__ == "__main__":
    main()
