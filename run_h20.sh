#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export DEEP_SELECT_CUDA_ARCHS="${DEEP_SELECT_CUDA_ARCHS:-90a}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

build() {
    python3 setup.py build_ext --inplace
}

benchmark() {
    python3 tests/learn_topk.py \
        --warmup "${BENCH_WARMUP:-30}" \
        --repetitions "${BENCH_REPETITIONS:-100}"
}

plot() {
    python3 tests/plot_topk_stages.py \
        --warmup "${PLOT_WARMUP:-20}" \
        --repetitions "${PLOT_REPETITIONS:-50}"
}

case "${1:-all}" in
    build)
        build
        ;;
    benchmark)
        benchmark
        ;;
    plot)
        plot
        ;;
    all)
        build
        benchmark
        plot
        ;;
    *)
        printf 'Usage: %s [build|benchmark|plot|all]\n' "$0" >&2
        exit 2
        ;;
esac
