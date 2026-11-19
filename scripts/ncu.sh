#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_METRICS='gpu__time_duration.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'

ncu="${NCU:-ncu}"
kernel_name=""
launch_skip=1
launch_count=1
metrics="$DEFAULT_METRICS"
output=""
csv=1
declare -a extra_ncu_args=()

usage() {
    cat <<'EOF'
Usage: scripts/ncu.sh [options] -- <executable> [executable arguments...]

Profiles a program with NVIDIA Nsight Compute. Set NCU to use a specific ncu
binary; by default, `ncu` is resolved from PATH.

Options:
  --kernel <name-or-pattern>  Profile matching CUDA kernel(s) only.
  --skip <count>              Skip this many matching kernel launches (default: 1).
  --count <count>             Profile this many matching launches (default: 1).
  --metrics <metric-list>     Comma-separated Nsight Compute metrics.
  --output <path>             Write a .ncu-rep report (path without extension is OK).
  --no-csv                    Use Nsight Compute's normal terminal report instead of CSV.
  --ncu <path>                Nsight Compute executable (overrides $NCU and PATH).
  --ncu-arg <argument>        Add one additional argument to Nsight Compute; repeatable.
  -h, --help                  Show this help.

The default metrics are:
  gpu__time_duration.sum
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum

Examples:
  scripts/ncu.sh --kernel linear_forward_kernel -- ./build/Release/bench_train_step.exe
  scripts/ncu.sh --kernel 'linear_*' --skip 0 --count 10 --output profiles/linear -- ./app
  scripts/ncu.sh --ncu-arg --set --ncu-arg full -- ./app
EOF
}

while (($#)); do
    case "$1" in
        --kernel) kernel_name="${2:?--kernel requires a value}"; shift 2 ;;
        --skip) launch_skip="${2:?--skip requires a value}"; shift 2 ;;
        --count) launch_count="${2:?--count requires a value}"; shift 2 ;;
        --metrics) metrics="${2:?--metrics requires a value}"; shift 2 ;;
        --output) output="${2:?--output requires a value}"; shift 2 ;;
        --no-csv) csv=0; shift ;;
        --ncu) ncu="${2:?--ncu requires a path}"; shift 2 ;;
        --ncu-arg) extra_ncu_args+=("${2:?--ncu-arg requires a value}"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if (($# == 0)); then
    printf 'Missing executable. Put -- before the program to profile.\n\n' >&2
    usage >&2
    exit 2
fi

if ! command -v "$ncu" >/dev/null 2>&1 && [[ ! -x "$ncu" ]]; then
    printf 'Nsight Compute executable not found: %s\n' "$ncu" >&2
    printf 'Install Nsight Compute, add ncu to PATH\n' >&2
    exit 127
fi

cmd=("$ncu" --launch-skip "$launch_skip" --launch-count "$launch_count" --metrics "$metrics")
[[ -n "$kernel_name" ]] && cmd+=(--kernel-name "$kernel_name")
[[ -n "$output" ]] && cmd+=(--export "$output")
((csv)) && cmd+=(--csv)
cmd+=("${extra_ncu_args[@]}" -- "$@")

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
