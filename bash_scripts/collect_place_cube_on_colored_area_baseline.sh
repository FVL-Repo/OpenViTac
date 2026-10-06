#!/usr/bin/env bash

# Baseline data collection for envs/place_cube_on_colored_area.py.
#
# Benchmark baseline scope
# ------------------------
# Collect four explicit cases:
#   1. place on yellow area, yellow area on the left
#   2. place on yellow area, blue area on the left
#   3. place on blue area,   yellow area on the left
#   4. place on blue area,   blue area on the left
#
# These are the Cartesian product of:
#   target_area = yellow / blue
#   frame_order = yellow_left / blue_left
#
# Reset behavior
# --------------
# envs/place_cube_on_colored_area.py keeps its original reset-time sampling:
# both frames receive the same small xy offset, and the cube receives its own xy
# offset. This script only fixes target_area and frame_order before each
# collect_data.py run.
#
# Default collection size
# -----------------------
#   4 cases x 50 successful episodes = 200 successful episodes
#
# Output layout
# -------------
#   data/place_cube_on_colored_area/${MODALITY}/yellow/yellow_left/
#   data/place_cube_on_colored_area/${MODALITY}/yellow/blue_left/
#   data/place_cube_on_colored_area/${MODALITY}/blue/yellow_left/
#   data/place_cube_on_colored_area/${MODALITY}/blue/blue_left/
#
# Examples
# --------
#   MODALITY=gelsight CONFIG=task_config/gelsight.yml GPU=0 bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#   MODALITY=xense CONFIG=task_config/xense.yml GPU=1 bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#   MODALITY=neote CONFIG=task_config/neote.yml GPU=2 bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#
#   bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#   DRY_RUN=1 bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#   EPISODES_PER_CASE=20 MAX_SEED=199 \
#     bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
#   SKIP_CONDA=1 bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-OpenViTac}"
SKIP_CONDA="${SKIP_CONDA:-0}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

# Run one tactile modality at a time. Data is grouped under:
#   data/place_cube_on_colored_area/${MODALITY}/${target_area}/${frame_order}/
MODALITY="${MODALITY:-xense}"

# Base yaml config. If unset, task_config/${MODALITY}.yml is used.
# Relative paths are resolved from PROJECT_ROOT.
CONFIG="${CONFIG:-}"

EPISODES_PER_CASE="${EPISODES_PER_CASE:-50}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-499}"

RUN_NAME="${RUN_NAME:-place_cube_on_colored_area_baseline_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/collect_place_cube_on_colored_area_baseline}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${RUN_LOG_DIR}/summary.tsv"

# Case format: "target_area:frame_order".
CASES=(
    "yellow:yellow_left"
    "yellow:blue_left"
    "blue:yellow_left"
    "blue:blue_left"
)

valid_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

valid_nonnegative_or_minus_one() {
    [[ "$1" == "-1" || "$1" =~ ^[0-9]+$ ]]
}

validate_modality() {
    local modality="$1"
    case "${modality}" in
        gelsight|xense|neote|neote_force_field)
            return 0
            ;;
        *)
            echo "Unknown modality: ${modality}"
            echo "Valid modalities: gelsight xense neote neote_force_field"
            return 1
            ;;
    esac
}

count_successes() {
    local suc_map="$1"
    if [[ ! -f "${suc_map}" ]]; then
        echo 0
        return 0
    fi
    tr ' ' '\n' < "${suc_map}" | awk '$1 == "1" { count += 1 } END { print count + 0 }'
}

if [[ "${SKIP_CONDA}" != "1" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    # Conda activate/deactivate hooks may reference unset compiler backup
    # variables. Temporarily disable nounset so `set -u` does not break them.
    set +u
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        source /opt/conda/etc/profile.d/conda.sh
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        set -u
        echo "Could not find conda. Set SKIP_CONDA=1 if the correct environment is already active."
        exit 2
    fi
    conda activate "${CONDA_ENV}"
    conda_status=$?
    set -u
    if (( conda_status != 0 )); then
        echo "Failed to activate conda environment: ${CONDA_ENV}"
        exit 2
    fi
fi
cd "${PROJECT_ROOT}"

if ! valid_positive_int "${EPISODES_PER_CASE}"; then
    echo "EPISODES_PER_CASE must be a positive integer."
    exit 2
fi

if ! valid_nonnegative_or_minus_one "${START_SEED}" || ! valid_nonnegative_or_minus_one "${MAX_SEED}"; then
    echo "START_SEED and MAX_SEED must be non-negative integers or -1."
    exit 2
fi

if [[ "${START_SEED}" != "-1" && "${MAX_SEED}" != "-1" ]] && (( MAX_SEED < START_SEED )); then
    echo "MAX_SEED must be greater than or equal to START_SEED, unless MAX_SEED=-1."
    exit 2
fi

validate_modality "${MODALITY}" || exit 2

if [[ -z "${CONFIG}" ]]; then
    SOURCE_CONFIG="${PROJECT_ROOT}/task_config/${MODALITY}.yml"
elif [[ "${CONFIG}" = /* ]]; then
    SOURCE_CONFIG="${CONFIG}"
else
    SOURCE_CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi

if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "Missing yaml config: ${SOURCE_CONFIG}"
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${RUN_LOG_DIR}"

TEMP_CONFIG_DIR="$(mktemp -d /tmp/openvitac-place-cube-area-baseline.XXXXXX)"
cleanup() {
    if [[ "${TEMP_CONFIG_DIR}" == /tmp/openvitac-place-cube-area-baseline.* ]]; then
        rm -rf -- "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

printf 'modality\ttarget_area\tframe_order\tresult\texit_code\tsuccesses\tlog\n' > "${SUMMARY_FILE}"

passed=0
task_failed=0
program_error=0
dry_run_count=0
stop_requested=0
active_child_pid=""

terminate_all() {
    local exit_code="${1:-130}"
    echo
    echo "Termination requested. Stopping current collect_data.py run and exiting..."
    stop_requested=1
    if [[ -n "${active_child_pid}" ]] && kill -0 "${active_child_pid}" 2>/dev/null; then
        kill -TERM "-${active_child_pid}" 2>/dev/null \
            || kill -TERM "${active_child_pid}" 2>/dev/null \
            || true
        sleep 2
        if kill -0 "${active_child_pid}" 2>/dev/null; then
            kill -KILL "-${active_child_pid}" 2>/dev/null \
                || kill -KILL "${active_child_pid}" 2>/dev/null \
                || true
        fi
        wait "${active_child_pid}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}

run_collect_command() {
    local log_file="$1"
    shift
    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" > >(tee "${log_file}") 2>&1 &
        active_child_pid=$!
        wait "${active_child_pid}"
        local status=$?
        active_child_pid=""
        return "${status}"
    fi

    "$@" 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
}

trap 'terminate_all 130' INT
trap 'terminate_all 143' TERM

run_one() {
    local target_area="$1"
    local frame_order="$2"
    local config_stem="${MODALITY}"
    local run_stem="${MODALITY}_${target_area}_${frame_order}"
    local generated_config="${TEMP_CONFIG_DIR}/${config_stem}.yml"
    local output_dir="${OUTPUT_ROOT}/place_cube_on_colored_area/${MODALITY}/${target_area}/${frame_order}"
    local log_file="${RUN_LOG_DIR}/${run_stem}.log"
    local suc_map="${output_dir}/suc_map.txt"
    local suc_map_snapshot="${RUN_LOG_DIR}/${run_stem}.suc_map.txt"
    local exit_code=0
    local result=""
    local successes=0

    # Keep the base modality config intact, only changing save_dir and adding
    # explicit place_cube_on_colored_area controls consumed by collect_data.py.
    sed "s|^save_dir:.*|save_dir: ${OUTPUT_ROOT}|" "${SOURCE_CONFIG}" > "${generated_config}"
    {
        printf '\n'
        printf '# Added by bash_scripts/collect_place_cube_on_colored_area_baseline.sh\n'
        printf 'save_dir_exact: %s\n' "${output_dir}"
        printf 'target_area: %s\n' "${target_area}"
        printf 'frame_order: %s\n' "${frame_order}"
    } >> "${generated_config}"

    local command=(
        python scripts/collect_data.py
        place_cube_on_colored_area
        "${generated_config}"
        --episode_num "${EPISODES_PER_CASE}"
        --start_seed "${START_SEED}"
        --max_seed "${MAX_SEED}"
        --gpu "${GPU}"
    )

    echo
    echo "================================================================"
    echo "Modality:       ${MODALITY}"
    echo "Config:         ${SOURCE_CONFIG}"
    echo "Target area:    ${target_area}"
    echo "Frame order:    ${frame_order}"
    echo "Episodes:       ${EPISODES_PER_CASE} successful episodes requested"
    echo "Seed range:     ${START_SEED}-${MAX_SEED}"
    echo "Output:         ${output_dir}"
    echo "Log:            ${log_file}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%s\t%s\t%s\tDRY_RUN\t0\t0\t%s\n' \
            "${MODALITY}" "${target_area}" "${frame_order}" "${log_file}" \
            >> "${SUMMARY_FILE}"
        ((dry_run_count += 1))
        return 0
    fi

    run_collect_command "${log_file}" "${command[@]}"
    exit_code=$?
    successes="$(count_successes "${suc_map}")"
    if [[ -f "${suc_map}" ]]; then
        cp "${suc_map}" "${suc_map_snapshot}"
    fi

    local log_has_program_error=0
    if grep -qE 'Traceback \(most recent call last\)|Crate file missing|Could not load sublayer|Could not open asset|Boost\.Python\.ArgumentError|ModuleNotFoundError|ImportError' "${log_file}"; then
        log_has_program_error=1
    fi

    if (( exit_code != 0 || log_has_program_error == 1 )); then
        result="PROGRAM_ERROR"
        ((program_error += 1))
    elif (( successes >= EPISODES_PER_CASE )); then
        result="PASS"
        ((passed += 1))
    else
        result="TASK_FAILED"
        ((task_failed += 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${MODALITY}" "${target_area}" "${frame_order}" \
        "${result}" "${exit_code}" "${successes}" "${log_file}" \
        >> "${SUMMARY_FILE}"
    echo "Result: ${result} (${successes}/${EPISODES_PER_CASE} successes)"

    if [[ "${STOP_ON_ERROR}" == "1" && "${result}" != "PASS" ]]; then
        stop_requested=1
    fi
}

echo "Run name:           ${RUN_NAME}"
echo "Modality:           ${MODALITY}"
echo "Config:             ${SOURCE_CONFIG}"
echo "Cases:              ${#CASES[@]}"
echo "Episodes per case:  ${EPISODES_PER_CASE}"
echo "Seed range:         ${START_SEED}-${MAX_SEED}"
echo "Output root:        ${OUTPUT_ROOT}"
echo "Output dirs:        ${OUTPUT_ROOT}/place_cube_on_colored_area/${MODALITY}/{yellow,blue}/{yellow_left,blue_left}"
echo "Log root:           ${RUN_LOG_DIR}"
echo "Dry run:            ${DRY_RUN}"

for case_entry in "${CASES[@]}"; do
    IFS=':' read -r target_area frame_order <<< "${case_entry}"
    run_one "${target_area}" "${frame_order}"
    if (( stop_requested == 1 )); then
        break
    fi
done

echo
echo "============================== Summary =============================="
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${SUMMARY_FILE}"
else
    cat "${SUMMARY_FILE}"
fi
echo
echo "PASS:          ${passed}"
echo "TASK_FAILED:   ${task_failed}"
echo "PROGRAM_ERROR: ${program_error}"
echo "DRY_RUN:       ${dry_run_count}"
echo "Summary file:  ${SUMMARY_FILE}"
echo "Data output:   ${OUTPUT_ROOT}"
