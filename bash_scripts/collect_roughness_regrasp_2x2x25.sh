#!/usr/bin/env bash

# 2x2x25 data collection for envs/roughness_regrasp.py.
#
# Goal
# ----
# Collect four explicit cases in this order:
#   1. rough on the left,  first grasp left
#   2. rough on the left,  first grasp right
#   3. rough on the right, first grasp left
#   4. rough on the right, first grasp right
#
# Default collection size
# -----------------------
#   4 cases x 25 successful episodes = 100 successful episodes
#
# Output layout
# -------------
#   data_${MODALITY}/classify/roughness_regrasp/rough_left/grasp_left/
#   data_${MODALITY}/classify/roughness_regrasp/rough_left/grasp_right/
#   data_${MODALITY}/classify/roughness_regrasp/rough_right/grasp_left/
#   data_${MODALITY}/classify/roughness_regrasp/rough_right/grasp_right/
#
# Examples
# --------
#   MODALITY=gelsight CONFIG=task_config/gelsight.yml GPU=0 bash bash_scripts/collect_roughness_regrasp_2x2x25.sh
#   MODALITY=xense CONFIG=task_config/xense.yml GPU=1 bash bash_scripts/collect_roughness_regrasp_2x2x25.sh
#
#   bash bash_scripts/collect_roughness_regrasp_2x2x25.sh
#   DRY_RUN=1 bash bash_scripts/collect_roughness_regrasp_2x2x25.sh
#   EPISODES_PER_CASE=20 MAX_SEED=199 bash bash_scripts/collect_roughness_regrasp_2x2x25.sh
#   SKIP_CONDA=1 bash bash_scripts/collect_roughness_regrasp_2x2x25.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-OpenViTac}"
SKIP_CONDA="${SKIP_CONDA:-0}"
GPU="7"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

MODALITY="${MODALITY:-gelsight}"
CONFIG="${CONFIG:-}"

EPISODES_PER_CASE="${EPISODES_PER_CASE:-25}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-499}"

RUN_NAME="${RUN_NAME:-roughness_regrasp_2x2x25_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data_${MODALITY}/classify}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/collect_roughness_regrasp_2x2x25}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${RUN_LOG_DIR}/summary.tsv"

# Case format: "rough_block_side:initial_grasp_side".
CASES=(
    "left:left"
    "left:right"
    "right:left"
    "right:right"
)

valid_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

valid_nonnegative_or_minus_one() {
    [[ "$1" == "-1" || "$1" =~ ^[0-9]+$ ]]
}

validate_side() {
    case "$1" in
        left|right|random)
            return 0
            ;;
        *)
            echo "Unknown side: $1"
            echo "Valid sides: left right random"
            return 1
            ;;
    esac
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

TEMP_CONFIG_DIR="$(mktemp -d /tmp/openvitac-roughness-regrasp-2x2x25.XXXXXX)"
cleanup() {
    if [[ "${TEMP_CONFIG_DIR}" == /tmp/openvitac-roughness-regrasp-2x2x25.* ]]; then
        rm -rf -- "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

printf 'modality\trough_block_side\tinitial_grasp_side\tresult\texit_code\tsuccesses\tlog\n' > "${SUMMARY_FILE}"

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
    local rough_side="$1"
    local grasp_side="$2"

    validate_side "${rough_side}" || return 2
    validate_side "${grasp_side}" || return 2

    local run_stem="${MODALITY}_rough_${rough_side}_grasp_${grasp_side}"
    local generated_config="${TEMP_CONFIG_DIR}/${run_stem}.yml"
    local output_dir="${OUTPUT_ROOT}/roughness_regrasp/rough_${rough_side}/grasp_${grasp_side}"
    local log_file="${RUN_LOG_DIR}/${run_stem}.log"
    local suc_map="${output_dir}/suc_map.txt"
    local suc_map_snapshot="${RUN_LOG_DIR}/${run_stem}.suc_map.txt"
    local exit_code=0
    local result=""
    local successes=0

    sed "s|^save_dir:.*|save_dir: ${OUTPUT_ROOT}|" "${SOURCE_CONFIG}" > "${generated_config}"
    {
        printf '\n'
        printf '# Added by bash_scripts/collect_roughness_regrasp_2x2x25.sh\n'
        printf 'save_dir_exact: %s\n' "${output_dir}"
        printf 'rough_block_side: %s\n' "${rough_side}"
        printf 'initial_grasp_side: %s\n' "${grasp_side}"
    } >> "${generated_config}"

    local command=(
        python scripts/collect_data.py
        roughness_regrasp
        "${generated_config}"
        --episode_num "${EPISODES_PER_CASE}"
        --start_seed "${START_SEED}"
        --max_seed "${MAX_SEED}"
        --gpu "${GPU}"
    )

    echo
    echo "================================================================"
    echo "Modality:             ${MODALITY}"
    echo "Config:               ${SOURCE_CONFIG}"
    echo "Rough block side:     ${rough_side}"
    echo "Initial grasp side:   ${grasp_side}"
    echo "Episodes:             ${EPISODES_PER_CASE} successful episodes requested"
    echo "Seed range:           ${START_SEED}-${MAX_SEED}"
    echo "Output:               ${output_dir}"
    echo "Log:                  ${log_file}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%s\t%s\t%s\tDRY_RUN\t0\t0\t%s\n' \
            "${MODALITY}" "${rough_side}" "${grasp_side}" "${log_file}" \
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
        "${MODALITY}" "${rough_side}" "${grasp_side}" \
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
echo "Output dirs:        ${OUTPUT_ROOT}/roughness_regrasp/rough_{left,right}/grasp_{left,right}"
echo "Log root:           ${RUN_LOG_DIR}"
echo "Dry run:            ${DRY_RUN}"

for case_entry in "${CASES[@]}"; do
    IFS=':' read -r rough_side grasp_side <<< "${case_entry}"
    run_one "${rough_side}" "${grasp_side}"
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
