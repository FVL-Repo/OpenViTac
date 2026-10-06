#!/usr/bin/env bash

# Collect one successful episode for every selected modality/task pair.
# The default is 4 modalities x 8 tasks = 32 sequential runs on one GPU.
# Each pair tries seeds 0-4 and stops immediately after the first success.
# Data is written directly under ./data; only logs are grouped by run name.
#
# Examples:
#   bash bash_scripts/collect_data.sh
#   DRY_RUN=1 bash bash_scripts/collect_data.sh
#   GPU=1 MODALITIES="xense neote" TASKS="insert_USB pull_drawer" \
#     bash bash_scripts/collect_data.sh
#   MAX_SEED=9 bash bash_scripts/collect_data.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-OpenViTac}"
GPU="${GPU:-0}"
EPISODE_NUM="${EPISODE_NUM:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-4}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

MODALITIES="${MODALITIES:-gelsight xense neote neote_force_field}"
TASKS="${TASKS:-insert_USB insert_block grasp_in_clutter place_cube_on_colored_area pull_drawer pour_ball_to_cup move_cup turn_gear_pair}"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/collect_data}"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${RUN_LOG_DIR}/summary.tsv"

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

if ! [[ "${EPISODE_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODE_NUM must be a positive integer."
    exit 2
fi

if ! [[ "${START_SEED}" =~ ^[0-9]+$ && "${MAX_SEED}" =~ ^[0-9]+$ ]]; then
    echo "START_SEED and MAX_SEED must be non-negative integers."
    exit 2
fi

if (( MAX_SEED < START_SEED )); then
    echo "MAX_SEED must be greater than or equal to START_SEED."
    exit 2
fi

mkdir -p "${RUN_OUTPUT_DIR}" "${RUN_LOG_DIR}"

TEMP_CONFIG_DIR="$(mktemp -d /tmp/openvitac-collect-data.XXXXXX)"
cleanup() {
    if [[ "${TEMP_CONFIG_DIR}" == /tmp/openvitac-collect-data.* ]]; then
        rm -rf -- "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

read -r -a modality_list <<< "${MODALITIES}"
read -r -a task_list <<< "${TASKS}"

for modality in "${modality_list[@]}"; do
    case "${modality}" in
        gelsight|xense|neote|neote_force_field)
            ;;
        *)
            echo "Unknown modality: ${modality}"
            echo "Valid modalities: gelsight xense neote neote_force_field"
            exit 2
            ;;
    esac

    source_config="${PROJECT_ROOT}/task_config/${modality}.yml"
    generated_config="${TEMP_CONFIG_DIR}/${modality}.yml"
    if [[ ! -f "${source_config}" ]]; then
        echo "Missing config: ${source_config}"
        exit 2
    fi

    sed "s|^save_dir:.*|save_dir: ${RUN_OUTPUT_DIR}|" \
        "${source_config}" > "${generated_config}"
done

for task in "${task_list[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/envs/${task}.py" ]]; then
        echo "Missing task: ${PROJECT_ROOT}/envs/${task}.py"
        exit 2
    fi
done

printf 'modality\ttask\tresult\texit_code\tlog\n' > "${SUMMARY_FILE}"

passed=0
task_failed=0
program_error=0
dry_run_count=0
stop_requested=0

run_one() {
    local modality="$1"
    local task="$2"
    local config_file="${TEMP_CONFIG_DIR}/${modality}.yml"
    local log_file="${RUN_LOG_DIR}/${modality}__${task}.log"
    local suc_map="${RUN_OUTPUT_DIR}/${task}/${modality}/suc_map.txt"
    local suc_map_before=""
    local exit_code=0
    local result=""
    local command=(
        python scripts/collect_data.py
        "${task}"
        "${config_file}"
        --episode_num "${EPISODE_NUM}"
        --start_seed "${START_SEED}"
        --max_seed "${MAX_SEED}"
        --gpu "${GPU}"
    )

    echo
    echo "================================================================"
    echo "Modality: ${modality}"
    echo "Task:     ${task}"
    echo "GPU:      ${GPU}"
    echo "Log:      ${log_file}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%s\t%s\tDRY_RUN\t0\t%s\n' \
            "${modality}" "${task}" "${log_file}" >> "${SUMMARY_FILE}"
        ((dry_run_count += 1))
        return 0
    fi

    if [[ -f "${suc_map}" ]]; then
        # A failed simulator initialization can leave an older success marker
        # untouched. Record its metadata so it cannot classify this run.
        suc_map_before="$(stat -c '%i:%s:%Y:%Z' -- "${suc_map}" 2>/dev/null || true)"
    fi

    "${command[@]}" 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}

    local suc_map_after=""
    local suc_map_fresh=0
    if [[ -f "${suc_map}" ]]; then
        suc_map_after="$(stat -c '%i:%s:%Y:%Z' -- "${suc_map}" 2>/dev/null || true)"
        if [[ -z "${suc_map_before}" || "${suc_map_before}" != "${suc_map_after}" ]]; then
            suc_map_fresh=1
        fi
    fi

    # Isaac Sim may report exit code 0 after an exception during shutdown.
    # Treat traceback/asset-load failures as program errors explicitly.
    local log_has_program_error=0
    if grep -qE 'Traceback \(most recent call last\)|Crate file missing|Could not load sublayer|Could not open asset|Boost\.Python\.ArgumentError|ModuleNotFoundError|ImportError' "${log_file}"; then
        log_has_program_error=1
    fi

    if (( exit_code != 0 || log_has_program_error == 1 )); then
        result="PROGRAM_ERROR"
        ((program_error += 1))
    elif (( suc_map_fresh == 1 )) && grep -qw '1' "${suc_map}"; then
        result="PASS"
        ((passed += 1))
    else
        result="TASK_FAILED"
        ((task_failed += 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${modality}" "${task}" "${result}" "${exit_code}" "${log_file}" \
        >> "${SUMMARY_FILE}"
    echo "Result: ${result}"

    if [[ "${STOP_ON_ERROR}" == "1" && "${result}" != "PASS" ]]; then
        stop_requested=1
    fi
}

echo "Run name:       ${RUN_NAME}"
echo "Modalities:     ${MODALITIES}"
echo "Tasks:          ${TASKS}"
echo "Seed range:     ${START_SEED}-${MAX_SEED}"
echo "Output root:    ${RUN_OUTPUT_DIR}"
echo "Log root:       ${RUN_LOG_DIR}"
echo "Dry run:        ${DRY_RUN}"

for modality in "${modality_list[@]}"; do
    for task in "${task_list[@]}"; do
        run_one "${modality}" "${task}"
        if (( stop_requested == 1 )); then
            break 2
        fi
    done
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
echo "Data output:   ${RUN_OUTPUT_DIR}"

if (( program_error > 0 )); then
    exit 1
fi

if (( task_failed > 0 )); then
    exit 2
fi

exit 0
