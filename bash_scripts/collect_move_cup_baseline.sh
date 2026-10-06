#!/usr/bin/env bash

# Baseline data collection for envs/move_cup.py.
#
# Benchmark baseline scope
# ------------------------
# This script intentionally collects a compact, balanced baseline rather than
# every possible cup relation. It uses four semantic variants:
#   blue   left_of_yellow
#   blue   right_of_yellow
#   yellow left_of_blue
#   yellow right_of_blue
#
# The green cup remains in the scene as a distractor, but is not used as target
# or reference in this baseline.
#
# Layouts
# -------
# Layout entries use CUP_COLORS order from envs/move_cup.py:
#   (yellow, green, blue)
#
# Anchor indices in envs/move_cup.py:
#   0: (0.36, 0.00, 0.002)
#   1: (0.48, 0.00, 0.002)
#   2: (0.60, 0.00, 0.002)
#
# The default three layouts make yellow and blue each visit all three anchors:
#   layout0: yellow=0, green=1, blue=2
#   layout1: yellow=1, green=2, blue=0
#   layout2: yellow=2, green=0, blue=1
#
# Reset behavior
# --------------
# envs/move_cup.py still performs its original reset-time sampling: each cup is
# placed at its assigned layout anchor plus small xy noise, and samples are
# rejected if cups are too close. This script only chooses the fixed layout
# before each collect_data.py run; it does not reshuffle large positions inside a
# single environment reset.
#
# Default collection size
# -----------------------
#   4 semantic variants x 3 layouts x 20 successful episodes
#   = 240 successful episodes
#
# Output layout
# -------------
#   data/move_cup/${MODALITY}/blue/left_of_yellow/
#   data/move_cup/${MODALITY}/blue/right_of_yellow/
#   data/move_cup/${MODALITY}/yellow/left_of_blue/
#   data/move_cup/${MODALITY}/yellow/right_of_blue/
#
# Examples
# --------
#   MODALITY=gelsight CONFIG=task_config/gelsight.yml GPU=0 bash bash_scripts/collect_move_cup_baseline.sh
#   MODALITY=xense CONFIG=task_config/xense.yml GPU=1 bash bash_scripts/collect_move_cup_baseline.sh
#   MODALITY=neote CONFIG=task_config/neote.yml GPU=2 bash bash_scripts/collect_move_cup_baseline.sh
#
#   bash bash_scripts/collect_move_cup_baseline.sh
#   DRY_RUN=1 bash bash_scripts/collect_move_cup_baseline.sh
#   EPISODES_PER_LAYOUT=10 MAX_SEED_OFFSET=99 \
#     bash bash_scripts/collect_move_cup_baseline.sh
#   SKIP_CONDA=1 bash bash_scripts/collect_move_cup_baseline.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-OpenViTac}"
SKIP_CONDA="${SKIP_CONDA:-0}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

# Run one tactile modality at a time. Data is grouped by semantic variant under:
#   data/move_cup/${MODALITY}/${target}/${side}_of_${reference}/
MODALITY="${MODALITY:-xense}"

# Base yaml config. If unset, task_config/${MODALITY}.yml is used.
# Relative paths are resolved from PROJECT_ROOT.
CONFIG="${CONFIG:-}"

EPISODES_PER_LAYOUT="${EPISODES_PER_LAYOUT:-20}"
START_SEED="${START_SEED:-0}"

# Layouts within the same semantic output directory share hdf5/<seed>.hdf5
# naming, so each layout receives a non-overlapping seed range:
#   subrun_start_seed = START_SEED + layout_index * SEED_STRIDE
#   subrun_max_seed   = subrun_start_seed + MAX_SEED_OFFSET
#
# MAX_SEED is accepted as a backward-compatible alias for MAX_SEED_OFFSET.
MAX_SEED_OFFSET="${MAX_SEED_OFFSET:-${MAX_SEED:-199}}"
SEED_STRIDE="${SEED_STRIDE:-1000}"

RUN_NAME="${RUN_NAME:-move_cup_baseline_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/data}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/collect_move_cup_baseline}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${RUN_LOG_DIR}/summary.tsv"

# Semantic format: "target:side:reference".
SEMANTICS=(
    "blue:left:yellow"
    "blue:right:yellow"
    "yellow:left:blue"
    "yellow:right:blue"
)

# Layout format: "layout_name:yellow_anchor,green_anchor,blue_anchor".
LAYOUTS=(
    "layout0:0,1,2"
    "layout1:1,2,0"
    "layout2:2,0,1"
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

if ! valid_positive_int "${EPISODES_PER_LAYOUT}"; then
    echo "EPISODES_PER_LAYOUT must be a positive integer."
    exit 2
fi

if ! valid_nonnegative_or_minus_one "${START_SEED}" || ! valid_nonnegative_or_minus_one "${MAX_SEED_OFFSET}"; then
    echo "START_SEED and MAX_SEED_OFFSET must be non-negative integers or -1."
    exit 2
fi

if ! valid_positive_int "${SEED_STRIDE}"; then
    echo "SEED_STRIDE must be a positive integer."
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

TEMP_CONFIG_DIR="$(mktemp -d /tmp/openvitac-move-cup-baseline.XXXXXX)"
cleanup() {
    if [[ "${TEMP_CONFIG_DIR}" == /tmp/openvitac-move-cup-baseline.* ]]; then
        rm -rf -- "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

printf 'modality\ttarget_cup\treference_cup\tplacement_side\tlayout\tpose_indices\tresult\texit_code\tsuccesses\tlog\n' > "${SUMMARY_FILE}"

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
    local target="$1"
    local side="$2"
    local reference="$3"
    local layout_name="$4"
    local pose_indices="$5"
    local layout_index="$6"
    local config_stem="${MODALITY}"
    local semantic_dir="${side}_of_${reference}"
    local run_stem="${MODALITY}_${target}_${semantic_dir}_${layout_name}"
    local generated_config="${TEMP_CONFIG_DIR}/${config_stem}.yml"
    local output_dir="${OUTPUT_ROOT}/move_cup/${MODALITY}/${target}/${semantic_dir}"
    local log_file="${RUN_LOG_DIR}/${run_stem}.log"
    local suc_map="${output_dir}/suc_map.txt"
    local suc_map_snapshot="${RUN_LOG_DIR}/${run_stem}.suc_map.txt"
    local subrun_start_seed="${START_SEED}"
    local subrun_max_seed="${MAX_SEED_OFFSET}"
    local exit_code=0
    local result=""
    local successes=0

    if [[ "${START_SEED}" != "-1" ]]; then
        subrun_start_seed=$(( START_SEED + layout_index * SEED_STRIDE ))
        if [[ "${MAX_SEED_OFFSET}" == "-1" ]]; then
            subrun_max_seed="-1"
        else
            subrun_max_seed=$(( subrun_start_seed + MAX_SEED_OFFSET ))
        fi
    fi

    # Keep the base modality config intact, only changing save_dir and adding
    # explicit move_cup controls consumed by scripts/collect_data.py.
    sed "s|^save_dir:.*|save_dir: ${OUTPUT_ROOT}|" "${SOURCE_CONFIG}" > "${generated_config}"
    {
        printf '\n'
        printf '# Added by bash_scripts/collect_move_cup_baseline.sh\n'
        printf 'save_dir_exact: %s\n' "${output_dir}"
        printf 'target_cup: %s\n' "${target}"
        printf 'reference_cup: %s\n' "${reference}"
        printf 'placement_side: %s\n' "${side}"
        printf 'cup_base_pose_indices: [%s]\n' "${pose_indices}"
    } >> "${generated_config}"

    local command=(
        python scripts/collect_data.py
        move_cup
        "${generated_config}"
        --episode_num "${EPISODES_PER_LAYOUT}"
        --start_seed "${subrun_start_seed}"
        --max_seed "${subrun_max_seed}"
        --gpu "${GPU}"
    )

    echo
    echo "================================================================"
    echo "Modality:       ${MODALITY}"
    echo "Config:         ${SOURCE_CONFIG}"
    echo "Target cup:     ${target}"
    echo "Reference cup:  ${reference}"
    echo "Placement side: ${side}"
    echo "Layout:         ${layout_name}"
    echo "Pose indices:   ${pose_indices}  (yellow, green, blue)"
    echo "Episodes:       ${EPISODES_PER_LAYOUT} successful episodes requested"
    echo "Seed range:     ${subrun_start_seed}-${subrun_max_seed}"
    echo "Output:         ${output_dir}"
    echo "Log:            ${log_file}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%s\t%s\t%s\t%s\t%s\t%s\tDRY_RUN\t0\t0\t%s\n' \
            "${MODALITY}" "${target}" "${reference}" "${side}" \
            "${layout_name}" "${pose_indices}" "${log_file}" \
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
    elif (( successes >= EPISODES_PER_LAYOUT )); then
        result="PASS"
        ((passed += 1))
    else
        result="TASK_FAILED"
        ((task_failed += 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${MODALITY}" "${target}" "${reference}" "${side}" \
        "${layout_name}" "${pose_indices}" "${result}" "${exit_code}" \
        "${successes}" "${log_file}" \
        >> "${SUMMARY_FILE}"
    echo "Result: ${result} (${successes}/${EPISODES_PER_LAYOUT} successes)"

    if [[ "${STOP_ON_ERROR}" == "1" && "${result}" != "PASS" ]]; then
        stop_requested=1
    fi
}

echo "Run name:             ${RUN_NAME}"
echo "Modality:             ${MODALITY}"
echo "Config:               ${SOURCE_CONFIG}"
echo "Semantic variants:    ${#SEMANTICS[@]}"
echo "Layouts:              ${#LAYOUTS[@]}"
echo "Episodes per layout:  ${EPISODES_PER_LAYOUT}"
echo "First start seed:     ${START_SEED}"
echo "Max seed offset:      ${MAX_SEED_OFFSET}"
echo "Seed stride:          ${SEED_STRIDE}"
echo "Output root:          ${OUTPUT_ROOT}"
echo "Output directories:   ${OUTPUT_ROOT}/move_cup/${MODALITY}/{blue,yellow}/{left_of_*,right_of_*}"
echo "Log root:             ${RUN_LOG_DIR}"
echo "Dry run:              ${DRY_RUN}"

for semantic in "${SEMANTICS[@]}"; do
    IFS=':' read -r target side reference <<< "${semantic}"
    layout_index=0
    for layout_entry in "${LAYOUTS[@]}"; do
        layout_name="${layout_entry%%:*}"
        pose_indices="${layout_entry#*:}"
        run_one "${target}" "${side}" "${reference}" "${layout_name}" "${pose_indices}" "${layout_index}"
        ((layout_index += 1))
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
echo "Data output:   ${OUTPUT_ROOT}"
