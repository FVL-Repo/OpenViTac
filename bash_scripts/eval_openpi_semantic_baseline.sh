#!/usr/bin/env bash

# OpenPI semantic baseline eval for OpenViTac tasks.
#
# Supported tasks:
#   grasp_in_clutter
#   insert_block
#   move_cup
#   place_cube_on_colored_area
#
# Examples:
#   TASK=insert_block MODALITY=gelsight GPU=0 \
#     DEPLOY_CONFIG=openpi/eef_delta/deploy_eef_delta_vision \
#     bash bash_scripts/eval_openpi_semantic_baseline.sh
#
#   TASK=move_cup MODALITY=xense TOTAL_NUM_PER_CASE=10 DRY_RUN=1 \
#     bash bash_scripts/eval_openpi_semantic_baseline.sh

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-OpenViTac}"
SKIP_CONDA="${SKIP_CONDA:-0}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

TASK="${TASK:-insert_block}"
MODALITY="${MODALITY:-xense}"
CONFIG="${CONFIG:-}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-openpi/eef_delta/deploy_eef_delta_vision}"
TOTAL_NUM_PER_CASE="${TOTAL_NUM_PER_CASE:-20}"
START_SEED="${START_SEED:-0}"
MAX_SEED_OFFSET="${MAX_SEED_OFFSET:-${MAX_SEED:-199}}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
EXPERT_CHECK="${EXPERT_CHECK:-0}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
EVAL_VIDEO_FREQUENCY="${EVAL_VIDEO_FREQUENCY:-2}"
EVAL_SAVE_FREQUENCY="${EVAL_SAVE_FREQUENCY:-2}"
OPENPI_DEBUG_DUMP_FIRST_N_OBS="${OPENPI_DEBUG_DUMP_FIRST_N_OBS:-}"

RUN_NAME="${RUN_NAME:-eval_openpi_${TASK}_${MODALITY}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/eval_openpi_semantic_baseline}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
SUMMARY_FILE="${RUN_LOG_DIR}/summary.tsv"

valid_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

valid_nonnegative_or_minus_one() {
    [[ "$1" == "-1" || "$1" =~ ^[0-9]+$ ]]
}

validate_modality() {
    case "$1" in
        gelsight|xense|neote|neote_force_field)
            return 0
            ;;
        *)
            echo "Unknown modality: $1"
            echo "Valid modalities: gelsight xense neote neote_force_field"
            return 1
            ;;
    esac
}

target_to_grasp_block_name() {
    case "$1" in
        half_cylinder)
            echo "block_blue_half_cylinder"
            ;;
        cylinder)
            echo "block_yellow_cylinder"
            ;;
        hexagonal_prism)
            echo "block_red_hexagonal_prism"
            ;;
        *)
            return 1
            ;;
    esac
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
export CUDA_VISIBLE_DEVICES="${GPU}"

validate_modality "${MODALITY}" || exit 2
if ! valid_positive_int "${TOTAL_NUM_PER_CASE}"; then
    echo "TOTAL_NUM_PER_CASE must be a positive integer."
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

resolve_deploy_config() {
    local deploy_config="$1"
    local deploy_file=""
    if [[ "${deploy_config}" == *.yml || "${deploy_config}" == *.yaml ]]; then
        deploy_file="${PROJECT_ROOT}/policy/${deploy_config}"
    else
        deploy_file="${PROJECT_ROOT}/policy/${deploy_config}.yml"
    fi
    if [[ -f "${deploy_file}" ]]; then
        echo "${deploy_config}"
        return 0
    fi

    case "${deploy_config}" in
        openpi/eef_delta/deploy_vision)
            echo "openpi/eef_delta/deploy_eef_delta_vision"
            return 0
            ;;
        openpi/eef_delta/deploy_vision_tactile)
            echo "openpi/eef_delta/deploy_eef_delta_vision_tactile"
            return 0
            ;;
    esac

    echo "Missing deploy config: ${deploy_file}" >&2
    return 1
}

DEPLOY_CONFIG="$(resolve_deploy_config "${DEPLOY_CONFIG}")" || exit 2

mkdir -p "${RUN_LOG_DIR}"
TEMP_CONFIG_DIR="$(mktemp -d /tmp/openvitacorge-openpi-eval.XXXXXX)"
cleanup() {
    if [[ "${TEMP_CONFIG_DIR}" == /tmp/openvitacorge-openpi-eval.* ]]; then
        rm -rf -- "${TEMP_CONFIG_DIR}"
    fi
}
trap cleanup EXIT

active_child_pid=""
stop_requested=0
passed=0
failed=0
program_error=0
dry_run_count=0

terminate_all() {
    local exit_code="${1:-130}"
    echo
    echo "Termination requested. Stopping current eval_policy.py run and exiting..."
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

run_eval_command() {
    local log_file="$1"
    shift
    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" > >(stdbuf -oL -eL tee "${log_file}") 2>&1 &
        active_child_pid=$!
        wait "${active_child_pid}"
        local status=$?
        active_child_pid=""
        return "${status}"
    fi

    "$@" 2>&1 | stdbuf -oL -eL tee "${log_file}"
    return "${PIPESTATUS[0]}"
}

trap 'terminate_all 130' INT
trap 'terminate_all 143' TERM

write_config_header() {
    local generated_config="$1"
    local eval_case_name="$2"
    sed "s|^save_dir:.*|save_dir: ./data|" "${SOURCE_CONFIG}" > "${generated_config}"
    {
        printf '\n'
        printf '# Added by bash_scripts/eval_openpi_semantic_baseline.sh\n'
        printf 'eval_case_name: %s\n' "${eval_case_name}"
        printf 'video_frequency: %s\n' "${EVAL_VIDEO_FREQUENCY}"
        printf 'save_frequency: %s\n' "${EVAL_SAVE_FREQUENCY}"
    } >> "${generated_config}"
}

parse_final_successes() {
    local log_file="$1"
    local final_line=""
    final_line="$(grep -E 'Final Result: [0-9]+/[0-9]+' "${log_file}" | tail -1 || true)"
    if [[ -z "${final_line}" ]]; then
        echo "0/0"
        return 0
    fi
    echo "${final_line}" | sed -E 's/.*Final Result: ([0-9]+)\/([0-9]+).*/\1\/\2/'
}

run_one() {
    local case_name="$1"
    local config_append="$2"
    local case_index="$3"
    local generated_config="${TEMP_CONFIG_DIR}/${TASK}_${case_name}.yml"
    local log_file="${RUN_LOG_DIR}/${case_name}.log"
    local subrun_start_seed="${START_SEED}"
    local subrun_max_seed="${MAX_SEED_OFFSET}"
    local exit_code=0
    local score="0/0"
    local result=""

    if [[ "${START_SEED}" != "-1" ]]; then
        subrun_start_seed=$(( START_SEED + case_index * SEED_STRIDE ))
        if [[ "${MAX_SEED_OFFSET}" == "-1" ]]; then
            subrun_max_seed="-1"
        else
            subrun_max_seed=$(( subrun_start_seed + MAX_SEED_OFFSET ))
        fi
    fi

    write_config_header "${generated_config}" "${case_name}"
    printf '%s\n' "${config_append}" >> "${generated_config}"
    if [[ -n "${OPENPI_DEBUG_DUMP_FIRST_N_OBS}" ]]; then
        {
            printf '\n'
            printf '# OpenPI eval override\n'
            printf 'openpi_debug_dump_first_n_obs: %s\n' "${OPENPI_DEBUG_DUMP_FIRST_N_OBS}"
        } >> "${generated_config}"
    fi

    local command=(
        python -u scripts/eval_policy.py
        "${TASK}"
        "${generated_config}"
        "${DEPLOY_CONFIG}"
        --total_num "${TOTAL_NUM_PER_CASE}"
        --start_seed "${subrun_start_seed}"
        --max_seed "${subrun_max_seed}"
        --tactile_sensor "${MODALITY}"
    )
    if [[ "${EXPERT_CHECK}" == "1" ]]; then
        command+=(--expert_check)
    fi
    if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
        read -r -a extra_args <<< "${EXTRA_EVAL_ARGS}"
        command+=("${extra_args[@]}")
    fi

    echo
    echo "================================================================"
    echo "Task:             ${TASK}"
    echo "Modality:         ${MODALITY}"
    echo "Deploy config:    ${DEPLOY_CONFIG}"
    echo "Case:             ${case_name}"
    echo "Total evals:      ${TOTAL_NUM_PER_CASE}"
    echo "Seed range:       ${subrun_start_seed}-${subrun_max_seed}"
    echo "Generated config: ${generated_config}"
    echo "Log:              ${log_file}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY RUN: '
        printf '%q ' "${command[@]}"
        printf '\n'
        printf '%s\t%s\t%s\tDRY_RUN\t0\t0/0\t%s\n' \
            "${TASK}" "${MODALITY}" "${case_name}" "${log_file}" >> "${SUMMARY_FILE}"
        ((dry_run_count += 1))
        return 0
    fi

    run_eval_command "${log_file}" "${command[@]}"
    exit_code=$?
    score="$(parse_final_successes "${log_file}")"

    local log_has_program_error=0
    if grep -qE 'Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|OpenPI server 返回|OpenPI websocket|RuntimeError|ValueError|KeyError' "${log_file}"; then
        log_has_program_error=1
    fi

    if (( exit_code != 0 || log_has_program_error == 1 )); then
        result="PROGRAM_ERROR"
        ((program_error += 1))
    elif [[ "${score}" == "${TOTAL_NUM_PER_CASE}/${TOTAL_NUM_PER_CASE}" ]]; then
        result="PASS"
        ((passed += 1))
    else
        result="FAILED"
        ((failed += 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TASK}" "${MODALITY}" "${case_name}" "${result}" \
        "${exit_code}" "${score}" "${log_file}" >> "${SUMMARY_FILE}"
    echo "Result: ${result} (${score})"

    if [[ "${STOP_ON_ERROR}" == "1" && "${result}" != "PASS" ]]; then
        stop_requested=1
    fi
}

printf 'task\tmodality\tcase\tresult\texit_code\tsuccesses\tlog\n' > "${SUMMARY_FILE}"

echo "Run name:            ${RUN_NAME}"
echo "Task:                ${TASK}"
echo "Modality:            ${MODALITY}"
echo "Config:              ${SOURCE_CONFIG}"
echo "Deploy config:       ${DEPLOY_CONFIG}"
echo "Total num per case:  ${TOTAL_NUM_PER_CASE}"
echo "Start seed:          ${START_SEED}"
echo "Max seed offset:     ${MAX_SEED_OFFSET}"
echo "Seed stride:         ${SEED_STRIDE}"
echo "Log root:            ${RUN_LOG_DIR}"
echo "Dry run:             ${DRY_RUN}"

case_index=0
case "${TASK}" in
    grasp_in_clutter)
        TARGETS="${TARGETS:-half_cylinder cylinder hexagonal_prism}"
        LAYOUTS=(
            "layout0:0,1,2,3,4,5,6"
            "layout1:2,3,4,5,6,0,1"
            "layout2:4,5,6,0,1,2,3"
            "layout3:6,0,1,2,3,4,5"
        )
        read -r -a target_list <<< "${TARGETS}"
        for target in "${target_list[@]}"; do
            target_block="$(target_to_grasp_block_name "${target}")" || {
                echo "Unknown grasp_in_clutter target: ${target}"
                exit 2
            }
            for layout_entry in "${LAYOUTS[@]}"; do
                layout_name="${layout_entry%%:*}"
                pose_indices="${layout_entry#*:}"
                run_one "${target}_${layout_name}" "$(printf 'target_block: %s\nblock_base_pose_indices: [%s]\n' "${target_block}" "${pose_indices}")" "${case_index}"
                ((case_index += 1))
                (( stop_requested == 1 )) && break 2
            done
        done
        ;;
    insert_block)
        TARGETS="${TARGETS:-cube half_cylinder hexagon}"
        LAYOUTS=(
            "layout0:0,1,4"
            "layout1:1,3,2"
            "layout2:2,4,0"
            "layout3:3,2,1"
        )
        read -r -a target_list <<< "${TARGETS}"
        for target in "${target_list[@]}"; do
            case "${target}" in
                cube|half_cylinder|hexagon) ;;
                *)
                    echo "Unknown insert_block target: ${target}"
                    exit 2
                    ;;
            esac
            for layout_entry in "${LAYOUTS[@]}"; do
                layout_name="${layout_entry%%:*}"
                pose_indices="${layout_entry#*:}"
                run_one "${target}_${layout_name}" "$(printf 'target_block: %s\nblock_base_pose_indices: [%s]\n' "${target}" "${pose_indices}")" "${case_index}"
                ((case_index += 1))
                (( stop_requested == 1 )) && break 2
            done
        done
        ;;
    move_cup)
        SEMANTICS=(
            "blue:left:yellow"
            "blue:right:yellow"
            "yellow:left:blue"
            "yellow:right:blue"
        )
        LAYOUTS=(
            "layout0:0,1,2"
            "layout1:1,2,0"
            "layout2:2,0,1"
        )
        for semantic in "${SEMANTICS[@]}"; do
            IFS=':' read -r target side reference <<< "${semantic}"
            for layout_entry in "${LAYOUTS[@]}"; do
                layout_name="${layout_entry%%:*}"
                pose_indices="${layout_entry#*:}"
                run_one "${target}_${side}_of_${reference}_${layout_name}" "$(printf 'target_cup: %s\nreference_cup: %s\nplacement_side: %s\ncup_base_pose_indices: [%s]\n' "${target}" "${reference}" "${side}" "${pose_indices}")" "${case_index}"
                ((case_index += 1))
                (( stop_requested == 1 )) && break 2
            done
        done
        ;;
    place_cube_on_colored_area)
        CASES=(
            "yellow:yellow_left"
            "yellow:blue_left"
            "blue:yellow_left"
            "blue:blue_left"
        )
        for case_entry in "${CASES[@]}"; do
            IFS=':' read -r target_area frame_order <<< "${case_entry}"
            run_one "${target_area}_${frame_order}" "$(printf 'target_area: %s\nframe_order: %s\n' "${target_area}" "${frame_order}")" "${case_index}"
            ((case_index += 1))
            (( stop_requested == 1 )) && break
        done
        ;;
    *)
        echo "Unsupported TASK=${TASK}"
        echo "Supported tasks: grasp_in_clutter insert_block move_cup place_cube_on_colored_area"
        exit 2
        ;;
esac

echo
echo "============================== Summary =============================="
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${SUMMARY_FILE}"
else
    cat "${SUMMARY_FILE}"
fi
echo
echo "PASS:          ${passed}"
echo "FAILED:        ${failed}"
echo "PROGRAM_ERROR: ${program_error}"
echo "DRY_RUN:       ${dry_run_count}"
echo "Summary file:  ${SUMMARY_FILE}"
