#!/bin/bash
set -e

CONDA_ENV_NAME="OpenViTac"
RUN_SMOKE_TESTS="${RUN_SMOKE_TESTS:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
load_env() {
    if [ -z "$CONDA_PREFIX" ]; then
        echo "Conda environment is not activated. Please activate the conda environment and run the script again."
        exit 1
    fi
    conda_base=$(conda info --base)

    source ~/.bashrc
    source ${conda_base}/etc/profile.d/conda.sh

    # step 1: create/update conda environment
    if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
        echo "Creating conda environment '${CONDA_ENV_NAME}'..."
        conda create -n ${CONDA_ENV_NAME} python=3.10 -y
    fi
    conda env update -n ${CONDA_ENV_NAME} --file "${REPO_ROOT}/third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml"
    conda activate ${CONDA_ENV_NAME}

    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
}
export -f load_env
export REPO_ROOT

load_env
export python_exe=${CONDA_PREFIX}/bin/python
export pip_exe=${CONDA_PREFIX}/bin/pip

if [ ! -f "${CONDA_PREFIX}/bin/uv" ]; then
    ${pip_exe} install uv
fi
export uv_exe=${CONDA_PREFIX}/bin/uv

echo "Using Python executable: ${python_exe}"
echo "Using uv executable: ${uv_exe}"

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib

${pip_exe} install --upgrade 'pip<26'
${uv_exe} pip install 'setuptools<82' wheel
if ! command -v ffmpeg >/dev/null 2>&1; then
    conda install -n ${CONDA_ENV_NAME} --override-channels -c defaults ffmpeg -y
fi

# step 2: install isaacsim
if ${pip_exe} show isaacsim >/dev/null 2>&1; then
    echo "isaacsim is already installed. Skipping installation..."
else
    echo "Installing isaacsim..."
    ${uv_exe} pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
    ${uv_exe} pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
fi

# step 3: install isaaclab
if ${pip_exe} show isaaclab >/dev/null 2>&1; then
    echo "isaaclab is already installed. Skipping installation..."
else
    echo "Installing isaaclab..."
    # install dependencies via apt (Ubuntu)
    sudo apt install -y cmake build-essential

    cd third_party
    if [ -d "IsaacLab" ]; then
        cd IsaacLab
    else
        git clone https://github.com/isaac-sim/IsaacLab
        cd IsaacLab
    fi
    git checkout v2.1.1
    ${uv_exe} pip install flatdict==4.0.1 --no-build-isolation
    ./isaaclab.sh --install
    cd ../..
fi 

# step 4: install curobo
if ${pip_exe} show nvidia_curobo >/dev/null 2>&1; then
    echo "curobo is already installed. Skipping installation..."
else
    echo "Installing curobo..."
    sudo apt install git-lfs
    
    cd third_party
    if [ -d "curobo" ]; then
        cd curobo
    else
        git clone https://github.com/NVlabs/curobo.git
        cd curobo
    fi
    # checkout to v0.7.7
    git checkout 0a50de1ba72db304195d59d9d0b1ed269696047f

    if ${uv_exe} pip list 2>/dev/null | grep -q "torch"; then
        torch_version=$(${pip_exe} show torch 2>/dev/null | grep "Version:" | awk '{print $2}')
        echo "[INFO] Found PyTorch version ${torch_version} installed."
        if [[ "${torch_version}" != "2.5.1+cu124" ]]; then
            echo "[INFO] Uninstalling PyTorch version ${torch_version}..."
            ${uv_exe} pip uninstall -y torch torchvision torchaudio
            ${uv_exe} pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
        else
            echo "[INFO] PyTorch 2.5.1 is already installed."
        fi
    fi
    ${uv_exe} pip install warp-lang==1.0.0 --no-build-isolation
    ${uv_exe} pip install -e . --no-build-isolation

    if [ "${RUN_SMOKE_TESTS}" = "1" ]; then
        echo "Running curobo tests..."
        python3 -m pytest .
    fi
    cd ../..
fi

# step 5: install tacex (without libuipc)
if ${pip_exe} show tacex >/dev/null 2>&1; then
    echo "tacex is already installed. Skipping installation..."
else
    echo "Installing tacex..."
    cd third_party/TacEx
    ./tacex.sh -i
    ${uv_exe} pip uninstall torch_scatter -y
    ${uv_exe} pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

    if [ "${RUN_SMOKE_TESTS}" = "1" ]; then
        echo "Running tacex tests..."
        python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 1 --enable_cameras --headless
    fi
    cd ../..
fi

# step 6: install libuipc and tacex_uipc
if ${pip_exe} show libuipc >/dev/null 2>&1; then
    echo "libuipc is already installed. Skipping installation..."
else
    echo "Installing libuipc..."

    current_dir=$(pwd)
    sudo apt install -y cmake build-essential

    if [ ! -d "$HOME/Toolchain/vcpkg" ]; then
        mkdir -p ~/Toolchain
        cd ~/Toolchain
        git clone https://github.com/microsoft/vcpkg.git
        cd vcpkg
        ./bootstrap-vcpkg.sh -disableMetrics
    else
        echo "vcpkg is already installed. Skipping cloning vcpkg..."
    fi

    export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
    if ! grep -q "CMAKE_TOOLCHAIN_FILE" ~/.bashrc; then
        echo "export CMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE" >> ~/.bashrc
    else
        sed -i "s|^export CMAKE_TOOLCHAIN_FILE=.*$|export CMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE|g" ~/.bashrc
    fi
    
    load_env
    cd ${current_dir}/third_party/TacEx
    if [ -d "source/tacex_uipc/build" ]; then
        rm -rf source/tacex_uipc/build
    fi
    ${uv_exe} pip install -e source/tacex_uipc -v --no-build-isolation
fi

${uv_exe} pip install transforms3d trimesh tetgen

echo "Installation completed successfully!"
if [ "${RUN_SMOKE_TESTS}" = "1" ]; then
    echo "Trying to collect data for grasp classification demo."
    bash collect_data.sh grasp_classify demo 0
fi
