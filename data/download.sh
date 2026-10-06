#!/bin/bash

# Download the dataset using modelscope
if ! command -v modelscope &> /dev/null
then
    pip install modelscope
fi

modelscope download --dataset byml2024/OpenViTac