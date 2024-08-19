#!/bin/bash

DEFAULT_VILA_WORK_DIR="/home/$(whoami)/workspace/VILA-Internal"
DEFAULT_VILA_CONDA_ENV="vila"

eval "$(conda shell.bash hook)"

VILA_WORK_DIR=${VILA_WORK_DIR:-$DEFAULT_VILA_WORK_DIR}
echo "VILA_WORK_DIR = $VILA_WORK_DIR"
cd $VILA_WORK_DIR

VILA_CONDA_ENV=${VILA_CONDA_ENV:-$DEFAULT_VILA_CONDA_ENV}
echo "VILA_CONDA_ENV = $VILA_CONDA_ENV"
conda activate $VILA_CONDA_ENV
