#!/bin/bash

DEFAULT_VILA_WORK_DIR="/home/$(whoami)/workspace/VILA-Internal"
DEFAULT_VILA_CONDA_ENV="vila"
DEFAULT_VILA_SLURM_ACCOUNT="nvr_elm_llm"
DEFAULT_VILA_SLURM_PARTITION="batch_block1,batch_block2,batch_block3,batch_block4"

eval "$(conda shell.bash hook)"

VILA_WORK_DIR=${VILA_WORK_DIR:-$DEFAULT_VILA_WORK_DIR}
echo "VILA_WORK_DIR = $VILA_WORK_DIR"
cd $VILA_WORK_DIR

VILA_CONDA_ENV=${VILA_CONDA_ENV:-$DEFAULT_VILA_CONDA_ENV}
echo "VILA_CONDA_ENV = $VILA_CONDA_ENV"
conda activate $VILA_CONDA_ENV

VILA_SLURM_ACCOUNT=${VILA_SLURM_ACCOUNT:-$DEFAULT_VILA_SLURM_ACCOUNT}
echo "VILA_SLURM_ACCOUNT = $VILA_SLURM_ACCOUNT"

VILA_SLURM_PARTITION=${VILA_SLURM_PARTITION:-$DEFAULT_VILA_SLURM_PARTITION}
echo "VILA_SLURM_PARTITION = $VILA_SLURM_PARTITION"