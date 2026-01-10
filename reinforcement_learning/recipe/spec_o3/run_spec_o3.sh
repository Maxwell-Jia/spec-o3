#!/bin/bash

set -x

ray stop

# export RAY_DEBUG_POST_MORTEM=1
export VERL_LOGGING_LEVEL=WARNING

TIME_NOW=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="project-name"
EXPERIMENT_NAME="experiment-name"

BASEDIR=reinforcement_learning/verl/

# Get the current machine's IP address dynamically
NODE_IP=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address="${NODE_IP}" --port=6379 --num-gpus 8

# Create logs directory if it doesn't exist
mkdir -p ${BASEDIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=${BASEDIR}/recipe/astro/configs \
    --config-name='astro_multi_turn_grpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.rollout_data_dir=${BASEDIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.validation_data_dir=${BASEDIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    2>&1 | tee ./logs/${PROJECT_NAME}/${EXPERIMENT_NAME}.log

ray stop
