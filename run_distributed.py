#!/bin/bash

# load environment vars from .env file
set -o allexport
source .env
set +o allexport

# paths to the script on each device
PI0_PATH="/home/cc/programs/DistributedMobilenetV2-Pis/distributed_mobilenetv2.py"
PI1_PATH="/home/cc/programs/DistributedMobilenetV2-Pis/distributed_mobilenetv2.py"

# run on pi 0 (rank 0)
ssh pi@$PI0_IP "
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr='$MASTER_ADDR' --master_port=29500 $PI0_PATH
" &

# run on pi 1 (rank 1)
ssh pi@$PI1_IP "
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr='$MASTER_ADDR' --master_port=29500 $PI1_PATH
" &

wait
