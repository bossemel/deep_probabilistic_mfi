#!/bin/bash

#SBATCH -A BMAI-CDT-SL2-GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p ampere
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s1885912@ed.ac.uk
#SBATCH --job-name test
#SBATCH --output log.%j.log
#SBATCH --gres=gpu:1

set -e

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <value_to_save> <log_dir> [--quantize] [--compile_model] [--resample_masks] [--autocast] [--dtype <dtype>] [--batch_size <batch_size>] [--cudnn_benchmark <cudnn_benchmark>]"
    exit 1
fi

value_to_save=$1
log_dir=$2
shift
shift

# Parse optional arguments
num_factors=""
quantize=false
compile_model=false
resample_masks=false
autocast=false
cudnn_benchmark=false
dtype=""
batch_size=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --num_factors)
        num_factors="$2"
        shift
        shift
        ;;
        --quantize)
        quantize=true
        shift
        ;;
        --compile_model)
        compile_model=true
        shift
        ;;
        --resample_masks)
        resample_masks=true
        shift
        ;;
        --autocast)
        autocast=true
        shift
        ;;
        --cudnn_benchmark)
        cudnn_benchmark=true
        shift
        ;;
        --dtype)
        dtype="$2"
        shift
        shift
        ;;
        --batch_size)
        batch_size="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# source ~/miniconda3/bin/activate discrete_flows

cd scripts
# python3 made_genes_data.py --device cuda --log_dir ../tmp/run/made_sweep_learnings_csd3 --batch_size 128 --use_wandb True --n_epochs 20 --weight_decay 0.000001 --hidden_dims 2000 --num_layers 2 --n_masks 1 --lr 0.01 --step_size 30 --value_to_save $value_to_save

cd mfi

# Construct command based on arguments
# --log_dir ../../outputs/networks/made_sweep_learnings_csd3 
command="python3 mfi_value_pred.py --sequence_length 1000 --device cuda --log_dir $log_dir --value_to_save $value_to_save"

if [ -n "$num_factors" ]; then
    command+=" --num_factors $num_factors"
fi

if [ -n "$batch_size" ]; then
    command+=" --batch_size $batch_size"
fi

if [ "$quantize" = true ]; then
    command+=" --quantize"
fi

if [ "$compile_model" = true ]; then
    command+=" --compile_model"
fi

if [ "$resample_masks" = true ]; then
    command+=" --resample_masks"
fi

if [ "$autocast" = true ]; then
    command+=" --autocast"
fi

if [ "$autocast" = true ]; then
    command+=" --cudnn_benchmark"
fi

if [ -n "$dtype" ]; then
    command+=" --dtype $dtype"
fi

# Execute the constructed command
eval $command
