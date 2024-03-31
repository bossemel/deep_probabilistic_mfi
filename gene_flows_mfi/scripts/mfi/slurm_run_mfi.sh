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
    echo "Usage: $0 <value_to_save> <log_dir> [--quantize] [--compile_model] [--resample_masks] [--autocast] [--dtype <dtype>] [--batch_size <batch_size>] [--cudnn_benchmark <cudnn_benchmark>] [--save_every <save_every>]"
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
save_every=1000

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
        --save_every)
        save_every="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

source ~/miniconda3/bin/activate discrete_flows

cd scripts/mfi

# Construct command based on arguments
# --log_dir ../../outputs/networks/made_sweep_learnings_csd3 
command="python3 mfi_value_pred.py --sequence_length 1000 --device cuda --log_dir $log_dir --value_to_save $value_to_save --save_every $save_every"

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

if [ "$cudnn_benchmark" = true ]; then
    command+=" --cudnn_benchmark"
fi

if [ -n "$dtype" ]; then
    command+=" --dtype $dtype"
fi

# Execute the constructed command
echo $command
eval $command