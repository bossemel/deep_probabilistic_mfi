#!/bin/bash

set -e

# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float32 --batch_size 100000 --resample_masks > logs/mfi_3_float32_resample.txt
# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float16 --batch_size 200000 --resample_masks > logs/mfi_3_float16_resample.txt
# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float16 --batch_size 200000 > logs/mfi_3_float16.txt
# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float16 --batch_size 200000 --compile_model > logs/mfi_3_float16_compile.txt
# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float16 --batch_size 200000 --compile_model --quantize > logs/mfi_3_float16_compile_quantize.txt
# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 3 --dtype float16 --batch_size 200000 --compile_model --quantize --autocast > logs/mfi_3_float16_compile_quantize_autocast.txt

sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/ --num_factors 3 --dtype float16 --batch_size 200000 --compile_model --cudnn_benchmark
sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/ --num_factors 3 --dtype float16 --batch_size 200000 --compile_model --autocast

sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/ --num_factors 4 --dtype float16 --batch_size 400000 --compile_model
sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/ --num_factors 4 --dtype float16 --batch_size 300000 --save_every 2000 --compile_model

# sbatch scripts/mfi/slurm_run_mfi.sh 0 --num_factors 5 --dtype float16 --batch_size 400000 --compile_model

sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/ --num_factors 3 --dtype float16 --batch_size 200000 --compile_model

sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/quantized/quant --num_factors 3 --dtype float16 --batch_size 200000 --compile_model

sbatch scripts/mfi/slurm_run_mfi.sh 0 ../../outputs/networks/quantized/int8 --num_factors 3 --dtype float16 --batch_size 200000 --compile_model
