#!/bin/bash
#SBATCH -A BMAI-CDT-SL2-GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p ampere
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s1885912@ed.ac.uk
#SBATCH --job-name test
#SBATCH --output log.%j.log
#SBATCH --gres=gpu:1

source ~/miniconda3/bin/activate discrete_flows

bash scripts/run_pixelcnn.sh
