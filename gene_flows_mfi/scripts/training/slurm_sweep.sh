#!/bin/bash
#SBATCH -A BMAI-CDT-SL2-GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s1885912@ed.ac.uk
#SBATCH --job-name sweep_2
#SBATCH --output log.%j.log
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4

source ~/miniconda3/bin/activate discrete_flows

cd scripts
python3 sweep.py
