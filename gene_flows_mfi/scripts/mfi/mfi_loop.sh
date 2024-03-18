set -e 

for value_to_save in 0 1 2; do
    sbatch scripts/slurm_run_mfi_3.sh $value_to_save
done
