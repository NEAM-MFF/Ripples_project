#!/bin/bash
#SBATCH --job-name=write_filt_natim
#SBATCH --array=0-44%23
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/06_write_filtered_natim_%A_%a.out
#SBATCH -e output/06_write_filtered_natim_%A_%a.err
#SBATCH --nodes=1

# STAGE 0c - NATIM. Requires: 02 finished, and 04 (thresholds.pkl) written.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID NATIM
