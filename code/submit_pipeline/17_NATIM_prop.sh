#!/bin/bash
#SBATCH --job-name=NATIM_prop
#SBATCH --array=0-44%5
#SBATCH --mem=1000G
#SBATCH -o output/17_NATIM_prop_%A_%a.out
#SBATCH -e output/17_NATIM_prop_%A_%a.err
#SBATCH --nodes=1

# NATIM pipeline, independent group - one job per (monkey, date), 45 total.
# Requires: 06_write_filtered_natim.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_NATIM_prop_pkl_create.py $SLURM_ARRAY_TASK_ID
