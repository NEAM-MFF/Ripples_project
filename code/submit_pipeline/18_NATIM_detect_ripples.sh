#!/bin/bash
#SBATCH --job-name=NATIM_ripples
#SBATCH --array=0-31%5
#SBATCH --mem=1000G
#SBATCH -o output/18_NATIM_detect_ripples_%A_%a.out
#SBATCH -e output/18_NATIM_detect_ripples_%A_%a.err
#SBATCH --nodes=1

# NATIM pipeline, independent group.
# Requires: 06_write_filtered_natim.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python detect_ripples_df_one_arr_NATIM.py $SLURM_ARRAY_TASK_ID
