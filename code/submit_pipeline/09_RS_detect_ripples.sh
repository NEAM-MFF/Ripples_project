#!/bin/bash
#SBATCH --job-name=RS_ripples
#SBATCH --array=0-47%5
#SBATCH --mem=1000G
#SBATCH -o output/09_RS_detect_ripples_%A_%a.out
#SBATCH -e output/09_RS_detect_ripples_%A_%a.err
#SBATCH --nodes=1

# RS pipeline, independent group - 3 monkeys * 16 arrays = 48 jobs, capped at
# 5 concurrent. Requires: 05_write_filtered_rs.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python detect_ripples_df_one_arr_RS.py $SLURM_ARRAY_TASK_ID
