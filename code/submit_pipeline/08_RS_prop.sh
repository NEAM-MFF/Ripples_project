#!/bin/bash
#SBATCH --job-name=RS_prop
#SBATCH --array=0-2
#SBATCH --mem=1000G
#SBATCH -o output/08_RS_prop_%A_%a.out
#SBATCH -e output/08_RS_prop_%A_%a.err
#SBATCH --nodes=1

# RS pipeline, step 1 of the "independent" group - one job per monkey (L,N,F).
# Requires: 05_write_filtered_rs.sh finished for all arrays.
# Nothing else in the RS pipeline needs to wait for this EXCEPT 11 and 15/16.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_RS_prop_pkl_create.py $SLURM_ARRAY_TASK_ID
