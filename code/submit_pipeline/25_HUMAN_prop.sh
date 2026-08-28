#!/bin/bash
#SBATCH --job-name=HUMAN_prop
#SBATCH --array=0-1
#SBATCH --mem=500G
#SBATCH -o output/25_HUMAN_prop_%A_%a.out
#SBATCH -e output/25_HUMAN_prop_%A_%a.err
#SBATCH --nodes=1

# --array=0-1, one job per volunteer/subject.
# Requires: 07_write_filtered_human.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_HUMAN_prop_pkl_create.py $SLURM_ARRAY_TASK_ID
