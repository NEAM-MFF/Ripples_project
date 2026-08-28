#!/bin/bash
#SBATCH --job-name=filt_met_human
#SBATCH --array=0-185%30
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/03_filter_metrics_human_%A_%a.out
#SBATCH -e output/03_filter_metrics_human_%A_%a.err
#SBATCH --nodes=1

# STAGE 0a - HUMAN. Same idea, one job per (patient, date) pair. Writes
# dataframes_human/filter_metrics_HUMAN/{date}.pkl. Independent of 01/02.
# --array: the script prints the exact (patient,date) pair count on startup -
# confirm 0-185 still matches before submitting.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_filter_metrics_pkl_create_HUMAN.py $SLURM_ARRAY_TASK_ID
