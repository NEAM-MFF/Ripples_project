#!/bin/bash
#SBATCH --job-name=write_filt_rs
#SBATCH --array=0-6
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/05_write_filtered_rs_%A_%a.out
#SBATCH -e output/05_write_filtered_rs_%A_%a.err
#SBATCH --nodes=1

# STAGE 0c - RS. Reads the unfiltered spike files + the thresholds.pkl from
# step 04, writes spikes_filtered/{stem}_Array{N}_spikes_KS4_filtered.nix.
# Everything in code_new_filter/ downstream reads from spikes_filtered/, so
# this (and 06/07) must finish before ANY of the *_new_filter analysis
# scripts below are run.
# Requires: 01 finished, and 04 (thresholds.pkl) written.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID RS
