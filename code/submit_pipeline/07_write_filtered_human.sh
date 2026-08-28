#!/bin/bash
#SBATCH --job-name=write_filt_human
#SBATCH --array=0-185%30
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/07_write_filtered_human_%A_%a.out
#SBATCH -e output/07_write_filtered_human_%A_%a.err
#SBATCH --nodes=1

# STAGE 0c - HUMAN. Reads the SAME thresholds.pkl as RS/NATIM (no separate
# human thresholds file). Writes
# {patient}/spontaneous/spikes_filtered/{date}_spikes_filtered.nix.
# Requires: 03 finished, and 04 (thresholds.pkl) written.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_write_filtered_nix_HUMAN.py $SLURM_ARRAY_TASK_ID
