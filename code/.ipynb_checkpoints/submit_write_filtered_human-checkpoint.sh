#!/bin/bash
#SBATCH --job-name=write_filt_human
#SBATCH --array=0-185%30   ### set to (number of patients * number of dates per patient) - 1;
                       ### both scripts print the exact pair count and --array range on startup
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/13_write_filtered_human_%A_%a.out
#SBATCH -e output/13_write_filtered_human_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter

##### REFILTER HUMAN DATA #####
# Same two stages as the monkey RS/NATIM refiltering (submit_write_filtered.sh):
#   1. SUA_filter_metrics_pkl_create_HUMAN.py computes every QC metric for
#      every sorted unit, nothing removed. Writes
#      dataframes_human/filter_metrics_HUMAN/{date}.pkl.
#   2. SUA_write_filtered_nix_HUMAN.py applies the SAME criteria as RS/NATIM
#      (reads dataframes/filter_metrics_chosen/thresholds.pkl - the monkey
#      one, no separate human thresholds file needed) and writes
#      {patient}/spontaneous/spikes_filtered/{date}_spikes_filtered.nix.
#
# Run in order - uncomment one line at a time, matching --array above to the
# (patient, date) pair count each script prints on startup.

## python SUA_filter_metrics_pkl_create_HUMAN.py $SLURM_ARRAY_TASK_ID
python SUA_write_filtered_nix_HUMAN.py $SLURM_ARRAY_TASK_ID
