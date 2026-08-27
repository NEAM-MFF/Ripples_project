#!/bin/bash
#SBATCH --job-name=write_filt
#SBATCH --array=0-6  #44
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/13_write_filtered_%A_%a.out
#SBATCH -e output/13_write_filtered_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code

##### WRITE THE FILTERED NIX FILES #####
# Reads the unfiltered files and the metrics, applies the thresholds saved by
# the threshold notebook, and writes one file per array into a spikes_filtered
# directory beside the existing spikes directory.
#
# IMPORTANT: set --array to the pair count minus one for the chosen type. The
# script prints it on startup. NATIM has 45 pairs, RS far fewer.
#
# Memory is dominated by reading the per-spike waveforms of one array at a
# time, roughly 1.5 GB, plus overhead. The output is about two percent of the
# input size, since only the mean waveform is retained.

python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID RS
## python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID NATIM
