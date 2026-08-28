#!/bin/bash
#SBATCH --job-name=HUMAN_ripples
#SBATCH --mem=500G
#SBATCH -o output/24_HUMAN_detect_ripples_%A_%a.out
#SBATCH -e output/24_HUMAN_detect_ripples_%A_%a.err
#SBATCH --nodes=1

# No array number. Requires: 07_write_filtered_human.sh finished for every
# patient/date - i.e. spikes_filtered/{date}_spikes_filtered.nix must exist
# for all of them.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python detect_ripples_df_one_arr_HUMAN.py
