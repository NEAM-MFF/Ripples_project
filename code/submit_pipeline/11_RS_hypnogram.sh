#!/bin/bash
#SBATCH --job-name=RS_hypnogram
#SBATCH --mem=1000G
#SBATCH -o output/11_RS_hypnogram_%A_%a.out
#SBATCH -e output/11_RS_hypnogram_%A_%a.err
#SBATCH --nodes=1

# RS pipeline, independent group - no array, single job.
# Requires: 05_write_filtered_rs.sh finished (LFP-only, independent of 08/09).

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python hypnogram_df.py
