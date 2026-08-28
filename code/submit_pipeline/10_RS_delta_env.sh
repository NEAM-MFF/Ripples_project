#!/bin/bash
#SBATCH --job-name=RS_delta_env
#SBATCH --mem=1000G
#SBATCH -o output/10_RS_delta_env_%A_%a.out
#SBATCH -e output/10_RS_delta_env_%A_%a.err
#SBATCH --nodes=1

# RS pipeline, independent group - no array, single job.
# Requires: 05_write_filtered_rs.sh finished (reads LFP, not spikes, so does
# NOT need 08/09).

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python delta_rb_env_dict_figure1.py
