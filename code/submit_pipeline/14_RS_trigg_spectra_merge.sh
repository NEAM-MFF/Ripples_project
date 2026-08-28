#!/bin/bash
#SBATCH --job-name=RS_spectra_merge
#SBATCH --mem=200G
#SBATCH -o output/14_RS_trigg_spectra_merge_%A_%a.out
#SBATCH -e output/14_RS_trigg_spectra_merge_%A_%a.err
#SBATCH --nodes=1

# Merges the per-array tmp files from 13 into final per-date pickles, then
# deletes the tmp files. No array number.
# Requires: EVERY array job from 13 finished (not just started) - check
# squeue is empty for job 13 before submitting this.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_spectra_merge_RS_new_filter.py
