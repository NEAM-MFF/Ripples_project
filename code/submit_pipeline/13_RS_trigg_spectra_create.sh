#!/bin/bash
#SBATCH --job-name=RS_spectra_create
#SBATCH --array=0-111
#SBATCH --mem=700G
#SBATCH -o output/13_RS_trigg_spectra_create_%A_%a.out
#SBATCH -e output/13_RS_trigg_spectra_create_%A_%a.err
#SBATCH --nodes=1

# Array-wise (per monkey, date, array) ripple-triggered spectra. Replaces the
# old single-job ripple_trigg_spectra_create_RS.py (that version, and
# submit_RS.sh which still calls it, are superseded - use this one).
# --array=0-N-1, N = 3 monkeys * dates * 16 arrays; the script prints the
# exact N on startup, confirm 0-111 still matches.
# Requires: 09_RS_detect_ripples.sh finished for all arrays.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_spectra_create_RS_new_filter.py $SLURM_ARRAY_TASK_ID
