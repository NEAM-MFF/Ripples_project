#!/bin/bash
#SBATCH --job-name=NATIM_spectra_test
#SBATCH --array=0-1
#SBATCH --mem=700G
#SBATCH -o output/22_NATIM_spectra_testing_%A_%a.out
#SBATCH -e output/22_NATIM_spectra_testing_%A_%a.err
#SBATCH --nodes=1

# Requires: 17_NATIM_prop.sh AND 18_NATIM_detect_ripples.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python spectra_for_testing_NATIM.py $SLURM_ARRAY_TASK_ID
