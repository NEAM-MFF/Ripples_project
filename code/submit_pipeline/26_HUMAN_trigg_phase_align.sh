#!/bin/bash
#SBATCH --job-name=HUMAN_phase_align
#SBATCH --array=0-1
#SBATCH --mem=500G
#SBATCH -o output/26_HUMAN_trigg_phase_align_%A_%a.out
#SBATCH -e output/26_HUMAN_trigg_phase_align_%A_%a.err
#SBATCH --nodes=1

# --array=0-1, one job per volunteer/subject.
# Requires: BOTH 24_HUMAN_detect_ripples.sh AND 25_HUMAN_prop.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_phase_align_df_create_HUMAN.py $SLURM_ARRAY_TASK_ID
