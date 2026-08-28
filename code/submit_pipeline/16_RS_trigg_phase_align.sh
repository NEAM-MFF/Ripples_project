#!/bin/bash
#SBATCH --job-name=RS_phase_align
#SBATCH --array=0-5
#SBATCH --mem=1000G
#SBATCH -o output/16_RS_trigg_phase_align_%A_%a.out
#SBATCH -e output/16_RS_trigg_phase_align_%A_%a.err
#SBATCH --nodes=1

# 3 monkeys * 2 trigger options = 6 jobs.
# Requires: BOTH 08_RS_prop.sh AND 09_RS_detect_ripples.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_phase_align_df_create_RS.py $SLURM_ARRAY_TASK_ID
