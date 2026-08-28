#!/bin/bash
#SBATCH --job-name=RS_trigg_df
#SBATCH --array=0-17%5
#SBATCH --mem=1000G
#SBATCH -o output/15_RS_trigg_df_%A_%a.out
#SBATCH -e output/15_RS_trigg_df_%A_%a.err
#SBATCH --nodes=1

# 3 monkeys * 6 trigger options = 18 jobs, capped at 5 concurrent.
# Requires: BOTH 08_RS_prop.sh AND 09_RS_detect_ripples.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_df_create_new_RS.py $SLURM_ARRAY_TASK_ID
