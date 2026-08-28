#!/bin/bash
#SBATCH --job-name=NATIM_trigg_df
#SBATCH --array=0-13%5
#SBATCH --mem=1000G
#SBATCH -o output/19_NATIM_trigg_df_%A_%a.out
#SBATCH -e output/19_NATIM_trigg_df_%A_%a.err
#SBATCH --nodes=1

# Requires: BOTH 17_NATIM_prop.sh AND 18_NATIM_detect_ripples.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_df_create_new_NATIM.py $SLURM_ARRAY_TASK_ID
