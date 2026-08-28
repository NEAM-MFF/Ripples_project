#!/bin/bash
#SBATCH --job-name=filt_met_natim
#SBATCH --array=0-44%23
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/02_filter_metrics_natim_%A_%a.out
#SBATCH -e output/02_filter_metrics_natim_%A_%a.err
#SBATCH --nodes=1

# STAGE 0a - NATIM. Same as 01, for the natural-image recordings (45
# monkey,date pairs -> --array=0-44). Independent of 01, can run in parallel.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_filter_metrics_pkl_create.py $SLURM_ARRAY_TASK_ID NATIM
