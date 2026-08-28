#!/bin/bash
#SBATCH --job-name=filt_met_rs
#SBATCH --array=0-6
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/01_filter_metrics_rs_%A_%a.out
#SBATCH -e output/01_filter_metrics_rs_%A_%a.err
#SBATCH --nodes=1

# STAGE 0a - RS. First script that has to run at all: computes every QC
# metric (SNR, refractory violations, coincidence, presence ratio, line
# noise...) for every sorted unit in the UNFILTERED spike files. Nothing is
# removed here - thresholds are chosen afterwards (F22 notebook) and applied
# in stage 0c (03_write_filtered_rs.sh).
# --array must equal (number of monkey,date pairs for RS) - 1; the script
# prints the exact count on startup the first time you run it with any
# --array range, so if 0-6 is wrong, check the log and resubmit.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python SUA_filter_metrics_pkl_create.py $SLURM_ARRAY_TASK_ID RS
