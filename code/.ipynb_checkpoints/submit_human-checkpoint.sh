#!/bin/bash
#SBATCH --job-name=human_preprocess
#SBATCH --array=0-1
#SBATCH --mem=500G
#SBATCH -o output/0_human_%A_%a.out
#SBATCH -e output/0_human_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter

##### HUMAN ANALYSIS - NEW FILTER VERSION #####
##### files to be submitted in order #####
##### before calling a script change the number of arrays used in parameters
##### run only AFTER the refiltering stage (submit_write_filtered_human.sh)
##### has produced spikes_filtered/{date}_spikes_filtered.nix for every patient/date

## python detect_ripples_df_one_arr_HUMAN.py  # no additional array number
## python SUA_HUMAN_prop_pkl_create.py $SLURM_ARRAY_TASK_ID # --array=0-1, number of volunteers
python ripple_trigg_phase_align_df_create_HUMAN.py $SLURM_ARRAY_TASK_ID # --array=0-1, number of volunteers
