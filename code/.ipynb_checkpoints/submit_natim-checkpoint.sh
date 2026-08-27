#!/bin/bash
#SBATCH --job-name=align_NATIM
# #SBATCH --array=0-3
#SBATCH --mem=1000G
#SBATCH -o output/align_NATIM%A_%a.out
#SBATCH -e output/align_NATIM%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter

##### NATURAL IMAGES ANALYSIS - NEW FILTER VERSION #####
##### submit in order ######

## python SUA_NATIM_prop_pkl_create.py $SLURM_ARRAY_TASK_ID  #--array=0-44%5
## python detect_ripples_df_one_arr_NATIM.py $SLURM_ARRAY_TASK_ID  #--array=0-31%5
## python ripple_trigg_df_create_new_NATIM.py $SLURM_ARRAY_TASK_ID  #--array=0-13%5
## python ripple_trigg_phase_align_df_create_NATIM.py $SLURM_ARRAY_TASK_ID  #--array=0-3
python graph_SUA_preprocess.py NATIM
## python spectra_for_testing_NATIM.py $SLURM_ARRAY_TASK_ID  #--array=0-1
