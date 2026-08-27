#!/bin/bash
#SBATCH --job-name=graph
# #SBATCH --array=0-5
#SBATCH --mem=1000G
#SBATCH -o output/RS_%A_%a.out
#SBATCH -e output/RS_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory - NEW: code_new_filter, not code
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter

##### MONKEY RESTING STATE ANALYSIS, NEW FILTERING #####

#### INDEPENDENT
## python delta_rb_env_dict_figure1.py # no additional array numbers
## python hypnogram_df.py  # no additional array numbers
## python SUA_RS_prop_pkl_create.py $SLURM_ARRAY_TASK_ID  # --array=0-2, one job per monkey (L, N, F)
## python detect_ripples_df_one_arr_RS.py $SLURM_ARRAY_TASK_ID  # --array=0-47%5, 3 monkeys * 16 arrays = 48 jobs, max N at a time, for ripple detection

#### RUN AFTER SUA IS RUN TROUGH
## python shuffle_phases_df.py # no additional array numbers
python graph_SUA_preprocess.py  # no additional array numbers

#### RUN AFTER RIPPLES DETECTION IS RUN TROUGH
## python ripple_trigg_spectra_create_RS.py $SLURM_ARRAY_TASK_ID  # --array=0-2

#### RUN AFTER BOTH SUA AND RIPPLES RUN TROUGH
## python ripple_trigg_df_create_new_RS.py $SLURM_ARRAY_TASK_ID  # --array=0-17%5, 3 monkeys * 6 trigger options = 18 jobs, max 4 at a time, for trigg. stats
## python ripple_trigg_phase_align_df_create_RS.py $SLURM_ARRAY_TASK_ID  #--array=0-5, 3 monkeys * 2 trigger options = 6 jobs, for phase aligned trigg. stats

