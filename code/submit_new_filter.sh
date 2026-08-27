#!/bin/bash
#SBATCH --job-name=spectra_test
#SBATCH --array=0-1
#SBATCH --mem=700G
#SBATCH -o output/00spectra_%A_%a.out
#SBATCH -e output/00spectra_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory - NEW: code_new_filter, not code
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter

##### MONKEY RESTING STATE ANALYSIS, NEW FILTERING #####
##### files to be submitted in order #####

## python SUA_RS_prop_pkl_create.py $SLURM_ARRAY_TASK_ID  # --array=0-2, one job per monkey (L, N, F)
## python detect_ripples_df_one_arr_RS.py $SLURM_ARRAY_TASK_ID  # --array=0-47%5, 3 monkeys * 16 arrays = 48 jobs, max N at a time, for ripple detection
## python delta_rb_env_dict_figure1.py # no additional array numbers
## python shuffle_phases_df.py # no additional array numbers
## python hypnogram_df.py  # no additional array numbers
## python ripple_trigg_df_create_new_RS.py $SLURM_ARRAY_TASK_ID  # --array=0-17%5, 3 monkeys * 6 trigger options = 18 jobs, max 4 at a time, for trigg. stats
## python ripple_trigg_spectra_create_RS_new_filter.py $SLURM_ARRAY_TASK_ID  # --array=0-111, N = 3 monkeys * dates * 16 arrays; script prints exact N=112-1 on startup. 
## python ripple_trigg_spectra_merge_RS_new_filter.py  # no array number - merges the per-array tmp files into final per-date pickles, then deletes the tmp files. Run only after ALL array jobs above have finished.
## python build_NATIM_fig5_cache_new_filter.py
## python ripple_trigg_phase_align_df_create_RS.py $SLURM_ARRAY_TASK_ID  #--array=0-5, 3 monkeys * 2 trigger options = 6 jobs, for phase aligned trigg. stats
## python graph_SUA_preprocess.py
python spectra_for_testing_NATIM.py $SLURM_ARRAY_TASK_ID  #--array=0-1
