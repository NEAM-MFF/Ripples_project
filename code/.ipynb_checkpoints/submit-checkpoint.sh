#!/bin/bash
#SBATCH --job-name=SUA_RS_KS4
#SBATCH --array=0-2  # 0-6, 3 monkeys, all dates for pair analysis RS
# #SBATCH --array=0-2  # 3 monkeys for RS SUA prop.
# #SBATCH --array=0-47%12  # 0-47, 3 monkeys * 16 arrays = 48 jobs, max N at a time, for ripple detection
# #SBATCH --array=0-17%6  # 3 monkeys * 6 trigger options = 18 jobs, max 4 at a time, for trigg. stats
# #SBATCH --array=0-5%4  # 3 monkeys * 2 trigger options = 6 jobs, for phase aligned trigg. stats
# #SBATCH --array=0-2
#SBATCH --mem=1000G
#SBATCH -o output/sua_prop_RS_%A_%a.out
#SBATCH -e output/sua_prop_RS_%A_%a.err
#SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_paper_clean/code

## python detect_ripples_df_one_arr_RS.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_df_create_new_RS.py $SLURM_ARRAY_TASK_ID
## python cell_pairs_testing_different_bin_RS.py $SLURM_ARRAY_TASK_ID
python SUA_RS_prop_pkl_create.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_spectra_create_RS.py $SLURM_ARRAY_TASK_ID
## python cell_pairs_testing_different_bin_RS.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_phase_align_df_create_RS.py $SLURM_ARRAY_TASK_ID
## python delta_rb_env_dict_figure1.py
## python shuffle_phases_df.py
## python graph_SUA_preprocess.py
## python hypnogram_df.py