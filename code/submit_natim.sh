#!/bin/bash
#SBATCH --job-name=test_NATIM
#SBATCH --array=0-1 
# #SBATCH --mem=700G
# #SBATCH --exclusive
# #SBATCH --time=10:00:00
#SBATCH -o output/000test_NATIM%A_%a.out
#SBATCH -e output/000test_NATIM%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=w17

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_paper_clean/code

## python detect_ripples_df_one_arr_NATIM.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_df_create_new_NATIM.py $SLURM_ARRAY_TASK_ID
## python SUA_NATIM_prop_pkl_create.py $SLURM_ARRAY_TASK_ID  #0-44
## python ripple_trigg_phase_align_df_create_NATIM.py $SLURM_ARRAY_TASK_ID
python spectra_for_testing_NATIM.py $SLURM_ARRAY_TASK_ID