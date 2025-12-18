#!/bin/bash
#SBATCH --job-name=ph_TVSD_trigg
#SBATCH --array=0-3  
#SBATCH --mem=700G
# #SBATCH --exclusive
# #SBATCH --time=10:00:00
#SBATCH -o output/natim_ripple_%A_%a.out
#SBATCH -e output/natim_ripple_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_band_project/code

## python detect_ripples_df_one_arr_NATIM.py $SLURM_ARRAY_TASK_ID
python ripple_trigg_phase_align_df_create_NATIM.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_df_create_new_NATIM.py $SLURM_ARRAY_TASK_ID