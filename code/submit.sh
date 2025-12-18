#!/bin/bash
#SBATCH --job-name=LFP_ripples
#SBATCH --array=0-47%2  # 3 monkeys * 16 arrays = 48 jobs, max 2 at a time, for ripple detection
# #SBATCH --array=0-21%4  # 3 monkeys * 7 trigger options = 21 jobs, max 4 at a time, for trigg. stats
# #SBATCH --array=0-5%4  # 3 monkeys * 2 trigger options = 6 jobs, for phase aligned trigg. stats
# #SBATCH --array=0-6%3  
#SBATCH --mem=1000G
#SBATCH -o output/ripple_%A_%a.out
#SBATCH -e output/ripple_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source /home/studekat/virt_env/work/bin/activate

# Change to desired working directory
cd /CSNG/studekat/ripple_band_project/code

python detect_ripples_df_one_arr.py $SLURM_ARRAY_TASK_ID
## python ripple_trigg_df_create_new.py $SLURM_ARRAY_TASK_ID
##python cell_pairs_testing_different_bin.py $SLURM_ARRAY_TASK_ID