#!/bin/bash
#SBATCH --job-name=NATIM_phase_align
#SBATCH --array=0-3
#SBATCH --mem=1000G
#SBATCH -o output/20_NATIM_trigg_phase_align_%A_%a.out
#SBATCH -e output/20_NATIM_trigg_phase_align_%A_%a.err
#SBATCH --nodes=1

# Requires: BOTH 17_NATIM_prop.sh AND 18_NATIM_detect_ripples.sh finished.
# build_NATIM_fig5_cache_new_filter.py (step 21) reads THIS script's output,
# not the RS spectra - despite submit_new_filter.sh listing the cache script
# after the RS steps, its actual dependency is this NATIM phase-align step.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python ripple_trigg_phase_align_df_create_NATIM.py $SLURM_ARRAY_TASK_ID
