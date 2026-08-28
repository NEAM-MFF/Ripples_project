#!/bin/bash
#SBATCH --job-name=NATIM_fig5_cache
#SBATCH --mem=200G
#SBATCH -o output/21_build_NATIM_fig5_cache_%A_%a.out
#SBATCH -e output/21_build_NATIM_fig5_cache_%A_%a.err
#SBATCH --nodes=1

# Lightweight cache for F5_NATIM_cell_classif / F5_NATIM_phase_pref
# (df_merged_area_NATIM_light.pkl + phase_hist_NATIM.pkl). No array.
# Requires: 17_NATIM_prop.sh AND 20_NATIM_trigg_phase_align.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python build_NATIM_fig5_cache_new_filter.py
