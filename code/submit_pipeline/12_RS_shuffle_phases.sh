#!/bin/bash
#SBATCH --job-name=RS_shuffle_phases
#SBATCH --mem=1000G
#SBATCH -o output/12_RS_shuffle_phases_%A_%a.out
#SBATCH -e output/12_RS_shuffle_phases_%A_%a.err
#SBATCH --nodes=1

# Requires: 08_RS_prop.sh finished for all 3 monkeys.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python shuffle_phases_df.py
