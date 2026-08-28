#!/bin/bash
#SBATCH --job-name=graph_preprocess
#SBATCH --mem=1000G
#SBATCH -o output/23_graph_SUA_preprocess_%A_%a.out
#SBATCH -e output/23_graph_SUA_preprocess_%A_%a.err
#SBATCH --nodes=1

# Spatial-clustering graph (Figure 5A). Pools RS + NATIM together, so it
# needs BOTH pipelines' unit-property files, not just one - run once per
# recording type argument (submit_natim.sh only shows the NATIM call; run
# the RS one too).
# Requires: 08_RS_prop.sh finished AND 17_NATIM_prop.sh finished.

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code_new_filter
python graph_SUA_preprocess.py RS
python graph_SUA_preprocess.py NATIM
