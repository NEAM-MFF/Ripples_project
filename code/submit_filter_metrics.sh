#!/bin/bash
#SBATCH --job-name=filt_met
#SBATCH --array=0-6 #44%23
#SBATCH --mem=700G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH -o output/12_filter_metrics_%A_%a.out
#SBATCH -e output/12_filter_metrics_%A_%a.err
#SBATCH --nodes=1

source /home/studekat/virt_env/work/bin/activate
cd /CSNG/studekat/ripple_paper_clean_copy/code

##### FILTERING METRICS FROM THE UNFILTERED SPIKE FILES #####
# Stage 1 of two. Computes every metric for every sorted unit and saves them.
# Nothing is filtered; thresholds are chosen in the summary notebook and
# applied by a separate script.
#
# IMPORTANT: set --array to the pair count minus one for the chosen type. The
# script prints it on startup. NATIM has 45 pairs, RS far fewer.
#
# The slow parts are the cross-unit coincidence, which scales with the total
# spike count per array, and the waveform shape PCA. The unfiltered files hold
# more units than the good-units files, so expect this to take longer than the
# equivalent job on the filtered set.

python SUA_filter_metrics_pkl_create.py $SLURM_ARRAY_TASK_ID RS
## python SUA_filter_metrics_pkl_create.py $SLURM_ARRAY_TASK_ID NATIM
