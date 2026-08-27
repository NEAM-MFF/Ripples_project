########## Ripple triggered spectrum, RS - MERGE STEP ##########
#
# Run this AFTER all ripple_trigg_spectra_create_RS_new_filter.py array jobs
# have finished. For every (monkey, date), it collects the per-array
# intermediate pickles written under ripple_prop_triggered_spectra_tmp/,
# concatenates them into the same final per-date DataFrame the old
# single-shot script used to produce directly, writes it to
# ripple_prop_triggered_spectra/th__25_35/{TRIGG_POINT}_monkey{monkey}_all_arrays_date_{date}.pkl,
# and then deletes the intermediate per-array files for that date (only
# after the final file has been written successfully).
#
# No SLURM array needed - this is just concatenation/IO, cheap compared to
# the actual spectral computation, so it runs as a single job.

import pandas as pd
import numpy as np
import yaml
import pickle
import os
import glob

from functions_analysis import ensure_dir_exists

with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'

DUAL_TH = [2.5,3.5]
TRIGG_POINT = 'peak'

TMP_FOLDER = f'{DF_FOLDER}/ripple_prop_triggered_spectra_tmp/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}'
FINAL_FOLDER = f'{DF_FOLDER}/ripple_prop_triggered_spectra/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}'
ensure_dir_exists(FINAL_FOLDER)

monkeys = ['L', 'N', 'F']
N_ARRAYS = 16

n_dates_done = 0
n_dates_incomplete = 0

for monkey in monkeys:
    for date in params_analysis['dates'][monkey]['RS']:
        pattern = f'{TMP_FOLDER}/{TRIGG_POINT}_monkey{monkey}_date_{date}_array_*.pkl'
        tmp_files = sorted(glob.glob(pattern))

        if len(tmp_files) == 0:
            print(f'No intermediate files found for monkey {monkey}, date {date} - skipping (has the create script run yet?).')
            continue

        if len(tmp_files) < N_ARRAYS:
            print(f'WARNING: monkey {monkey}, date {date} has only {len(tmp_files)}/{N_ARRAYS} array files. '
                  f'Merging anyway with what is there, but check for failed/unsubmitted array jobs.')
            n_dates_incomplete += 1

        prop_list = []
        for tmp_path in tmp_files:
            with open(tmp_path, 'rb') as f:
                prop_list.extend(pickle.load(f))

        df_prop = pd.DataFrame(prop_list)
        final_path = f'{FINAL_FOLDER}/{TRIGG_POINT}_monkey{monkey}_all_arrays_date_{date}.pkl'
        df_prop.to_pickle(final_path)
        print(f'Saved merged file: {final_path} ({len(prop_list)} channel-rows from {len(tmp_files)} arrays)')

        # only delete intermediates once the merged file has been written successfully
        for tmp_path in tmp_files:
            os.remove(tmp_path)
        n_dates_done += 1

print(f'\nDone. Merged {n_dates_done} (monkey,date) pairs ({n_dates_incomplete} with missing array files).')
