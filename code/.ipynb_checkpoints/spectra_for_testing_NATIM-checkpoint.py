# Compare Spectra of putative Deep and L4 units
#
# NEW FILTER VERSION. This script never loads a spike/nix file directly - it
# reads the already-built sua_prop_all_NATIM dataframe (for the cell classes
# used to split channels into "blue"/"orange" cliques) and LFP blocks only.
# So it inherits the new filtering purely through which sua_prop_all_NATIM it
# reads (built by SUA_NATIM_prop_pkl_create_new_filter.py). Only change here:
#   1. params_analysis.yml read from code_new_filter/.
#   2. DF_FOLDER -> dataframes_new_filter/, which repoints both the
#      sua_prop_all_NATIM input and the spectra_NATIM/ output.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys

with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

NATIM_DATA_FOLDER = params_analysis['natim_data_folder'] # folder with all the preprocessed data
DATES = params_analysis['dates']
MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter' ### NEW: input (sua_prop_all_NATIM) and output
FINAL_CLASSES = params_analysis['final_classes']

NPERSEG = 1024

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)

task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = ['N', 'F']
monkey = monkeys[task_id]

print(f"Spectrum for Monkey {monkey}.")

blue_psd_list = []
orange_psd_list = []

for date in params_analysis['dates'][monkey]['NATIM']:
    try:
        print(date)
        # SUA csv, given date
        with open(f'{DF_FOLDER}/sua_prop_all_NATIM/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
        # trial csv, given date
        path_trial = f'{NATIM_DATA_FOLDER}/macaque{monkey}_TVSD_{date}/macaque{monkey}_TVSD_{date}_trial_metadata.csv'
        df_trial = pd.read_csv(path_trial)
        for array in range(1,17):
            try:
                if params_analysis['areas'][monkey][array-1] in ['V1','V2','V12']:
                    lfp_bl = load_block(monkey,array,'NATIM','LFP',date,NATIM_DATA_FOLDER)
                    lfp_FIX = cut_out_LFP(lfp_bl,df_trial,buffer=200)
                    ch_dict = aux_units_on_ch(df_sua,array,final_classes=FINAL_CLASSES)
                    clique_dict = aux_dominant_clique_on_ch(ch_dict)
                    blue_keys, orange_keys = aux_split_idx(clique_dict)
                    blue_lfps, orange_lfps, _ = aux_split_lfp(lfp_FIX,blue_keys,orange_keys)
                    blue_psds, f = spectrum_list(blue_lfps,NPERSEG)
                    orange_psds, f = spectrum_list(orange_lfps,NPERSEG)
                    blue_psd_list.append(blue_psds)
                    orange_psd_list.append(orange_psds)
            except:
                print(f'Array {array} not used.')
    except:
        print(f'Date {date} not used.')

ensure_dir_exists(f'{DF_FOLDER}/spectra_NATIM/')

file = f'{DF_FOLDER}/spectra_NATIM/blue_sp_NATIM_monkey{monkey}.pkl'
with open(file, "wb") as f:
    pickle.dump(blue_psd_list, f)

file = f'{DF_FOLDER}/spectra_NATIM/orange_sp_NATIM_monkey{monkey}.pkl'
with open(file, "wb") as f:
    pickle.dump(orange_psd_list, f)
