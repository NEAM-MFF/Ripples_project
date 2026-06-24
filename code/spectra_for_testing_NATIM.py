# Compare Spectra of putative Deep and L4 units

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys

with open("/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

NATIM_DATA_FOLDER = params_analysis['natim_data_folder'] # folder with all the preprocessed data
DATES = params_analysis['dates']
MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes' ### here the resulting dataframes will be saved
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
        with open(f'/CSNG/studekat/ripple_paper_clean/dataframes/sua_prop_all_NATIM/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
        # trial csv, given date
        path_trial = f'/CSNG/Ephys_data/Macaque_data/TVSD_data/macaque{monkey}_TVSD_{date}/macaque{monkey}_TVSD_{date}_trial_metadata.csv'
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

file = f'/CSNG/studekat/ripple_paper_clean/dataframes/spectra_NATIM/blue_sp_NATIM_monkey{monkey}.pkl'
with open(file, "wb") as f:
    pickle.dump(blue_psd_list, f)

file = f'/CSNG/studekat/ripple_paper_clean/dataframes/spectra_NATIM/orange_sp_NATIM_monkey{monkey}.pkl'
with open(file, "wb") as f:
    pickle.dump(orange_psd_list, f)

                
            
