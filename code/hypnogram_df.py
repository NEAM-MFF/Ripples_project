############# HYPNOGRAMS ############
#
# NEW FILTER VERSION. Only path changes - this script is entirely independent
# of SUA filtering and of ripple detection. load_all_arr_list(type_sig='LFP',
# only_good_ch=False) here only loads LFP and cuts it to common times; with
# only_good_ch=False it never touches df_bad_ch_folder, and nothing in this
# script reads a spike or ripples file at all. Same as the other two purely
# signal-based scripts, output is identical regardless of unit filtering -
# re-running is only for a self-contained dataframes_new_filter/ copy.

### IMPORT ###
from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import scipy
import mne
import yasa

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

### PARAMETERS ###
with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DATA_FOLDER = params['data_folder']
DATES = params['dates']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'  # NEW: final destination
METADATA_FOLDER = f'{MAIN_FOLDER}/metadata'

for monkey in ['L','N','F']:
    print(monkey)
    for date in params['dates'][monkey]['RS']:
        print(date)
        LFP_list = load_all_arr_list(monkey,dates=[date],type_sig='LFP',areas='all',only_good_ch=False,
                             params=params,zscore_arr=True,
                             data_folder=DATA_FOLDER,df_ripp_folder='',df_bad_ch_folder='')
        #hyp_list = [] ### list of hypnograms (information from all channels used) for all arrays
        for array in range(16):
            print(array)
            # Load data
            LFP_array = LFP_list[array]

            # Create MNE object
            ch_names = [f"LFP ch. {i}" for i in range(64)]
            info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types="misc")
            raw = mne.io.RawArray(LFP_array, info)

            prob_dfs = []
            for ch in range(64):
                sls = yasa.SleepStaging(raw, eeg_name=f"LFP ch. {ch}", eog_name=None, emg_name=None, metadata=None)
                y_pred = sls.predict()
                prob_dfs.append(sls.predict_proba())

            #### MERGING ALL CHANNELS (FROM ONE ARRAY) HYPNOGRAMS INTO ONE , based on probabilities
            probs_merged = sum(prob_dfs) / len(prob_dfs) ### mean probability based on results of all channels
            ensure_dir_exists(f'{DF_FOLDER}/hypnograms/{monkey}/')
            probs_merged.to_csv(f'{DF_FOLDER}/hypnograms/{monkey}/hypnogram_30s_{monkey}_{date}_array_{array+1}.csv',index=False)
            print(f'Data for {monkey}, {date}, array {array+1} saved.')
