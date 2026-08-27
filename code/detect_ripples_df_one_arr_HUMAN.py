### CREATE DATAFRAMES WITH DETECTED RIPPLES, HUMAN ###
#
# NEW FILTER VERSION. Same two changes as the monkey RS/NATIM ripple
# detection scripts:
#   1. params_analysis.yml read from code_new_filter/.
#   2. DF_FOLDER points at dataframes_human_new_filter/ instead of
#      dataframes_human/.
#
# ripple_prop_human() only reads LFP/RB signal blocks, never a spike/unit
# file, so ripple detection is identical regardless of which SUA filtering is
# in use - re-running is not strictly necessary if you already trust the
# dataframes_human/All_ripples_.../ output from before. Writing to
# dataframes_human_new_filter/ here keeps a self-contained copy under the new
# pipeline's folder, same choice made for the monkey scripts.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys

with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

DATA_FOLDER = params_analysis['human_data_folder']  # folder with all the preprocessed data
DATES = params_analysis['dates_human']
MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human_new_filter'  # NEW: here the resulting dataframes will be saved

PATIENT_LIST = ['Patient2','Patient3']

DUAL_TH = [2.5,3.5]
TYPE_REC = 'All'
MIN_DUR = 0.04

LOWPASS = 40  # Hz

for patient in PATIENT_LIST:
    for date in DATES[patient]:
        print(date)
        EC_indic = None
        if True:  #try:
            df_ripples = ripple_prop_human(patient, date, dual_th=DUAL_TH, f_range= [80,150],
                                type_rec=TYPE_REC, magnitude_type='amplitude', avg_type='std',
                                EC_indicator=EC_indic, min_burst_duration=MIN_DUR,
                                     fs=1000,lowpass_freq=LOWPASS,data_folder=DATA_FOLDER,params=params_analysis)
            folder_name = f'{DF_FOLDER}/{TYPE_REC}_ripples_lowpass_{LOWPASS}Hz_min_dur_{int(MIN_DUR*1000)}/{patient}/'
            ensure_dir_exists(folder_name)
            df_ripples.to_csv(f'{folder_name}/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}_{date}_ripples.csv', index=False)
        else: #except:
            print(f'For patient {date}, the ripples cannot be detected.')
