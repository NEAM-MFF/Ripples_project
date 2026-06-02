### CREATE DATAFRAMES WITH DETECTED RIPPLES ###
from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys

with open("/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

DATA_FOLDER = params_analysis['data_folder']  # folder with all the preprocessed data
DATES = params_analysis['dates']
MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes'  # here the resulting dataframes will be saved

AREAS_MERGED = params_analysis['areas_merged']

DUAL_TH = [2.5,3.5]
TYPE_REC = 'RS'
MIN_DUR = 0.04

LOWPASS = 40  # Hz

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)
    
task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = ['L', 'N', 'F']
arrays = list(range(1, 17))
combinations = [(m, a) for m in monkeys for a in arrays]

monkey, array = combinations[task_id]
print(f"Running ripple detection for Monkey {monkey}, Array {array}.")

for date in DATES[monkey][TYPE_REC]:
    print(date)
    if TYPE_REC=='RS':
        file_path = f'{MAIN_FOLDER}/metadata/EC_EO_indicators/eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl'
        with open(file_path, 'rb') as file:
            eyes_dict = pickle.load(file)

        EC_indic = eyes_dict['EC']
        #EO_indic = eyes_dict['EO']
    else:
        EC_indic = None
    try:
        df_ripples = ripple_prop(monkey, array, date, dual_th=DUAL_TH, f_range= [80,150], 
                            type_rec=TYPE_REC, magnitude_type='amplitude', avg_type='std', 
                            EC_indicator=EC_indic, min_burst_duration=MIN_DUR, 
                                 fs=1000,lowpass_freq=LOWPASS,data_folder=DATA_FOLDER,params=params_analysis)
        folder_name = f'{DF_FOLDER}/{TYPE_REC}_ripples_lowpass_{LOWPASS}Hz_min_dur_{int(MIN_DUR*1000)}/{monkey}/{date}'
        ensure_dir_exists(folder_name)                
        df_ripples.to_csv(f'{folder_name}/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}_{monkey}_{date}_arr{array}_ripples.csv', index=False)

    except:
        print(f'For monkey {monkey}, array {array}, the ripples cannot be detected.')
                
                
            
