### Figure 1, preprocessing delta for analysis of its relationship with RB

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import find_peaks

lowpass_freq = 40
MAIN_FOLDER = '/CSNG/studekat/ripple_paper_clean/'
DF_FOLDER = '/CSNG/studekat/ripple_paper_clean/dataframes'

with open("/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml") as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['data_folder'] ### folder with all the preprocessed data
DATES = params['dates']
AREAS = params['areas']

### RIPPLE ENVELOPE ###
# For each monkey, we have three dictionaries: All, EC, EO. In each of them, dictionary with array-wise ripple envelope avg. np array

dates_list = list_merge([DATES['L']['RS'],DATES['N']['RS'],DATES['F']['RS']])
monkeys_list = ['L','L','L','N','N','F','F','F']
all_env_dict = {k:{} for k in dates_list}

for date, monkey in zip(all_env_dict.keys(), monkeys_list):
    print(date, monkey)
    all_env_dict[date] = {k: {} for k in ['All','EC','EO']}
    for array in range(1,17):
        if AREAS[monkey][array-1] in ['V1','V2']:
            try:
                # loading RB block (+ 40 Hz lowpass) and cutting into common times, EC and EO
                file_path = f'{MAIN_FOLDER}/metadata/EC_EO_indicators/eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl'
                with open(file_path, 'rb') as file:
                    eyes_dict = pickle.load(file)
                
                RB_block = load_block(monkey,array,type_rec='RS',type_sig='RB',date=date,data_folder=DATA_FOLDER)
                RB_sig_array = sig_block_to_arr(RB_block,'RB_filtered_zsc')
                RB_env_array = sig_block_to_arr(RB_block,'RB_envelope_norm')
            
                if lowpass_freq is not None:
                    RB_env_array = elephant.signal_processing.butter(RB_env_array, highpass_frequency=None, lowpass_frequency=lowpass_freq, 
                                                                order=6, filter_function='sosfiltfilt',sampling_frequency=1000, axis=-1)
                    # renormalize, because we filtered again
                    RB_env_array = RB_env_array/RB_env_array.std(axis=1,keepdims=True)
                
                n_channels = 64
                start_rec = float(RB_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)
                stop_rec = float(RB_block.segments[0].analogsignals[0].t_stop.rescale('ms').magnitude)
                #print(start_rec)
    
                EC_indicator = eyes_dict['EC'].astype(bool)
                EO_indicator = eyes_dict['EO'].astype(bool)
                min_len = np.int64(np.min([len(EC_indicator),len(np.sum(RB_env_array,axis=0))]))
                print(f'Min. len: {min_len}')
    
                #RB_env_array = cut_abs_times(RB_env_array,np.int64(start_rec),monkey,rec_type='RS',date=date,params=params_analysis)
                all_env_dict[date]['All'][array] = np.sum(RB_env_array,axis=0)[:min_len]
                all_env_dict[date]['EC'][array] = np.sum(RB_env_array,axis=0)[:min_len][EC_indicator[:min_len]]
                all_env_dict[date]['EO'][array] = np.sum(RB_env_array,axis=0)[:min_len][EO_indicator[:min_len]]
                print(array)
            except:
                print(f'The data for array {array} could not be loaded.')
        else:
            print(f'Array {array} is not in V12.')
        
ensure_dir_exists(f'{DF_FOLDER}/delta_rb_env_analysis/')
with open(f'{DF_FOLDER}/delta_rb_env_analysis/sum_rb_envelopes_dict.pkl', "wb") as f:
    pickle.dump(all_env_dict, f)
print("Dictionary saved as all_env_dict.pkl")


### DELTA FILTERED ###
# For each monkey, we have three dictionaries: All, EC, EO. In each of them, dictionary with array-wise delta avg. np array

dates_list = list_merge([DATES['L']['RS'],DATES['N']['RS'],DATES['F']['RS']])
monkeys_list = ['L','L','L','N','N','F','F','F']
all_delta_dict = {k:{} for k in dates_list}

for date, monkey in zip(all_env_dict.keys(), monkeys_list):
    print(date, monkey)
    all_delta_dict[date] = {k: {} for k in ['All','EC','EO']}
    for array in range(1,17):
        if AREAS[monkey][array-1] in ['V1','V2']:
            try: 
                # loading LFP block (+ 40 Hz lowpass) and cutting into common times, EC and EO
                file_path = f'{MAIN_FOLDER}/metadata/EC_EO_indicators/eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl'
                with open(file_path, 'rb') as file:
                    eyes_dict = pickle.load(file)
                
                LFP_block = load_block(monkey,array,type_rec='RS',type_sig='LFP',date=date,data_folder=DATA_FOLDER)
                LFP_sig_array = sig_block_to_arr(LFP_block,'LFP_zsc')
            
                delta_array = elephant.signal_processing.butter(LFP_sig_array, highpass_frequency=0, lowpass_frequency=4, 
                                                            order=6, filter_function='sosfiltfilt',sampling_frequency=1000, axis=-1)

                n_channels = 64
                start_rec = float(LFP_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)
                stop_rec = float(LFP_block.segments[0].analogsignals[0].t_stop.rescale('ms').magnitude)
                #print(start_rec)
    
                EC_indicator = eyes_dict['EC'].astype(bool)
                EO_indicator = eyes_dict['EO'].astype(bool)
                min_len = np.int64(np.min([len(EC_indicator),len(np.sum(delta_array,axis=0))]))
                print(f'Min. len: {min_len}')
    
                #delta_array = cut_abs_times(delta_array,np.int64(start_rec),monkey,rec_type='RS',date=date,params=params_analysis)
                all_delta_dict[date]['All'][array] = np.sum(delta_array,axis=0)[:min_len]
                all_delta_dict[date]['EC'][array] = np.sum(delta_array,axis=0)[:min_len][EC_indicator[:min_len]]
                all_delta_dict[date]['EO'][array] = np.sum(delta_array,axis=0)[:min_len][EO_indicator[:min_len]]
                print(array)
            except:
                print(f'The data for array {array} could not be loaded.')
        else:
            print(f'Array {array} is not in V12.')

ensure_dir_exists(f'{DF_FOLDER}/delta_rb_env_analysis/')
with open(f'{DF_FOLDER}/delta_rb_env_analysis/sum_delta_dict.pkl', "wb") as f:
    pickle.dump(all_delta_dict, f)
print("Dictionary saved as all_delta_dict.pkl")
        