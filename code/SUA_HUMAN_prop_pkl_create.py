# Preprocessing SUA properties, mainly classification of units, for Resting state

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml') as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['human_data_folder'] ### folder with all the preprocessed data
DATES = params['dates_human']
MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human' ### here the resulting dataframes will be saved

PATIENT = 'Patient4'

print(f"Running Spike prop. for {PATIENT}.")

calculate_SUA_prop = True # computationaly expensive, calculates all phase properties and saves pkl
calculate_other_prop = True  # modifies pkl by adding more properties

WIDTH_INTERVALS = params['width_intervals']
FINAL_CLASSES = params['final_classes']
PEAK_HEIGHT = params['first_peak_height']

SEL_TH = 0.05  # threshold for selectivity of cells, based on the histogram of all selectivities

##### CALCULATING MAIN SUA PROPERTIES DATAFRAMES #####
if calculate_SUA_prop:
    print('Calculating SUA properties, main dataframes.')
    for date in DATES[PATIENT]:
        print(date)
        prop_list = []
        try:
            try:
                spike_block = load_block_human(PATIENT,type_rec='All',type_sig='spikes',date=date,data_folder=DATA_FOLDER)  # SUA
                RB_block = load_block_human(PATIENT,type_rec='All',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                LFP_block = load_block_human(PATIENT,type_rec='All',type_sig='LFP',date=date,data_folder=DATA_FOLDER)  # LFP
                num_cells = len(spike_block.segments[0].spiketrains)
                start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude)*1000))
                start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                print(f'Start t RB: {start_t_RB_ms}')
                print(f'Start t spikes: {start_t_spikes_ms}')
                print(f'Start t LFP: {start_t_LFP_ms}')
                if start_t_spikes_ms!=start_t_RB_ms:
                    print('Spikes and ripples do not have the same start time.')
            except:
                print(f'Cannot read the spike file for date {date}.')
            
            for cell in range(num_cells):
                try:
                    spike_train = spike_block.segments[0].spiketrains[cell]
                    #print(spike_train.annotations.keys())
                    cell_name = spike_train.annotations['nix_name']
                    electrode_ID = spike_train.annotations['new_electrode_ids']
                    channel_ID = spike_train.annotations['channel_ids']
                    
                    ### channel prop - additional info for a channel
                    channel_prop = {}
                    channel_prop['train_order'] = cell # order in the spike train
                    avg_waveform = np.mean(spike_train.waveforms,axis=0)
                    channel_prop['avg_wf'] = avg_waveform
                    channel_prop['nix_name'] = cell_name
                    channel_prop['new_electrode_ids'] = electrode_ID
                    channel_prop['channel_ids'] = channel_ID
    
                    rb_sig_arr = sig_block_to_arr(RB_block,'RB_filtered_zsc')
                    LFP_sig_arr = sig_block_to_arr(LFP_block,'LFP_zsc')
                    rb_phase_arr = sig_block_to_arr(RB_block,'RB_phase')
                    rb_envelope_arr = sig_block_to_arr(RB_block,'RB_envelope_norm')
                    rb_env_phase_arr = sig_block_to_arr(RB_block,'RB_envelope_phase')
    
                    spike_arr = spike_block_to_arr(spike_block)
    
                    ch = aux_electrodeID_to_ch_order_human(PATIENT,date,electrode_ID,DATA_FOLDER,type_rec='All')
                    print(f'ch:{ch}')
                    channel_prop['channel_order'] = ch
    
                    rb_sig = rb_sig_arr[ch,:]
                    LFP_sig = LFP_sig_arr[ch,:]
                    rb_phase = rb_phase_arr[ch,:]
                    rb_envelope = rb_envelope_arr[ch,:]
                    rb_env_phase = rb_env_phase_arr[ch,:]
                    
                    spike_vector = spike_arr[cell,:]
    
                    prop_dict = spike_train_prop_vec(spike_vector,rb_sig,LFP_sig,rb_phase,rb_envelope,rb_env_phase,channel_prop=channel_prop) ### input already binned spikes
                    prop_list.append(prop_dict)
                except:
                    pass
        except:
            print(f'For {date} the SUA properties were not calculated.')
            
        df_prop = pd.DataFrame(prop_list)
        ensure_dir_exists(f'{DF_FOLDER}/sua_prop/')
        df_prop.to_pickle(f'{DF_FOLDER}/sua_prop/{date}.pkl')


if calculate_other_prop:
    print('Calculating SUA properties - additional modification of main dataframes.')
    for date in DATES[PATIENT]:
        try:
            print(date)
            with open(f'{DF_FOLDER}/sua_prop/{date}.pkl', "rb") as file:
                df_sua = pickle.load(file)
            df_added = aux_add_waveform_prop(df_sua)
            df_added = aux_add_zscored_avg_waveform(df_added)
            #df_added = df_added[df_added['channel_order']>-1] ### erasing not working arrays
            df_added = aux_add_width_classes(df_added,width_intervals=WIDTH_INTERVALS)
            df_added = aux_add_up_down_classes(df_added)
            df_added = aux_add_final_classes(df_added,final_classes=FINAL_CLASSES,peak_height_th=PEAK_HEIGHT)
    
            #### saving new dataframes with properties as pickle
            ensure_dir_exists(f'{DF_FOLDER}/sua_prop_all/')
            df_added.to_pickle(f'{DF_FOLDER}/sua_prop_all/{date}.pkl')
            ### the copy warning is there only for the case of empty arrays, no worries about it
        except:
            print(f'The rec. {date} not added to the final processing.')




