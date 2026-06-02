##### SHUFFLE ANALYSIS FOR RS SUA #####

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo

with open("/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml") as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['data_folder'] ### folder with all the preprocessed data
DATES = params['dates']
MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes' ### here the resulting dataframes will be saved
MONKEY_LIST = ['L','N','F']

NUM_REPEAT = 100 ### repetitions per cell

##### CALCULATING SHUFFLE SUA PROPERTIES DATAFRAMES #####
for monkey in MONKEY_LIST:
    print(monkey)
    for date in DATES[monkey]['RS']:
        print(date)
        prop_list = []
        for array in range(1,17): 
            print(array)
            try:
                try:
                    spike_block = load_block(monkey,array,type_rec='RS',type_sig='spikes_KS4',date=date,data_folder=DATA_FOLDER)  # SUA
                    RB_block = load_block(monkey,array,type_rec='RS',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                    num_cells = len(spike_block.segments[0].spiketrains)
                    start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude)*1000))
                    start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                    print(f'Start t RB: {start_t_RB_ms}')
                    print(f'Start t spikes: {start_t_spikes_ms}')
                    if start_t_spikes_ms!=start_t_RB_ms:
                        print('Spikes and ripples do not have the same start time.')
                except:
                    print(f'Cannot read the spike file for date {date}, monkey {monkey}, array {array}.')
                    
                for cell in range(num_cells):
                    spike_train = spike_block.segments[0].spiketrains[cell]
                    cell_name = spike_train.annotations['nix_name']
                    electrode_ID = spike_train.annotations['Electrode_ID']
                    
                    ### channel prop - additional info for a channel, such as OP, bad channel ID, array and area
                    channel_prop = {}
                    channel_prop['cell_name'] = cell_name

                    ### channel order
                    ch = aux_electrodeID_to_ch_order(monkey,date,electrode_ID,array,data_folder=DATA_FOLDER,type_rec='RS')
                    channel_prop['channel_order'] = ch
                    ### array
                    channel_prop['array'] = array
                    ### area
                    if monkey in ['N','F']:
                        name_area = 'Area'
                    else:
                        name_area = 'cortical_area'
                    ch_area = spike_train.annotations[name_area]
                    channel_prop['area'] = ch_area
                    channel_prop['train_order'] = cell # order in the spike train

                    ### the noise properties and cell final classification from the other DF
                    try:
                        with open(f'{DF_FOLDER}/sua_prop_all/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
                            df_sua = pickle.load(file)
                    except:
                        print('Cannot open the SUA prop. file.')
                    df_cell = df_sua[df_sua['cell_name']==cell_name]
                    channel_prop['ch_is_noisy_100Hz'] = df_cell['ch_is_noisy_100Hz'].values[0]
                    channel_prop['ch_is_noisy_120Hz'] = df_cell['ch_is_noisy_120Hz'].values[0]
                    channel_prop['norm_RB_phase_selectivity_spikes'] = df_cell['norm_RB_phase_selectivity_spikes'].values[0]
                    channel_prop['pref_RB_env_phase_spikes'] = df_cell['pref_RB_env_phase_spikes'].values[0]
                    channel_prop['width_wf_class'] = df_cell['width_wf_class'].values[0]
                    if df_cell['area'].values[0] in ['V1','V2']:
                        channel_prop['area_merged'] = 'V12'
                    else:
                        channel_prop['area_merged'] = df_cell['area'].values[0]
                    channel_prop['final_class'] = df_cell['final_class'].values[0]
                    channel_prop['wf_direction'] = df_cell['wf_direction'].values[0]
                    channel_prop['rec_date'] = date
                    channel_prop['array'] = array

                    rb_phase_arr = sig_block_to_arr(RB_block,'RB_phase')
                    rb_env_phase_arr = sig_block_to_arr(RB_block,'RB_envelope_phase')
                    spike_arr = spike_block_to_arr(spike_block)

                    rb_phase_vec = rb_phase_arr[ch,:]
                    rb_env_phase_vec = rb_env_phase_arr[ch,:]
                    
                    spike_vec = spike_arr[cell,:]

                    prop_dict = shuffle_distrib_ph(spike_vec,rb_phase_vec,rb_env_phase_vec,channel_prop=channel_prop,num_repeat=NUM_REPEAT)
                    prop_list.append(prop_dict)
                    
                    print(f'Prop. for cell no. {cell+1}/{num_cells} calculated.')
            except:
                print(f'For array {array}, the shuffle properties were not calculated.')
        df_prop = pd.DataFrame(prop_list)
        ensure_dir_exists(f'{DF_FOLDER}/sua_shuffled_phases/')
        df_prop.to_pickle(f'{DF_FOLDER}/sua_shuffled_phases/monkey{monkey}_all_arrays_date_{date}.pkl')