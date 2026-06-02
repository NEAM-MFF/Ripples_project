from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys
import elephant

with open("/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml") as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DATA_FOLDER = params['data_folder'] ### folder with all the preprocessed data
DATES = params['dates']
FINAL_CLASSES = params['final_classes']

DF_FOLDER = f'{MAIN_FOLDER}/dataframes' ### here the resulting dataframes will be saved

DUAL_TH = [2.5,3.5]
LOWPASS = 40

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)
    
task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = ['L', 'N', 'F']
trigg_points = ['start', 'first_pos_peak', 'first_neg_peak', 'signal_max_pos_peak', 'signal_max_neg_peak','stop'] # 'env_max_peak'

combinations = [(m, t) for m in monkeys for t in trigg_points]
monkey, trigg_point = combinations[task_id]

print(f"Running trig. stats. for Monkey {monkey}, all arrays.")

for date in params['dates'][monkey]['RS']:
    print(date)
    try:
        with open(f'{DF_FOLDER}/sua_prop_all/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
    except:
        print('Cannot open the SUA prop. file.')
    prop_list = []
    for array in range(1,17): 
        print(array)
        try:  
            try:
                df_ripp_arr = load_ripples_df([monkey],dual_th=DUAL_TH,date=date,array=array,area=None,condition='RS',
                                            params=params,df_folder=DF_FOLDER,exclude_noisy=True,verbose=False)
                spike_block = load_block(monkey,array,type_rec='RS',type_sig='spikes_KS4',date=date,data_folder=DATA_FOLDER)  # SUA
                RB_block = load_block(monkey,array,type_rec='RS',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                LFP_block = load_block(monkey,array,type_rec='RS',type_sig='LFP',date=date,data_folder=DATA_FOLDER)  # LFP non-norm.
                num_cells = len(spike_block.segments[0].spiketrains)
                
                start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.rescale('ms').magnitude)))
                start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)))
                start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)))
                
                print(f'Start t RB: {start_t_RB_ms}')
                print(f'Start t spikes: {start_t_spikes_ms}')
                print(f'Start t LFP: {start_t_LFP_ms}')
                if start_t_spikes_ms!=start_t_RB_ms:
                    print('Spikes and ripples do not have the same start time.')
                if start_t_spikes_ms!=start_t_LFP_ms:
                    print('Spikes and LFP do not have the same start time.')
                #df_ripp_arr = df_ripp[df_ripp['array']==array]
            except:
                print(f'Cannot read files for date {date}, monkey {monkey}, array {array}.')
            if df_ripp_arr.shape[0]>0:
                rb_sig_arr = sig_block_to_arr(RB_block,'RB_filtered_zsc')
                rb_envelope_arr = sig_block_to_arr(RB_block,'RB_envelope_norm')

                if LOWPASS is not None:
                    rb_envelope_arr = elephant.signal_processing.butter(rb_envelope_arr, highpass_frequency=None, lowpass_frequency=LOWPASS, 
                                                                order=6, filter_function='sosfiltfilt',sampling_frequency=1000, axis=-1)
                    # renormalize, because we filtered again
                    rb_envelope_arr = rb_envelope_arr/rb_envelope_arr.std(axis=1,keepdims=True)
                    
                LFP_arr = sig_block_to_arr(LFP_block,'LFP_zsc')
                spike_arr = spike_block_to_arr(spike_block)

                # creating spike vectors grouped by cell class, spikes on the whole array considered
                cell_classes = np.array(find_classes_cells(spike_block,df_sua))
                spikes_sum_dict_array = {}
                for cl in params['final_classes']:
                    cl_idx = np.where(cell_classes==cl)[0]
                    if len(cl_idx)>0:
                        cell_vector = np.sum(spike_arr[cl_idx,:],axis=0)
                        spikes_sum_dict_array[cl] = cell_vector
                    else:
                        spikes_sum_dict_array[cl] = np.zeros(spike_arr.shape[1])
                
                for ch in range(64):
                    cell_classes = np.array(find_classes_cells(spike_block,df_sua))
                    cell_channels = np.array(find_channels_cells(spike_block,df_sua))
                    mask_no_ch = cell_channels!=ch
                    cell_classes[mask_no_ch] = 'do_not_include_me'
                    spikes_sum_dict_ch = {}
                    for cl in params['final_classes']:
                        cl_idx = np.where(cell_classes==cl)[0]
                        if len(cl_idx)>0:
                            cell_vector = np.sum(spike_arr[cl_idx,:],axis=0)
                            spikes_sum_dict_ch[cl] = cell_vector
                        else:
                            spikes_sum_dict_ch[cl] = np.zeros(spike_arr.shape[1])
                    channel_prop = {}
                    channel_prop['channel_0_63'] = ch
                    channel_prop['array'] = array
                    channel_prop['monkey'] = monkey
                    channel_prop['date'] = date
                    channel_prop['area'] = params['areas'][monkey][array-1]
                    
                    rb_sig_vec = rb_sig_arr[ch,:]
                    rb_env_vec = rb_envelope_arr[ch,:]
                    LFP_vec = LFP_arr[ch,:]
                    df_ch = df_ripp_arr[df_ripp_arr['channel_0_63']==ch]
                    if df_ch.shape[0]>0:
                        ### ALL RIPPLES
                        common_start = params['times_all_arr'][monkey]['RS'][date][0]
                        arr_start = start_t_RB_ms
                        trigger_dict = {
                                    'start': 'start_time_ms',
                                    'first_pos_peak': 'first_pos_peak_time_ms',
                                    'first_neg_peak': 'first_neg_peak_time_ms',
                                    #'env_max_peak': 'envelope_peak_time_ms',
                                    'signal_max_pos_peak': 'positive_peak_time_ms',
                                    'signal_max_neg_peak': 'negative_peak_time_ms',
                                    'stop': 'stop_time_ms'
                                }
                        if trigg_point in trigger_dict:
                            column_name = trigger_dict[trigg_point]
                            only_common_peaks = df_ch[column_name].values + arr_start
                        else:
                            print(f"Invalid trigg_point: {trigg_point}")
                            raise SystemExit("Exiting due to invalid trigger parameter.")
                            
                        only_common_peaks = only_common_peaks[only_common_peaks>common_start]
                        ripp_centers_in_common_time = (only_common_peaks - common_start).astype(np.int64)
                        ripple_center_vec = np.zeros(LFP_vec.shape[0])
                        ripp_centers_in_common_time = ripp_centers_in_common_time[ripp_centers_in_common_time<LFP_vec.shape[0]]  ### recordings sometimes end sooner
                        ripple_center_vec[ripp_centers_in_common_time] = 1

                        df_ch = df_ch[df_ch['eyes_closed'].notna()]  # not using non-classified times of recording
                        df_ch_EC = df_ch[df_ch['eyes_closed']]
                        df_ch_EO = df_ch[np.logical_not(df_ch['eyes_closed'])]

                        ### EYES CLOSED/OPEN RIPPLES, EARLY, MID and LATE PEAK ripples
                        if trigg_point in trigger_dict:
                            column_name = trigger_dict[trigg_point]
                            EC_only_common_peaks = df_ch_EC[column_name].values + arr_start
                            EO_only_common_peaks = df_ch_EO[column_name].values + arr_start
                        else:
                            print(f"Invalid trigg_point: {trigg_point}")
                            raise SystemExit("Exiting due to invalid trigger parameter.")
                            
                        EC_only_common_peaks = EC_only_common_peaks[EC_only_common_peaks>common_start]
                        EC_ripp_centers_in_common_time = (EC_only_common_peaks - common_start).astype(np.int64)
                        EC_ripp_centers_in_common_time = EC_ripp_centers_in_common_time[EC_ripp_centers_in_common_time<LFP_vec.shape[0]] 
                        EC_ripple_center_vec = np.zeros(LFP_vec.shape[0])
                        EC_ripple_center_vec[EC_ripp_centers_in_common_time] = 1

                        EO_only_common_peaks = EO_only_common_peaks[EO_only_common_peaks>common_start]
                        EO_ripp_centers_in_common_time = (EO_only_common_peaks - common_start).astype(np.int64)
                        EO_ripple_center_vec = np.zeros(LFP_vec.shape[0])
                        EO_ripp_centers_in_common_time = EO_ripp_centers_in_common_time[EO_ripp_centers_in_common_time<LFP_vec.shape[0]]
                        EO_ripple_center_vec[EO_ripp_centers_in_common_time] = 1

                        prop_dict = ripple_triggered_prop(ripple_center_vec,EC_ripple_center_vec,EO_ripple_center_vec,
                                                          rb_sig_vec,rb_env_vec,LFP_vec,
                                                          spikes_sum_dict_array=spikes_sum_dict_array,spikes_sum_dict_ch=spikes_sum_dict_ch,
                                                          width_cut=10000,cell_classes=params['final_classes'],channel_prop=channel_prop)
                        prop_list.append(prop_dict)
                    
                        print(f'Prop. for ch. {ch} calculated.')
                    else:
                        print(f'No ripples on ch. {ch} detected.') ### usually indicates the noisy channel that was letf out
        except:
          print(f'For array {array}, the ripple band trigg. properties were not calculated.')
    df_prop = pd.DataFrame(prop_list)
    ensure_dir_exists(f'{DF_FOLDER}/ripple_prop_triggered_10s/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/')
    df_prop.to_pickle(f'{DF_FOLDER}/ripple_prop_triggered_10s/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/{trigg_point}_monkey{monkey}_all_arrays_date_{date}.pkl')
    print(f'Saved DF for date {date}.')


