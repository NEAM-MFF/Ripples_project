### Ripple triggered phase aligned SUA properties, NATIM ###

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

DATA_FOLDER = params['natim_data_folder']
DATES = params['dates']
FINAL_CLASSES = params['final_classes']

MAIN_FOLDER = '/CSNG/studekat/ripple_paper_clean'

DF_FOLDER = f'{MAIN_FOLDER}/dataframes' ### here the resulting dataframes will be saved
MONKEY_LIST = ['N','F']

DUAL_TH = [2.5,3.5]

LOWPASS = 40

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)
    
task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = MONKEY_LIST
trigg_points = ['first_neg_peak', 'signal_max_neg_peak']

combinations = [(m, t) for m in monkeys for t in trigg_points]
monkey, trigg_point = combinations[task_id]

print(f"Running trig. stats. for Monkey {monkey}, all arrays.")

for date in params['dates'][monkey]['NATIM']:
    print(date)
    try:
        with open(f'{DF_FOLDER}/sua_prop_all_NATIM/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
    except:
        print('Cannot open the SUA prop. file.')
    prop_list = []
    for array in range(1,17): 
        print(array)
        try:  
            try:
                df_ripp_arr = load_ripples_df([monkey],dual_th=DUAL_TH,date=date,array=array,area=None,condition='NATIM',
                                            params=params,df_folder=DF_FOLDER,exclude_noisy=True,verbose=False)
                print('Ripples loaded.')
                spike_block = load_block(monkey,array,type_rec='NATIM',type_sig='spikes_KS4',date=date,data_folder=DATA_FOLDER)  # SUA
                RB_block = load_block(monkey,array,type_rec='NATIM',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                print('Spikes and RB loaded.')
                num_cells = len(spike_block.segments[0].spiketrains)
                
                start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.rescale('ms').magnitude)))
                start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)))
                
                print(f'Start t RB: {start_t_RB_ms}')
                print(f'Start t spikes: {start_t_spikes_ms}')
                if start_t_spikes_ms!=start_t_RB_ms:
                    print('Spikes and ripples do not have the same start time.')
            except:
                print(f'Cannot read files for date {date}, monkey {monkey}, array {array}.')
            if df_ripp_arr.shape[0]>0:
                rb_phase_arr = sig_block_to_arr(RB_block,'RB_phase')
                spike_arr = spike_block_to_arr(spike_block)  # bined spikes

                cell_channels = np.array(find_channels_cells(spike_block,df_sua))
                cell_selective = np.array(find_selective_cells(spike_block,df_sua))

                # spikes on a given channel
                for ch in range(64):
                    cell_classes = np.array(find_classes_cells(spike_block,df_sua)) # recalculate this for every class, because we edit it later
                    mask_no_ch = cell_channels!=ch  # cells on a different channel
                    cell_classes[mask_no_ch] = 'do_not_include_me'  # Picking only cells of a given class on the given channel
                    spikes_sum_dict_ch = {}
                    spikes_sum_dict_ch_select = {}
                    spikes_sum_dict_ch_non_select = {}
                    for cl in params['final_classes']:  # Creates one vector with spikes of all cells of a given type on the channel summed up
                        cl_idx = np.where(cell_classes==cl)[0]
                        # ALL SPIKES
                        if len(cl_idx)>0:
                            cell_vector = np.sum(spike_arr[cl_idx,:],axis=0)
                            spikes_sum_dict_ch[cl] = cell_vector
                        else:
                            spikes_sum_dict_ch[cl] = np.zeros(spike_arr.shape[1])
                        # SELECTIVE or NON-SELECTIVE CELLS ONLY
                        spikes_sum_dict_ch_select[cl] = np.zeros(spike_arr.shape[1])
                        spikes_sum_dict_ch_non_select[cl] = np.zeros(spike_arr.shape[1])
                        for j in cl_idx:
                            if cell_selective[j]:  # if the cell is selective
                                spikes_sum_dict_ch_select[cl]+=spike_arr[j,:]
                            else:
                                spikes_sum_dict_ch_non_select[cl]+=spike_arr[j,:]
                        
                    channel_prop = {}
                    channel_prop['channel_0_63'] = ch
                    channel_prop['array'] = array
                    channel_prop['monkey'] = monkey
                    channel_prop['date'] = date
                    channel_prop['area'] = params['areas'][monkey][array-1]
                    
                    rb_phase_vec = rb_phase_arr[ch,:]
                    
                    df_ch = df_ripp_arr[df_ripp_arr['channel_0_63']==ch]  # Ripples on a given channel
                    if df_ch.shape[0]>0:
                        ### ALL RIPPLES
                        arr_start = start_t_RB_ms
                        trigger_dict = {
                                    'first_neg_peak': 'first_neg_peak_time_ms',
                                    'signal_max_neg_peak': 'negative_peak_time_ms',
                                }
                        if trigg_point in trigger_dict:
                            column_name = trigger_dict[trigg_point]
                            only_common_peaks = df_ch[column_name].values + arr_start  # Picking triggers in the common time only, to match EC, EO alignment
                        else:
                            print(f"Invalid trigg_point: {trigg_point}")
                            raise SystemExit("Exiting due to invalid trigger parameter.")

                        # IN NATIM WE DO NOT ALIGN
                        #only_common_peaks = only_common_peaks[only_common_peaks>common_start]
                        ripp_centers_in_common_time = (only_common_peaks - arr_start).astype(np.int64)
                        ripple_center_vec = np.zeros(rb_phase_vec.shape[0])
                        #ripp_centers_in_common_time = ripp_centers_in_common_time[ripp_centers_in_common_time<rb_phase_vec.shape[0]]  ### recordings sometimes end sooner
                        ripple_center_vec[ripp_centers_in_common_time] = 1

                        ### EYES CLOSED/OPEN RIPPLES, EARLY, MID and LATE PEAK ripples
                        df_ch = df_ch[df_ch['eyes_closed'].notna()]  # not using non-classified times of recording

                        if trigg_point in trigger_dict:
                            column_name = trigger_dict[trigg_point] 
                        else:
                            print(f"Invalid trigg_point: {trigg_point}")
                            raise SystemExit("Exiting due to invalid trigger parameter.")

                        prop_dict = ripple_triggered_phase_prop_NATIM(ripple_center_vec,rb_phase_vec,
                                                                spikes_sum_dict_ch,spikes_sum_dict_ch_select,spikes_sum_dict_ch_non_select,
                                                          width_cut=5000,cell_classes=params['final_classes'],channel_prop=channel_prop)
                        prop_list.append(prop_dict)
                    
                        print(f'Prop. for ch. {ch} calculated.')
                    else:
                        print(f'No ripples on ch. {ch} detected.') ### usually indicates the noisy channel that was letf out
        except:
          print(f'For array {array}, the ripple band trigg. properties were not calculated.')
    df_prop = pd.DataFrame(prop_list)
    ensure_dir_exists(f'{DF_FOLDER}/ripple_prop_triggered_phase_NATIM/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/')
    df_prop.to_pickle(f'{DF_FOLDER}/ripple_prop_triggered_phase_NATIM/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/{trigg_point}_monkey{monkey}_all_arrays_date_{date}.pkl')
    print(f'Saved DF for date {date}.')


