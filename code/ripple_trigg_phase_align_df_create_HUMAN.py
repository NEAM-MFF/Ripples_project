######## Ripple-triggered phase aligned spikes, HUMAN ##########
#
# NEW FILTER VERSION. Depends on SUA in two places, same shape as the monkey
# new-filter trigger scripts:
#   1. spike_block now loaded via type_sig='spikes_filtered' instead of the
#      original's type_sig='spikes' (the pre-refiltering file). No change
#      needed in load_block_human() - see the header of
#      SUA_HUMAN_prop_pkl_create_new_filter.py for why the generic path
#      pattern already resolves correctly for humans.
#   2. find_classes_cells_human(spike_block, df_sua) / find_channels_cells_human(...)
#      match every unit in spike_block against df_sua by cell_name - only
#      safe when spike_block and df_sua are built from the SAME unit set.
#      df_sua here comes from sua_prop_all under dataframes_human_new_filter/
#      (built by SUA_HUMAN_prop_pkl_create_new_filter.py, which also reads
#      spikes_filtered), so the two sets match.
# Also depends on the ripple dataframes (load_ripples_df_human), repointed
# via DF_FOLDER same as the other human new-filter scripts - requires
# detect_ripples_df_one_arr_HUMAN_new_filter.py to have already been run.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import sys
import elephant

with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['human_data_folder'] ### folder with all the preprocessed data
DATES = params['dates_human']

FINAL_CLASSES = params['final_classes']

MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human_new_filter' ### NEW: input (sua_prop_all, ripples) and output

DUAL_TH = [2.5,3.5]
LOWPASS = 40

trigg_points = ['first_neg_peak']
trigg_point = trigg_points[0]

PATIENT_LIST = ['Patient2','Patient3']

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)

task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID

PATIENT = PATIENT_LIST[task_id]

print(f"Running trig. stats. for {PATIENT}.")

for date in DATES[PATIENT]:
    print(date)
    try:
        with open(f'{DF_FOLDER}/sua_prop_all/{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
    except:
        print('Cannot open the SUA prop. file.')
    prop_list = []
    try:
        try:
            df_ripp_arr = load_ripples_df_human(PATIENT,dual_th=DUAL_TH,dates=[date],condition='All',
                                        params=params,df_folder=DF_FOLDER,exclude_noisy=True,verbose=False)
            spike_block = load_block_human(PATIENT,type_rec='All',type_sig='spikes_filtered',date=date,data_folder=DATA_FOLDER)  # SUA, current filtering
            if spike_block is None:
                raise FileNotFoundError('load_block_human returned None (see printed path above)')
            RB_block = load_block_human(PATIENT,type_rec='All',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
            num_cells = len(spike_block.segments[0].spiketrains)

            start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.rescale('ms').magnitude))) \
                if num_cells > 0 else None
            start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.rescale('ms').magnitude)))

            print(f'Start t RB: {start_t_RB_ms}')
            print(f'Start t spikes: {start_t_spikes_ms}')
            if start_t_spikes_ms is not None and start_t_spikes_ms!=start_t_RB_ms:
                print('Spikes and ripples do not have the same start time.')
        except Exception as e:
            print(f'Cannot read files for {date}. ({e})')
        if df_ripp_arr.shape[0]>0:
            rb_phase_arr = sig_block_to_arr(RB_block,'RB_phase')
            spike_arr = spike_block_to_arr(spike_block)  # bined spikes

            cell_channels = np.array(find_channels_cells_human(spike_block,df_sua))

            # spikes on a given channel
            for ch in range(64):
                cell_classes = np.array(find_classes_cells_human(spike_block,df_sua)) # recalculate this for every class, because we edit it later
                mask_no_ch = cell_channels!=ch  # cells on a different channel
                cell_classes[mask_no_ch] = 'do_not_include_me'  # Picking only cells of a given class on the given channel
                spikes_sum_dict_ch = {}
                for cl in params['final_classes']:  # Creates one vector with spikes of all cells of a given type on the channel summed up
                    cl_idx = np.where(cell_classes==cl)[0]
                    # ALL SPIKES
                    if len(cl_idx)>0:
                        cell_vector = np.sum(spike_arr[cl_idx,:],axis=0)
                        spikes_sum_dict_ch[cl] = cell_vector
                    else:
                        spikes_sum_dict_ch[cl] = np.zeros(spike_arr.shape[1])

                channel_prop = {}
                channel_prop['channel_0_95'] = ch
                channel_prop['patient'] = PATIENT
                channel_prop['date'] = date

                rb_phase_vec = rb_phase_arr[ch,:]

                df_ch = df_ripp_arr[df_ripp_arr['channel_0_95']==ch]  # Ripples on a given channel
                if df_ch.shape[0]>0:
                    ### ALL RIPPLES
                    trigger_dict = {
                                'first_neg_peak': 'first_neg_peak_time_ms',
                                'signal_max_neg_peak': 'negative_peak_time_ms',
                            }
                    if trigg_point in trigger_dict:
                        column_name = trigger_dict[trigg_point]
                        only_common_peaks = df_ch[column_name].values # just one array, all times in common
                    else:
                        print(f"Invalid trigg_point: {trigg_point}")
                        raise SystemExit("Exiting due to invalid trigger parameter.")

                    ripp_centers_in_common_time = only_common_peaks.astype(np.int64)   #(only_common_peaks - common_start).astype(np.int64)
                    ripple_center_vec = np.zeros(rb_phase_vec.shape[0])
                    ripple_center_vec[ripp_centers_in_common_time] = 1

                    if trigg_point in trigger_dict:
                        column_name = trigger_dict[trigg_point]
                    else:
                        print(f"Invalid trigg_point: {trigg_point}")
                        raise SystemExit("Exiting due to invalid trigger parameter.")

                    prop_dict = ripple_triggered_phase_prop_HUMAN(ripple_center_vec,rb_phase_vec,
                                                            spikes_sum_dict_ch, width_cut=5000,cell_classes=params['final_classes'],
                                                            channel_prop=channel_prop)
                    prop_list.append(prop_dict)

                    print(f'Prop. for ch. {ch} calculated.')
                else:
                    print(f'No ripples on ch. {ch} detected.') ### usually indicates the noisy channel that was letf out
    except Exception as e:
        print(f'For {date} the data were not processed. ({e})')
    df_prop = pd.DataFrame(prop_list)
    ensure_dir_exists(f'{DF_FOLDER}/ripple_prop_triggered_phase/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/')
    df_prop.to_pickle(f'{DF_FOLDER}/ripple_prop_triggered_phase/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/{trigg_point}_{date}.pkl')
    print(f'Saved DF for {date}.')
