# Preprocessing SUA properties, mainly classification of units, for human
# recordings - NEW FILTER VERSION.
#
# Mirrors SUA_RS_prop_pkl_create_new_filter.py's approach, applied to human
# recordings, now that SUA_write_filtered_nix_HUMAN.py's refiltered
# spikes_filtered/{date}_spikes_filtered.nix exists to read from.
#
# WHAT CHANGED VS. THE ORIGINAL SCRIPT
#   1. Spike loading swapped from load_block_human(type_sig='spikes') (the
#      pre-refiltering file) to load_block_human(type_sig='spikes_filtered').
#      No change needed in load_block_human() itself: for humans the
#      directory name and filename suffix are both simply 'spikes_filtered'
#      (unlike monkeys, where the directory is spikes_filtered/ but the
#      filename suffix is spikes_KS4_filtered - that mismatch is what forced
#      a special-cased branch there). The existing generic
#      f'{data_folder}/{patient}/spontaneous/{type_sig}/{date}_{type_sig}.nix'
#      pattern already resolves correctly for type_sig='spikes_filtered'.
#   2. The filtered files carry NO per-spike waveforms (same as monkeys), so
#      `avg_waveform = np.mean(spike_train.waveforms, axis=0)` would raise on
#      None. The average waveform is instead read from the 'avg_wf'
#      annotation the filtering already computed, via the annot() helper
#      that converts the NIX 'NaN' string convention back to a real NaN -
#      same fix as the monkey scripts.
#   3. train_order_original (position in the *unfiltered* file) captured
#      alongside train_order (position within *this* filtered file), for
#      traceability - same as the monkey scripts.
#   4. aux_add_selectivity(...) is now actually called. The original script
#      computed SEL_TH but never used it - is_RB_phase_selective was never
#      added to sua_prop_all for humans at all, unlike the monkey scripts.
#      Added here to match.
#   5. SEL_TH now reads params['select_th'] instead of a hardcoded literal,
#      matching the fix also applied to SUA_RS_prop_pkl_create_new_filter.py
#      and SUA_NATIM_prop_pkl_create_new_filter.py - all three now stay in
#      sync with whatever params_analysis.yml says, instead of three
#      independently hardcoded numbers.
#   6. params_analysis.yml read from code_new_filter/, DF_FOLDER points at
#      dataframes_human_new_filter/, under the SAME subfolder names
#      (sua_prop/, sua_prop_all/) the rest of the pipeline expects.
#   7. Everything else - RB/LFP signal loading, OP-independent since humans
#      have no OP maps, waveform/width/final-class classification - is
#      UNCHANGED, since none of that depends on which unit set is classified.

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
with open('/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml') as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['human_data_folder'] ### folder with all the preprocessed data
DATES = params['dates_human']
MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human_new_filter' ### NEW: separate root

PATIENT_LIST = ['Patient2','Patient3']

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)

task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID

PATIENT = PATIENT_LIST[task_id]

calculate_SUA_prop = True # computationaly expensive, calculates all phase properties and saves pkl
calculate_other_prop = True  # modifies pkl by adding more properties

WIDTH_INTERVALS = params['width_intervals']
FINAL_CLASSES = params['final_classes']
PEAK_HEIGHT = params['first_peak_height']

SEL_TH = params['select_th']  # threshold for selectivity of cells - now sourced from params_analysis.yml, same as RS/NATIM


##### HELPERS #####

def annot(st, key, default=np.nan):
    """NIX cannot store NaN, so it is written as the string 'NaN'."""
    v = st.annotations.get(key, default)
    if isinstance(v, str) and v.strip().lower() == 'nan':
        return np.nan
    return v


##### CALCULATING MAIN SUA PROPERTIES DATAFRAMES #####
if calculate_SUA_prop:
    print('Calculating SUA properties, main dataframes (new filter).')
    for date in DATES[PATIENT]:
        print(date)
        prop_list = []
        try:
            try:
                spike_block = load_block_human(PATIENT,type_rec='All',type_sig='spikes_filtered',date=date,data_folder=DATA_FOLDER)  # SUA, current filtering
                if spike_block is None:
                    raise FileNotFoundError('load_block_human returned None (see printed path above)')
                RB_block = load_block_human(PATIENT,type_rec='All',type_sig='RB',date=date,data_folder=DATA_FOLDER)  # Ripple band
                LFP_block = load_block_human(PATIENT,type_rec='All',type_sig='LFP',date=date,data_folder=DATA_FOLDER)  # LFP
                num_cells = len(spike_block.segments[0].spiketrains)
                start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude)*1000)) \
                    if num_cells > 0 else None
                start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
                print(f'Start t RB: {start_t_RB_ms}')
                print(f'Start t spikes: {start_t_spikes_ms}')
                print(f'Start t LFP: {start_t_LFP_ms}')
                if start_t_spikes_ms is not None and start_t_spikes_ms!=start_t_RB_ms:
                    print('Spikes and ripples do not have the same start time.')
            except Exception as e:
                print(f'Cannot read the spike file for date {date}. ({e})')

            for cell in range(num_cells):
                try:
                    spike_train = spike_block.segments[0].spiketrains[cell]
                    #print(spike_train.annotations.keys())
                    cell_name = spike_train.annotations['nix_name']
                    electrode_ID = spike_train.annotations['new_electrode_ids']
                    channel_ID = spike_train.annotations['channel_ids']

                    ### channel prop - additional info for a channel
                    channel_prop = {}
                    channel_prop['train_order'] = cell # order within THIS (filtered) file
                    channel_prop['train_order_original'] = int(annot(spike_train, 'train_order_original', -1))  # NEW: order in the unfiltered file, for traceability
                    channel_prop['patient'] = PATIENT  # NEW: needed for a monkey/date-style join key across patients
                    channel_prop['date'] = date

                    ### NEW: average waveform read from the filtering's own
                    ### annotation, since filtered files carry no per-spike
                    ### waveforms to average here
                    avg_waveform = np.asarray(annot(spike_train, 'avg_wf'), dtype=float)
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
                except Exception as e:
                    print(f'   unit {cell} failed: {e}')
        except Exception as e:
            print(f'For {date} the SUA properties were not calculated. ({e})')

        df_prop = pd.DataFrame(prop_list)
        ensure_dir_exists(f'{DF_FOLDER}/sua_prop/')
        df_prop.to_pickle(f'{DF_FOLDER}/sua_prop/{date}.pkl')


if calculate_other_prop:
    print('Calculating SUA properties - additional modification of main dataframes (new filter).')
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
            df_added = aux_add_selectivity(df_added,sel_th=SEL_TH)  # NEW: was computed but never applied in the original script

            #### saving new dataframes with properties as pickle
            ensure_dir_exists(f'{DF_FOLDER}/sua_prop_all/')
            df_added.to_pickle(f'{DF_FOLDER}/sua_prop_all/{date}.pkl')
            ### the copy warning is there only for the case of empty arrays, no worries about it
        except Exception as e:
            print(f'The rec. {date} not added to the final processing. ({e})')
