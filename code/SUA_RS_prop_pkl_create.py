# Preprocessing SUA properties, mainly classification of units, for Resting
# state - NEW FILTER VERSION.
#
# Adapted from SUA_RS_prop_pkl_create.py to read from the current filtering's
# output (spikes_filtered/..._spikes_KS4_filtered.nix) instead of the earlier
# colleague's "good units" file (spikes_KS4/..._spikes_KS4.nix). Per
# HANDOVER_filtered_data.md section 4, the old sua_prop_all is one of the
# dataframes built on the earlier good-units set and does not cover units the
# current filtering admits - this script produces the new-filter equivalent
# rather than overwriting it, so the two remain independently inspectable.
#
# THIS SCRIPT NOW DEPENDS ON functions_analysis_new_filter.py (two fixes
# there, see that file's diff from the original):
#   - load_block() gained a type_sig='spikes_KS4_filtered' branch, since the
#     filtered files live in spikes_filtered/ but are named
#     ..._spikes_KS4_filtered.nix - directory name and filename suffix don't
#     match, which the old generic type_sig substitution assumed they would.
#   - aux_add_zscored_avg_waveform() no longer assumes avg_wf is a
#     pq.Quantity. It isn't here: NIX cannot store Quantity units in an
#     annotation, so avg_wf comes back as a plain float array (see point 2
#     below), and the old code's `wf.magnitude` would have raised
#     AttributeError on it.
# Deploy that file alongside this one in code_new_filter/ (as
# functions_analysis.py, replacing/shadowing the original within that
# directory) - `from functions_analysis import *` below resolves relative to
# whatever's on the path after os.chdir, so this only works if the fixed
# version is what code_new_filter/functions_analysis.py actually is.
#
# WHAT ACTUALLY CHANGED VS. THE ORIGINAL SCRIPT
#   1. Spike loading swapped from load_block(type_sig='spikes_KS4') (the old
#      good-units file) to load_block(type_sig='spikes_KS4_filtered'), using
#      the new branch above.
#   2. The filtered files carry NO per-spike waveforms (see handover section
#      3), so `avg_waveform = np.mean(spike_train.waveforms, axis=0)` cannot
#      work here - it would raise on None. The average waveform is instead
#      read straight from the 'avg_wf' annotation the filtering already
#      computed, via the annot() helper that converts the NIX 'NaN' string
#      convention back to a real NaN.
#   3. train_order_original (the position in the *unfiltered* file, an
#      annotation the filtered files carry) is captured alongside train_order
#      (position within *this* filtered file) for traceability, since the two
#      differ now that units have been removed.
#   4. Output goes to DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'
#      instead of f'{MAIN_FOLDER}/dataframes', under the SAME subfolder names
#      (sua_prop/, sua_prop_all/) the rest of the pipeline expects. Confirmed
#      against the real load_prop_df() in functions_analysis.py: it takes
#      df_folder as an argument and appends 'sua_prop_all' (for RS) itself,
#      so pointing DF_FOLDER at a different root and keeping the subfolder
#      name works with load_prop_df() completely unmodified.
#   5. Everything not related to which spike file is read - the RB/LFP signal
#      loading, EC/EO indicator creation, OP map lookup, phase-property
#      calculation, waveform/width/final-class classification - is UNCHANGED,
#      since none of that depends on which unit set is being classified.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import os
import sys

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml') as f:
    params = yaml.safe_load(f)

DATA_FOLDER = params['data_folder']  ### folder with all the preprocessed data (raw/filtered spikes, RB, LFP)
DATES = params['dates']
MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'  ### NEW: separate root, see note 4 above

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)

task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = ['L', 'N', 'F']

monkey = monkeys[task_id]
print(f"Running Spike prop. (new filter) for Monkey {monkey}.")

create_EC_EO_indicators = True  # creates dictionaries with EC, EO indicators, and saves them
calculate_SUA_prop = True       # computationaly expensive, calculates all phase properties and saves pkl
calculate_other_prop = True     # modifies pkl by adding more properties

WIDTH_INTERVALS = params['width_intervals']
FINAL_CLASSES = params['final_classes']
PEAK_HEIGHT = params['first_peak_height']
SEL_TH = params['select_th'] 

##### HELPERS #####

def annot(st, key, default=np.nan):
    """NIX cannot store NaN, so it is written as the string 'NaN'."""
    v = st.annotations.get(key, default)
    if isinstance(v, str) and v.strip().lower() == 'nan':
        return np.nan
    return v


##### CREATING INDICATORS #####
# unrelated to unit filtering (per-recording eye state, not per-unit), same
# location as before - no reason to duplicate this per pipeline
if create_EC_EO_indicators:
    print('Calculating EC, EO indicators.')
    for date in DATES[monkey]['RS']:
        df_eyes = pd.read_csv(f'/CSNG/Ephys_data/EO_EC_detection_from_LFP/epochs_macaque{monkey}_RS_{date}.csv')
        ### loading indicator for the whole duration of array 1 rec time, sometimes not even that time is covered, a bit shorter
        EC_indic = create_indicator(df_eyes, start_col='t_start', stop_col='t_stop', state_col='state',
                             positive_state='Closed_eyes', mult_factor=1000)
        EO_indic = create_indicator(df_eyes, start_col='t_start', stop_col='t_stop', state_col='state',
                             positive_state='Open_eyes', mult_factor=1000)

        duration_rec = np.int64(params['times_all_arr'][monkey]['RS'][date][1])
        cut_EC_indic = np.zeros(duration_rec)
        cut_EO_indic = np.zeros(duration_rec)
        cut_EC_indic[:len(EC_indic)] = EC_indic
        cut_EO_indic[:len(EO_indic)] = EO_indic
        ### saving as a dictionary
        eyes_indic_dict = {}
        eyes_indic_dict['EC'] = cut_EC_indic
        eyes_indic_dict['EO'] = cut_EO_indic

        name = f'eyes_indic_monkey_{monkey}_RS_date_{date}_common_times'
        ensure_dir_exists(f'{MAIN_FOLDER}/metadata/EC_EO_indicators/')
        with open(f'{MAIN_FOLDER}/metadata/EC_EO_indicators/{name}.pkl', 'wb') as file:
            pickle.dump(eyes_indic_dict, file)


##### CALCULATING MAIN SUA PROPERTIES DATAFRAMES #####
if calculate_SUA_prop:
    print('Calculating SUA properties, main dataframes (new filter).')
    for date in params['dates'][monkey]['RS']:
        print(date)
        prop_list = []
        for array in range(1, 17):
            print(array)
            try:
                try:
                    spike_block = load_block(monkey, array, type_rec='RS', type_sig='spikes_KS4_filtered', date=date, data_folder=DATA_FOLDER)  # SUA, current filtering
                    if spike_block is None:
                        raise FileNotFoundError('load_block returned None (see printed path above)')
                    RB_block = load_block(monkey, array, type_rec='RS', type_sig='RB', date=date, data_folder=DATA_FOLDER)  # Ripple band
                    LFP_block = load_block(monkey, array, type_rec='RS', type_sig='LFP', date=date, data_folder=DATA_FOLDER)  # LFP
                    num_cells = len(spike_block.segments[0].spiketrains)
                    start_t_spikes_ms = int(np.floor(np.float64(spike_block.segments[0].spiketrains[0].t_start.magnitude) * 1000)) \
                        if num_cells > 0 else None
                    start_t_RB_ms = int(np.floor(np.float64(RB_block.segments[0].analogsignals[0].t_start.magnitude) * 1000))
                    start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.magnitude) * 1000))
                    print(f'Start t RB: {start_t_RB_ms}')
                    print(f'Start t spikes: {start_t_spikes_ms}')
                    print(f'Start t LFP: {start_t_LFP_ms}')
                    if start_t_spikes_ms is not None and start_t_spikes_ms != start_t_RB_ms:
                        print('Spikes and ripples do not have the same start time.')
                except Exception as e:
                    print(f'Cannot read the spike file for date {date}, monkey {monkey}, array {array}. ({e})')
                try:
                    df_OP = pd.read_csv(f'{DATA_FOLDER}/metadata/OP_maps_dataframes/{monkey}/OP_prop_OG_array{array}.csv')
                except Exception:
                    print(f'Cannot read OP maps for date {date}, monkey {monkey}, array {array}.')

                for cell in range(num_cells):
                    spike_train = spike_block.segments[0].spiketrains[cell]
                    cell_name = spike_train.annotations['nix_name']
                    electrode_ID = spike_train.annotations['Electrode_ID']

                    ### channel prop - additional info for a channel, such as OP, bad channel ID, array and area
                    channel_prop = {}
                    channel_prop['cell_name'] = cell_name
                    ### OP
                    try:
                        ch_OP = df_OP[df_OP['Electrode_ID'] == electrode_ID]
                        if ch_OP['selectivity_01'].values[0] > 0.2 and ch_OP['num_f0_high_jump'].values[0] < 3:
                            channel_prop['pref_OP'] = ch_OP['pref_OP'].values[0]
                            channel_prop['selectivity_OP_01'] = ch_OP['selectivity_01'].values[0]
                        else:
                            channel_prop['pref_OP'] = np.nan
                            channel_prop['selectivity_OP_01'] = ch_OP['selectivity_01'].values[0]
                    except Exception:
                        channel_prop['pref_OP'] = np.nan
                        channel_prop['norm_selectivity_OP'] = np.nan
                    ### channel order
                    ch = aux_electrodeID_to_ch_order(monkey, date, electrode_ID, array, data_folder=DATA_FOLDER, type_rec='RS')
                    channel_prop['channel_order'] = ch
                    ### array
                    channel_prop['array'] = array
                    ### area
                    if monkey in ['N', 'F']:
                        name_area = 'Area'
                    else:
                        name_area = 'cortical_area'
                    ch_area = spike_train.annotations[name_area]
                    channel_prop['area'] = ch_area
                    channel_prop['train_order'] = cell  # order within THIS (filtered) file
                    channel_prop['train_order_original'] = int(annot(spike_train, 'train_order_original', -1))  # NEW: order in the unfiltered file, for traceability

                    ### NEW: average waveform read from the filtering's own
                    ### annotation, since filtered files carry no per-spike
                    ### waveforms to average here (see note 2 at top of file)
                    avg_waveform = np.asarray(annot(spike_train, 'avg_wf'), dtype=float)
                    channel_prop['avg_wf'] = avg_waveform

                    rb_sig_arr = sig_block_to_arr(RB_block, 'RB_filtered_zsc')
                    LFP_sig_arr = sig_block_to_arr(LFP_block, 'LFP_zsc')
                    rb_phase_arr = sig_block_to_arr(RB_block, 'RB_phase')
                    rb_envelope_arr = sig_block_to_arr(RB_block, 'RB_envelope_norm')
                    rb_env_phase_arr = sig_block_to_arr(RB_block, 'RB_envelope_phase')

                    spike_arr = spike_block_to_arr(spike_block)

                    rb_sig = rb_sig_arr[ch, :]
                    LFP_sig = LFP_sig_arr[ch, :]
                    rb_phase = rb_phase_arr[ch, :]
                    rb_envelope = rb_envelope_arr[ch, :]
                    rb_env_phase = rb_env_phase_arr[ch, :]

                    spike_vector = spike_arr[cell, :]

                    # EC and EO properties
                    print('Calculating EC, EO properties.')
                    file_path = f'{MAIN_FOLDER}/metadata/EC_EO_indicators/eyes_indic_monkey_{monkey}_RS_date_{date}_common_times.pkl'
                    with open(file_path, 'rb') as file:
                        eyes_dict = pickle.load(file)

                    EC_indic = eyes_dict['EC']
                    EO_indic = eyes_dict['EO']
                    EC_dict = spike_train_prop_vec(spike_vector, rb_sig, LFP_sig, rb_phase, rb_envelope, rb_env_phase, channel_prop=None,
                                                   indicator=EC_indic, indicator_name='EC')
                    EO_dict = spike_train_prop_vec(spike_vector, rb_sig, LFP_sig, rb_phase, rb_envelope, rb_env_phase, channel_prop=None,
                                                   indicator=EO_indic, indicator_name='EO')

                    # adding EC and EO properties to the channel_prop dict.
                    for k in EC_dict.keys():
                        channel_prop[k] = EC_dict[k]
                    for k in EO_dict.keys():
                        channel_prop[k] = EO_dict[k]

                    # whole rec. time properties
                    prop_dict = spike_train_prop_vec(spike_vector, rb_sig, LFP_sig, rb_phase, rb_envelope, rb_env_phase, channel_prop=channel_prop,
                                                    indicator=None, indicator_name=None)  ### input already binned spikes
                    prop_list.append(prop_dict)
            except Exception as e:
                print(f'For array {array}, the SUA properties were not calculated. ({e})')
        df_prop = pd.DataFrame(prop_list)
        ensure_dir_exists(f'{DF_FOLDER}/sua_prop/')
        df_prop.to_pickle(f'{DF_FOLDER}/sua_prop/monkey{monkey}_all_arrays_date_{date}.pkl')

if calculate_other_prop:
    print('Calculating SUA properties - additional modification of main dataframes (new filter).')
    all_RS_dates = params['dates'][monkey]['RS']
    for date in all_RS_dates:
        print(date)
        with open(f'{DF_FOLDER}/sua_prop/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
            df_sua = pickle.load(file)
        df_added = aux_add_waveform_prop(df_sua)
        df_added = aux_add_zscored_avg_waveform(df_added)
        df_added = df_added[df_added['channel_order'] > -1]  ### erasing not working arrays
        df_added = aux_add_width_classes(df_added, width_intervals=WIDTH_INTERVALS)
        df_added = aux_add_up_down_classes(df_added)
        df_added = aux_add_final_classes(df_added, final_classes=FINAL_CLASSES, peak_height_th=PEAK_HEIGHT)
        df_added = aux_add_selectivity(df_added, sel_th=SEL_TH)

        #### saving new dataframes with properties as pickle
        ensure_dir_exists(f'{DF_FOLDER}/sua_prop_all/')
        df_added.to_pickle(f'{DF_FOLDER}/sua_prop_all/monkey{monkey}_all_arrays_date_{date}.pkl')
        ### the copy warning is there only for the case of empty arrays, no worries about it
