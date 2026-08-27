########## Ripple triggered spectrum, RS ##########
#
# NEW FILTER VERSION + ARRAY-WISE PARALLELISATION.
#
# WHY THIS CHANGED FROM THE OLD VERSION: originally this script was
# parallelised only over monkeys (--array=0-2), and each of those 3 jobs did
# ALL dates x all 16 arrays x 64 channels sequentially - the actual expensive
# step (the wavelet/spectral computation per channel in ripple_spectral_prop)
# never got split up, so wall-clock time was dominated by whichever monkey
# had the most dates/arrays. This version submits one SLURM array task per
# (monkey, date, array) triple instead, so the per-channel spectral work is
# spread across as many nodes as you're willing to run at once.
#
# Each task computes properties for its ONE array only and writes them to a
# small intermediate pickle under ripple_prop_triggered_spectra_tmp/. Run
# ripple_trigg_spectra_merge_RS_new_filter.py afterwards (once all array jobs
# have finished) to concatenate the per-array pickles for each (monkey,date)
# back into the same final file this script used to write directly
# (ripple_prop_triggered_spectra/th__25_35/{TRIGG_POINT}_monkey{monkey}_all_arrays_date_{date}.pkl),
# and to delete the intermediate files once the merge succeeds.
#
# Same dependency notes as before: this only reads ripple dataframes
# (load_ripples_df) and LFP (load_block(type_sig='LFP')) - no SUA/unit
# filtering dependency at all. DF_FOLDER points at dataframes_new_filter/,
# so make sure detect_ripples_df_one_arr_RS_new_filter.py has already been
# run for all monkeys/arrays before this.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch, spectrogram
import pywt
import sys
import os

with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

DATA_FOLDER = params_analysis['data_folder'] ### folder with all the preprocessed data
DATES = params_analysis['dates']

MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter' ### NEW: input (ripples) and output (this script's result)

DUAL_TH = [2.5,3.5]
TRIGG_POINT = 'peak' ## start, peak, stop of the ripple, has to be peak in this case

WIDTH_WIN_MS_WAVELET = 500 # ms before and after the ripple peak
WIDTH_WIN_MS_SPECTRA = 250 #[100,250,500,1000,2000]

TMP_FOLDER = f'{DF_FOLDER}/ripple_prop_triggered_spectra_tmp/th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}'

##### BUILDING THE (monkey, date, array) TASK LIST #####

monkeys = ['L', 'N', 'F']
combinations = []
for m in monkeys:
    for d in params_analysis['dates'][m]['RS']:
        for arr in range(1, 17):
            combinations.append((m, d, arr))

print(f"Total (monkey, date, array) combinations: {len(combinations)} -> use --array=0-{len(combinations)-1}")

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)

task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkey, date, array = combinations[task_id]
print(f"Running Spectral ripp. tri. prop. for Monkey {monkey}, date {date}, array {array}.")

##### CALCULATING RIPPLE TRIGG. PROPERTIES FOR THIS ONE ARRAY #####

prop_list = []
df_ripp = load_ripples_df([monkey],dual_th=DUAL_TH,date=date,area=None,condition='RS',
                                        params=params_analysis,df_folder=DF_FOLDER,exclude_noisy=True,verbose=False)
try:
    try:
        LFP_block = load_block(monkey,array,type_rec='RS',type_sig='LFP',date=date,data_folder=DATA_FOLDER)  # Ripple band
        df_ripp_arr = df_ripp[df_ripp['array']==array]
        start_t_LFP_ms = int(np.floor(np.float64(LFP_block.segments[0].analogsignals[0].t_start.magnitude)*1000))
    except:
        print(f'Cannot read files for date {date}, monkey {monkey}, array {array}.')
    if df_ripp_arr.shape[0]>0:
        LFP_arr = sig_block_to_arr(LFP_block,'LFP_zsc')
        for ch in range(64):
            channel_prop = {}
            channel_prop['channel_0_63'] = ch
            channel_prop['array'] = array
            channel_prop['monkey'] = monkey
            channel_prop['date'] = date
            channel_prop['area'] = params_analysis['areas'][monkey][array-1]

            LFP_vec = LFP_arr[ch,:]
            df_ch = df_ripp_arr[df_ripp_arr['channel_0_63']==ch]
            if df_ch.shape[0]>0:  # this does need to be realigned now in the final data, but does nothing bad if kept here
                common_start = params_analysis['times_all_arr'][monkey]['RS'][date][0]
                arr_start = start_t_LFP_ms
                if TRIGG_POINT=='peak':
                    only_common_peaks = df_ch['positive_peak_time_ms'].values + arr_start  # conversion to absolute time
                elif TRIGG_POINT=='start':
                    only_common_peaks = df_ch['start_time_ms'].values + arr_start  # conversion to absolute time
                elif TRIGG_POINT=='stop':
                    only_common_peaks = df_ch['stop_time_ms'].values + arr_start  # conversion to absolute time
                else:
                    print('Wrong trigger option.')
                    raise SystemExit('Exitting with wrong param. error.')

                only_common_peaks = only_common_peaks[only_common_peaks>common_start]
                ripp_centers_in_common_time = (only_common_peaks - common_start).astype(np.int64)
                ripple_center_vec = np.zeros(LFP_vec.shape[0])
                ripp_centers_in_common_time = ripp_centers_in_common_time[ripp_centers_in_common_time<LFP_vec.shape[0]]  ### recordings sometimes end sooner
                ripple_center_vec[ripp_centers_in_common_time] = 1

                df_ch = df_ch[df_ch['eyes_closed'].notna()]  # not using non-classified times of recording
                df_ch_EC = df_ch[df_ch['eyes_closed']]
                df_ch_EO = df_ch[np.logical_not(df_ch['eyes_closed'])]

                ### EYES CLOSED
                if TRIGG_POINT=='peak':
                    EC_only_common_peaks = df_ch_EC['positive_peak_time_ms'].values + arr_start
                elif TRIGG_POINT=='start':
                    EC_only_common_peaks = df_ch_EC['start_time_ms'].values + arr_start
                elif TRIGG_POINT=='stop':
                    EC_only_common_peaks = df_ch_EC['stop_time_ms'].values + arr_start
                else:
                    print('Wrong trigger option.')
                    raise SystemExit('Exitting with wrong param. error.')

                EC_only_common_peaks = EC_only_common_peaks[EC_only_common_peaks>common_start]
                EC_ripp_centers_in_common_time = (EC_only_common_peaks - common_start).astype(np.int64)
                EC_ripp_centers_in_common_time = EC_ripp_centers_in_common_time[EC_ripp_centers_in_common_time<LFP_vec.shape[0]]
                EC_ripple_center_vec = np.zeros(LFP_vec.shape[0])
                EC_ripple_center_vec[EC_ripp_centers_in_common_time] = 1

                ### EYES OPEN
                if TRIGG_POINT=='peak':
                    EO_only_common_peaks = df_ch_EO['positive_peak_time_ms'].values + arr_start
                elif TRIGG_POINT=='start':
                    EO_only_common_peaks = df_ch_EO['start_time_ms'].values + arr_start
                elif TRIGG_POINT=='stop':
                    EO_only_common_peaks = df_ch_EO['stop_time_ms'].values + arr_start
                else:
                    print('Wrong trigger option.')
                    raise SystemExit('Exitting with wrong param. error.')

                EO_only_common_peaks = EO_only_common_peaks[EO_only_common_peaks>common_start]
                EO_ripp_centers_in_common_time = (EO_only_common_peaks - common_start).astype(np.int64)
                EO_ripple_center_vec = np.zeros(LFP_vec.shape[0])
                EO_ripp_centers_in_common_time = EO_ripp_centers_in_common_time[EO_ripp_centers_in_common_time<LFP_vec.shape[0]]
                EO_ripple_center_vec[EO_ripp_centers_in_common_time] = 1

                prop_dict = ripple_spectral_prop(ripple_center_vec,EC_ripple_center_vec,EO_ripple_center_vec,
                                              LFP_vec,width_cut_spectra=WIDTH_WIN_MS_SPECTRA,width_cut_wavelet=WIDTH_WIN_MS_WAVELET,
                                                 fs=1000,channel_prop=channel_prop)
                prop_list.append(prop_dict)

                #print(f'Prop. for ch. {ch} calculated.')
            else:
                print(f'No ripples on ch. {ch} detected.') ### usually indicates the noisy channel that was letf out
    else:
        print(f'No ripples on array {array} detected.')
except:
    print(f'For array {array}, the ripple band trigg. properties were not calculated.')

##### SAVING THE PER-ARRAY INTERMEDIATE FILE #####
# Written even if prop_list is empty, so the merge step can tell "this array
# was processed and had nothing" apart from "this array's job never ran".

ensure_dir_exists(TMP_FOLDER)
tmp_path = f'{TMP_FOLDER}/{TRIGG_POINT}_monkey{monkey}_date_{date}_array_{array}.pkl'
with open(tmp_path, 'wb') as f:
    pickle.dump(prop_list, f)
print(f'Saved intermediate file: {tmp_path}')
