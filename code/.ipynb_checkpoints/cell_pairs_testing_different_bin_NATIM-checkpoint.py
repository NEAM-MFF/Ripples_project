from functions_analysis import *
from functions_plotting import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
from itertools import combinations
import time
import sys

with open("/CSNG/studekat/ripple_band_project/code/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

DATA_FOLDER = params_analysis['natim_data_folder'] 
DATES = params_analysis['dates']
MAIN_FOLDER = params_analysis['main_folder']

DF_FOLDER = f'{MAIN_DOLFER}/dataframes' ### here the resulting dataframes will be saved
MONKEY_LIST = ['N','F'] 
AREAS_MERGED = params_analysis['areas_merged']
FINAL_CLASSES = params_analysis['final_classes']
CLASS_NAMES = params_analysis['classes_names']
AREAS = params_analysis['areas']

WIDTH_LAG = 20
REPEAT = 1000

if len(sys.argv) < 2:
    print("Error: Missing SLURM_ARRAY_TASK_ID argument.")
    sys.exit(1)
    
task_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
monkeys = MONKEY_LIST

combins = [(m, d) for m in monkeys for d in DATES[m]['NATIM']]  # in total 
monkey, date = combins[task_id]

# creating bins centered at each integer number, corresponding to lags
bin_centers = np.arange(-WIDTH_LAG, WIDTH_LAG+1, 1) 
bin_edges = bin_centers - 0.5
bin_edges = np.append(bin_edges, bin_centers[-1] + 0.5)  # add final edge

with open(f'{DF_FOLDER}/sua_prop_all_NATIM/monkey{monkey}_all_arrays_date_{date}.pkl', "rb") as file:
    df_sua = pickle.load(file)

# Spikes
sp_arr_dict = {}
cl_dict = {}
ch_dict = {}
### DATA ORGANISATION
for arr in range(16):
    if AREAS[monkey][arr] in ['V1']:
        try:
            #print(arr)
            aux_bl = load_block(monkey,arr+1,'NATIM','spikes',date,data_folder=DATA_FOLDER)
            aux_arr = spike_block_to_arr(aux_bl,bin_size=0.5*pq.ms)
            sp_arr_dict[arr+1] = aux_arr
            cl_dict[arr+1] = np.array(find_classes_cells(aux_bl,df_sua))
            ch_cells = np.array(find_channels_cells(aux_bl,df_sua))
            ch_dict[arr+1] = ch_cells
        except:
            print(f'Data for {monkey}, {date}, array {arr+1} were not loaded.')

print('Organized data.')

dict_list = []
### Creating dictionary for testing
for ARRAY_IDX in range(1,17):
     if AREAS[monkey][ARRAY_IDX-1] in ['V1']:
         print(ARRAY_IDX)
         try:
            ch_arr = ch_dict[ARRAY_IDX]
            cl_arr = cl_dict[ARRAY_IDX]
            sp_arr = sp_arr_dict[ARRAY_IDX]
             
            # grouping data per array (organising data fro all groups of cells)
            spikes_list, class_list, channel_list, _ = aux_group_data(sp_arr,ch_arr,cl_arr,None)

            for gr_idx in range(len(class_list)): # Itterating over groups of cells on one array
                print(gr_idx)
                cl_names_gr = class_list[gr_idx]
                size_group = len(cl_names_gr)
                spikes_gr = spikes_list[gr_idx]
                channels_gr = channel_list[gr_idx]
                #rb_env_gr = rb_env_list[gr_idx]  # is the same for all the cells in the group 
                for pair in list(combinations(range(size_group), 2)):  # all pairs of cells in the group
                    pair = np.array(pair)
                    spikes_pair = spikes_gr[pair,:]
                    cl_names_pair = cl_names_gr[pair]
                    min_pos, min_neg = find_spike_distances(spikes_pair[0,:], spikes_pair[1,:])
                    all_dist = np.array(list_merge([min_pos,min_neg]))
                    max_lag, val_max_lag = find_peak(min_pos,min_neg,width_lag=WIDTH_LAG)  # positive if the second class follows the first
                    list_max, list_lag_val = shuffle_distrib(spikes_pair[0,:], spikes_pair[1,:],
                                                             max_lag,REPEAT,WIDTH_LAG)
                    aux_dict = {'monkey': monkey,
                                'date': date,
                        'channel': channels_gr,
                        'pair_idx': pair,
                        'class_names': cl_names_pair,
                        'size_whole_group': size_group,
                        'max_lag': max_lag,
                        'val_max_lag': val_max_lag,
                        'data_spike_dist': all_dist,
                        'max_density_shuffle': list_max, # list of REPEAT values
                        'lag_density_shuffle': list_lag_val,
                    }
                    dict_list.append(aux_dict)
         except:
            print('Pass.')

# Saving the list of dictionaries
final_path = f'{DF_FOLDER}/shuffle_cell_pairs_bin_05_NATIM/{monkey}_{date}_N_{REPEAT}_width_{WIDTH_LAG}.pkl'
ensure_dir_exists(f'{DF_FOLDER}/shuffle_cell_pairs_bin_05_NATIM/')
with open(final_path, "wb") as f:
    pickle.dump(dict_list, f)

