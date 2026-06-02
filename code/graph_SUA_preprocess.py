from functions_analysis import *
from functions_plotting import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import seaborn as sns
from collections import defaultdict


with open('/CSNG/studekat/ripple_paper_clean/code/params_analysis.yml') as f:
    params_analysis = yaml.safe_load(f)

DATA_FOLDER = params_analysis['data_folder']  # folder with all the preprocessed data
DATES = params_analysis['dates']

DF_FOLDER = '/CSNG/studekat/ripple_paper_clean/dataframes'  # save the resulting dataframes here

MONKEY_LIST = ['L','N','F'] 
FINAL_CLASSES = params_analysis['final_classes']

# creating the paired classes names
all_pairs = list(itertools.combinations(FINAL_CLASSES, 2))
pairs_merged = ["+".join(sorted(t)) for t in all_pairs]
for cl in FINAL_CLASSES:
    pairs_merged.append("+".join((cl,cl)))

all_pairs_list = []
close_pairs_list = []
classes_counter = {cl:0 for cl in FINAL_CLASSES}

for monkey in MONKEY_LIST:
    print(monkey)
    layout_odd = np.array(params_analysis['layout'][f'{monkey}_odd'])
    layout_even = np.array(params_analysis['layout'][f'{monkey}_even'])
    df_merged = load_prop_df([monkey],'RS',params_analysis,DF_FOLDER,exclude_noisy=True)
    print('SUA loaded.')
    for date in DATES[monkey]['RS']:
        print(date)
        for array in range(16):
            if params_analysis['areas'][monkey][array] in ['V1','V2']:
                print(array)
                if (array+1)%2==0:
                    layout = layout_even
                    #print('Even.')
                else:
                    layout = layout_odd
                    #print('Odd.')
                try:
                    spike_block = load_block(monkey,array+1,type_rec='RS',type_sig='spikes_KS4',date=date,data_folder=DATA_FOLDER)  # SUA
                    print('Data loaded.')
                    spike_classes = find_classes_cells(spike_block,df_merged)
                    #print(spike_classes)
                    spike_classes = [str(item.values[0]) if hasattr(item, "values") else str(item) for item in spike_classes]
                    for cl in spike_classes:
                        classes_counter[cl]+=1
                    spike_channels = find_channels_cells(spike_block,df_merged)
                    spike_channels = [int(item.values[0]) if hasattr(item, "values") else int(item) for item in spike_channels]
                    all_pairs, close_pairs = aux_calculate_pairs(spike_classes,spike_channels,layout,pairs_merged)
        
                    all_pairs_list.append(all_pairs)
                    close_pairs_list.append(close_pairs)
                except:
                    print(f'Data not processed for array {array+1}.')  


# summing up per class
all_sums = defaultdict(float)
for d in all_pairs_list:
    for k, v in d.items():
        all_sums[k] += v

# summing up per class
close_sums = defaultdict(float)
for d in close_pairs_list:
    for k, v in d.items():
        close_sums[k] += v

pooled_count = {}
for t in list(itertools.combinations(FINAL_CLASSES, 2)):
    aux_t = sorted(t)
    pooled_count["+".join(aux_t)] = classes_counter[aux_t[0]]*classes_counter[aux_t[1]]  # number of possible pairs, as edges in a bipartite graph, pooled out of all animals

ensure_dir_exists(f'{DF_FOLDER}/dict_pair_occur')
with open(f'{DF_FOLDER}/dict_pair_occur/all_pairs_count', 'wb') as f:
    pickle.dump(all_sums, f)

with open(f'{DF_FOLDER}/dict_pair_occur/close_pairs_count', 'wb') as f:
    pickle.dump(close_sums, f)

with open(f'{DF_FOLDER}/dict_pair_occur/classes_counter', 'wb') as f:
    pickle.dump(classes_counter, f)

with open(f'{DF_FOLDER}/dict_pair_occur/pooled_count', 'wb') as f:
    pickle.dump(pooled_count, f)



