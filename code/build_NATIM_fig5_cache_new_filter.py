########## Lightweight cache builder for F5_NATIM_cell_classif / F5_NATIM_phase_pref ##########
#
# TWO caches, for two different problems:
#
# 1. df_merged_area_NATIM_light.pkl (used by both notebooks). Both notebooks
#    used to call load_prop_df(['N','F'],'NATIM',...), which reads and
#    concatenates one pickle per (monkey,date) - ~45 recordings for NATIM vs
#    RS's 7 - and returns every column in sua_prop_all_NATIM for every area.
#    Both notebooks then immediately throw most of that away: they filter to
#    a single area (AREA='V12') and only ever touch a handful of columns. This
#    does that load ONCE, filters to that one area, keeps only the columns
#    actually plotted/tested on, and writes the (small) result.
#
# 2. phase_hist_NATIM.pkl (F5_NATIM_phase_pref only). This used to be a full
#    copy of the phases_spikes_* columns - a raw per-spike phase float for
#    every channel x array x date x monkey, concatenated across ~45
#    recordings - which is what actually produced a 30 GB file, since the
#    notebook only ever collapses that into a smoothed histogram per class
#    anyway. This version histograms per source file as it goes and keeps a
#    running per-class/per-variant count array instead, so the raw phase
#    values are never retained anywhere, on disk or in memory. See the
#    "Phase histograms" section below for the exact accounting.
#
# Run this once (and again whenever the underlying pkls change - e.g. after
# re-running SUA_NATIM_prop_pkl_create_new_filter.py or
# ripple_trigg_phase_align_df_create_NATIM_new_filter.py, or if you need a
# different AREA):
#
#   python build_NATIM_fig5_cache_new_filter.py
#
# No SLURM array needed - single IO + filtering/histogramming job.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import os


def save_pickle_atomic(obj, path):
    """
    Write obj to path without ever leaving a truncated/corrupt file behind.

    A plain to_pickle(path) writes directly to the final filename - if the
    job is killed, OOM'd, or times out partway through (the failure mode that
    produced 'EOFError: Ran out of input' when a notebook tried to unpickle
    this cache), the result is a truncated file at the real path that looks
    like it exists but can't be read. Writing to a temp file first and only
    os.replace()-ing it into place once the write is complete means a crash
    mid-write leaves either nothing (first run) or the previous good cache
    (a rebuild) - never a corrupt file the notebook would trip over.
    """
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)  # atomic on the same filesystem


with open("/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml") as f:
    params_analysis = yaml.safe_load(f)

MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'
MONKEY_LIST = ['N', 'F']
DUAL_TH = [2.5, 3.5]
FINAL_CLASSES = params_analysis['final_classes']

# Matches AREA in both notebooks. Rebuild with a different value here (and
# rerun) if either notebook's AREA constant ever changes.
AREA = 'V12'

CACHE_DIR = f'{DF_FOLDER}/fig5_cache'
ensure_dir_exists(CACHE_DIR)

##### 1. df_merged_area - shared, lightweight, area-filtered #####
# Columns kept, and why (union of everything both notebooks touch):
#   monkey                          - per-animal split (F5_NATIM_cell_classif bar plot)
#   final_class                     - x-axis / grouping for every plot and every stats test
#   avg_wf_zscored                  - waveform plots (tSNE, per-class average waveforms,
#                                      first-peak-height violin)
#   wf_direction                    - DOWN-only filter in the waveform-width histogram
#   width_wf_class, width_wf        - waveform-width histogram
#   norm_RB_phase_selectivity_spikes- selectivity boxplot, selectivity ratio bars, stats tests
#   FR_high_env_low_env_median_ratio- FR-ratio boxplot, stats tests
#   pref_RB_phase_spikes            - circular histogram of phase preference
#
# 'area_merged' is used only to select AREA below, then dropped - after
# filtering to one area it is constant and carries no information for
# plotting. 'is_RB_phase_selective' is recomputed in the notebook itself from
# norm_RB_phase_selectivity_spikes (so the threshold can be changed without
# rebuilding this cache), so it is not stored here either.
KEEP_COLS = [
    'monkey', 'final_class', 'avg_wf_zscored', 'wf_direction',
    'width_wf_class', 'width_wf',
    'norm_RB_phase_selectivity_spikes', 'FR_high_env_low_env_median_ratio',
    'pref_RB_phase_spikes',
]

print('Loading df_merged (pooled NATIM SUA properties, all dates, all areas)...')
df_merged = load_prop_df(MONKEY_LIST, 'NATIM', params_analysis, DF_FOLDER, exclude_noisy=True)
print(f'  {df_merged.shape[0]} units total, {len(df_merged.columns)} columns before pruning')

df_merged_area = df_merged[df_merged['area_merged'] == AREA]
missing = [c for c in KEEP_COLS if c not in df_merged_area.columns]
if missing:
    raise RuntimeError(f'KEEP_COLS references column(s) not present in the loaded data: {missing}. '
                        f'Check for upstream column renames before trusting this cache.')
df_merged_area = df_merged_area[KEEP_COLS].reset_index(drop=True)
print(f'  {df_merged_area.shape[0]} units in area {AREA}, {len(KEEP_COLS)} columns kept')

merged_path = f'{CACHE_DIR}/df_merged_area_NATIM_light.pkl'
save_pickle_atomic(df_merged_area, merged_path)
print(f'Saved: {merged_path}')
del df_merged  # the full, unfiltered frame is not needed past this point

##### 2. Phase histograms - lightweight, F5_NATIM_phase_pref only #####
# REDONE: the previous version of this section kept the phases_spikes_*
# columns as-is - one Python list of raw per-spike phase FLOATS per
# channel-row, for every channel x array x date x monkey. Concatenated across
# ~45 NATIM recordings that was the ~30 GB file. But the notebook never reads
# a single raw phase value back out - every use (cells 46/47) immediately
# collapses the whole pooled array into one smoothed histogram per class via
# np.histogram(...,bins=400) + gaussian_filter1d. So: do that histogramming
# HERE, per source file, accumulating into a running per-class/per-variant
# count array, and never keep the raw phase values around at all. Final cache
# is 6 classes x 3 variants x 400 bins of floats - a few hundred KB, not GB.
#
# One deliberate, harmless behaviour change: the notebook used to call
# np.histogram(data, bins=400) with an AUTO range (data.min(), data.max()) of
# the already-MIN_VAL/MAX_VAL-filtered pooled array, so the actual bin edges
# depended slightly on the data. Here the range is fixed explicitly to
# (MIN_VAL, MAX_VAL) - required so that per-file histograms computed one at a
# time can be summed into a valid pooled histogram at all. In practice this
# only pins the edges to the exact filter boundary instead of the (very
# slightly narrower) empirical data extent, which is not visually
# distinguishable, and is arguably more correct (fixed, reproducible bins
# rather than data-dependent ones).
MIN_VAL = -16 * np.pi
MAX_VAL = 16 * np.pi
N_BINS = 400
N_PI_MULT = 8  # matches the notebook's N, used only for x-tick labelling there

VARIANTS = ['all', 'selective', 'non_selective']

def phase_cols_for(cl):
    return {'all': f'phases_spikes_{cl}',
            'selective': f'phases_spikes_{cl}_selective',
            'non_selective': f'phases_spikes_{cl}_non_selective'}

bin_edges = np.linspace(MIN_VAL, MAX_VAL, N_BINS + 1)
counts = {cl: {v: np.zeros(N_BINS, dtype=np.int64) for v in VARIANTS} for cl in FINAL_CLASSES}
n_recordings_used = 0

print('\nBuilding phase histograms (ripple-triggered phase, all dates)...')
for monkey in MONKEY_LIST:
    for date in params_analysis['dates'][monkey]['NATIM']:
        file_name = (f'{DF_FOLDER}/ripple_prop_triggered_phase_NATIM/'
                     f'th__{int(DUAL_TH[0]*10)}_{int(DUAL_TH[1]*10)}/'
                     f'first_neg_peak_monkey{monkey}_all_arrays_date_{date}.pkl')
        try:
            with open(file_name, "rb") as f:
                d = pickle.load(f)
        except FileNotFoundError:
            print(f'  MISSING: {file_name}')
            continue
        except (EOFError, pickle.UnpicklingError) as e:
            # a source file itself truncated/corrupt (e.g. an earlier
            # ripple_trigg_phase_align_df_create_NATIM_new_filter.py array job
            # got killed mid-write) - skip it and keep going rather than
            # letting one bad recording take down the whole cache build.
            print(f'  CORRUPT, skipping: {file_name} ({e})')
            continue

        for cl in FINAL_CLASSES:
            for variant, col in phase_cols_for(cl).items():
                if col not in d.columns:
                    continue
                phases = np.asarray(list_merge(d[col].values), dtype=float)
                if phases.size == 0:
                    continue
                phases = phases[(phases > MIN_VAL) & (phases < MAX_VAL)]  # same filter the notebook applied
                c, _ = np.histogram(phases, bins=bin_edges)
                counts[cl][variant] += c
        # d (and every raw phase list in it) goes out of scope here - never
        # accumulated across files, only its histogram contribution is kept
        n_recordings_used += 1

hist_cache = {
    'MIN_VAL': MIN_VAL, 'MAX_VAL': MAX_VAL, 'N_BINS': N_BINS, 'N_PI_MULT': N_PI_MULT,
    'bin_edges': bin_edges,
    'counts': counts,  # counts[cl][variant] -> int64 array, shape (N_BINS,), NOT density-normalised
}
for cl in FINAL_CLASSES:
    print(f'  {cl}: all={counts[cl]["all"].sum()}, selective={counts[cl]["selective"].sum()}, '
          f'non_selective={counts[cl]["non_selective"].sum()} spikes pooled')

hist_path = f'{CACHE_DIR}/phase_hist_NATIM.pkl'
save_pickle_atomic(hist_cache, hist_path)
print(f'Saved: {hist_path} (from {n_recordings_used} recordings)')

print('\nDone. Point the notebook at these cache files instead of re-loading the full raw pkls.')
