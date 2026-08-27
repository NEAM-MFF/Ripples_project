# Write filtered spike files, one per array, keeping only the units that pass
# the chosen criteria.
#
#   python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID RS
#   python SUA_write_filtered_nix.py $SLURM_ARRAY_TASK_ID NATIM
#
# INPUT
#   the unfiltered file          .../spikes/..._Array{N}_spikes_all_units.nix
#   the metrics for that date    dataframes/filter_metrics_{TYPE}/monkey{M}_..._{D}.pkl
#   the thresholds               dataframes/filter_metrics_chosen/thresholds.pkl
#
# OUTPUT
#   .../spikes_filtered/..._Array{N}_spikes_filtered.nix
#
# The output directory sits alongside the existing spikes directory, so the
# recording folder layout is unchanged and the usual loaders reach it by
# swapping one path element.
#
# WHAT IS AND IS NOT CARRIED OVER
# Spike times are copied unchanged, and the surviving units keep their relative
# order from the unfiltered file, with the original index recorded as
# train_order_original so a unit can be traced back.
#
# Every annotation from the unfiltered file is inherited. The metrics computed
# by SUA_filter_metrics_pkl_create.py are added, along with one boolean per
# criterion and the threshold that produced it, so a file records the rule that
# made it.
#
# PER-SPIKE WAVEFORMS ARE NOT WRITTEN. Instead each unit carries avg_wf, the
# mean waveform across its spikes, and avg_wf_zscored, which is that single
# averaged trace z-scored: the mean is taken over spikes first, and the
# resulting waveform is then standardised as a whole. Individual spikes are
# never z-scored, and no per-sample normalisation across spikes is applied.
# This is a large saving: the per-spike waveforms are roughly 1.5 GB per array,
# while two vectors of 90 samples per unit are negligible.
#
# NaN AND NIX
# The NIX format cannot store a numeric NaN. The sorting pipeline handles this
# by writing the string 'NaN' instead, and the same convention is followed here
# so that the files are consistent with the rest of the dataset. A helper for
# reading them back is given at the bottom of this file; note that a direct
# numeric comparison against an unconverted annotation raises a TypeError
# rather than evaluating false.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import quantities as pq
import os
import sys
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean_copy/code/params_analysis.yml') as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes'

if len(sys.argv) < 3:
    print("Usage: SUA_write_filtered_nix.py <SLURM_ARRAY_TASK_ID> <RS|NATIM>")
    sys.exit(1)

task_id = int(sys.argv[1])
TYPE_REC = sys.argv[2].upper()

if TYPE_REC == 'RS':
    MONKEYS = ['L', 'N', 'F']
    DATA_FOLDER = params['data_folder']
elif TYPE_REC == 'NATIM':
    MONKEYS = ['N', 'F']
    DATA_FOLDER = params['natim_data_folder']
else:
    print(f'Unknown recording type {TYPE_REC}')
    sys.exit(1)

pairs_md = []
for monkey in MONKEYS:
    for d in params['dates'][monkey][TYPE_REC]:
        pairs_md.append((monkey, d))

print(f'{len(pairs_md)} (monkey, date) pairs for {TYPE_REC} '
      f'-> use --array=0-{len(pairs_md)-1}')
if task_id >= len(pairs_md):
    print(f'task_id {task_id} out of range.')
    sys.exit(0)

monkey, date = pairs_md[task_id]
print(f'Writing filtered files: {TYPE_REC}, monkey {monkey}, date {date}.')

##### CONFIGURATION #####
CFG = {
    'nan_fails': True,          # a missing metric value fails that criterion

    # Mutual coincidence pairs, where each unit is the other's most coincident
    # partner, are usually one neuron split in two by the four-channel sorting.
    # A threshold removes both, which loses a real neuron. With this set to
    # True the pair is resolved instead: the unit with more spikes is kept and
    # the other dropped, and both are annotated so the decision is visible.
    # Left False by default so the script reproduces exactly what the threshold
    # notebook reported. Turn it on deliberately.
    'resolve_mutual_pairs': False,
    'mutual_ratio_th':      10.0,

    'overwrite': True,
}

# metrics written to every surviving unit
METRIC_KEYS = [
    'n_spikes', 'FR_computed', 'duration_s',
    'viol_ratio_1p5ms', 'viol_ratio_min', 'viol_ratio_slope', 'n_viol_1p5ms',
    'coinc_ratio_worst', 'coinc_ratio_total', 'coinc_partner_name',
    'coinc_mutual',
    'wf_shape_pc1_modesep', 'wf_shape_pc1_bic', 'wf_width_cv',
    'wf_snr_computed', 'wf_corr_median', 'wf_corr_p05',
    'amp_median', 'amp_cv', 'amp_mode_separation',
    'atten_index', 'presence_ratio_computed', 'rate_cv',
    'isi_frac_below_rp', 'isi_median_s',
]


##### PATHS #####
##### SPIKE FILE PATHS #####
#     spikes_KS4/{stem}_Array{N}_spikes_KS4.nix             earlier filtering
#     spikes_KS4/{stem}_Array{N}_spikes_KS4_unfiltered.nix  every sorted unit
#     spikes_filtered/{stem}_Array{N}_spikes_KS4_filtered.nix   this filtering

def recording_folder(monkey, date, type_rec, data_folder):
    """The recording directory, which differs between the two paradigms."""
    if type_rec == 'NATIM':
        return f'{data_folder}/macaque{monkey}_TVSD_{date}'
    return f'{data_folder}/macaque{monkey}_{type_rec}_{date}'


def file_stem(monkey, date, type_rec):
    """The filename prefix, again paradigm dependent."""
    if type_rec == 'NATIM':
        return f'macaque{monkey}_TVSD_{date}'
    return f'macaque{monkey}_{type_rec}_{date}'


def spike_path(monkey, date, array, type_rec, data_folder, kind='all'):
    """
    Path to one array's spike file.

    kind is 'all' for every sorted unit, 'good' for the earlier filtering, or
    'filtered' for the output of this filtering. Returns None if absent.
    """
    base = recording_folder(monkey, date, type_rec, data_folder)
    stem = file_stem(monkey, date, type_rec)
    sub, suf = {'all':      ('spikes_KS4',      'spikes_KS4_unfiltered'),
                'good':     ('spikes_KS4',      'spikes_KS4'),
                'filtered': ('spikes_filtered', 'spikes_KS4_filtered')}[kind]
    p = f'{base}/{sub}/{stem}_Array{array}_{suf}.nix'
    return p if os.path.isfile(p) else None


def input_path(monkey, date, array, type_rec, data_folder):
    return spike_path(monkey, date, array, type_rec, data_folder, kind='all')


def output_path(monkey, date, array, type_rec, data_folder):
    return (f'{recording_folder(monkey, date, type_rec, data_folder)}/'
            f'spikes_filtered/'
            f'{file_stem(monkey, date, type_rec)}_Array{array}'
            f'_spikes_KS4_filtered.nix')


##### HELPERS #####
def sanitise_for_nix(value):
    """
    NIX cannot store a numeric NaN. The sorting pipeline writes the string
    'NaN' instead, and the same convention is kept here so the files match the
    rest of the dataset. Arrays are returned as float arrays with NaN replaced
    by zero and a companion flag recorded by the caller where it matters.
    """
    if value is None:
        return 'NaN'
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    try:
        v = float(value)
        return 'NaN' if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return str(value)


def zscore_avg_waveform(avg_wf):
    """
    Z-score one averaged waveform.

    The input is already the mean over that unit's spikes. This subtracts the
    mean of that single trace and divides by its standard deviation, so the
    result has zero mean and unit standard deviation. Individual spikes are
    never z-scored, and nothing is normalised per sample across spikes.
    """
    wf = np.asarray(avg_wf, dtype=float)
    if wf.size == 0 or not np.all(np.isfinite(wf)):
        return np.full_like(wf, np.nan)
    s = np.std(wf)
    if s <= 0:
        return np.zeros_like(wf)
    return (wf - np.mean(wf)) / s


def load_thresholds():
    p = f'{DF_FOLDER}/filter_metrics_chosen/thresholds.pkl'
    with open(p, 'rb') as f:
        t = pickle.load(f)
    return t['criteria'], t.get('nan_fails', CFG['nan_fails'])


def load_metrics(monkey, date, type_rec):
    p = (f'{DF_FOLDER}/filter_metrics_{type_rec}/'
         f'monkey{monkey}_all_arrays_date_{date}.pkl')
    with open(p, 'rb') as f:
        return pickle.load(f)


##### DECIDE WHICH UNITS SURVIVE #####
CRITERIA, NAN_FAILS = load_thresholds()
print('criteria in use:')
for col, (direc, th) in CRITERIA.items():
    print(f'   {col:28s} keep {direc:5s} {th}')
print(f'   NaN fails: {NAN_FAILS}')

df_met = load_metrics(monkey, date, TYPE_REC)
print(f'\n{df_met.shape[0]} units in the metrics for this recording')

flags = pd.DataFrame(index=df_met.index)
for col, (direc, th) in CRITERIA.items():
    if col not in df_met.columns:
        print(f'WARNING: {col} absent from the metrics, criterion skipped')
        continue
    v = df_met[col].replace([np.inf, -np.inf], np.nan)
    ok = (v <= th) if direc == 'below' else (v >= th)
    flags[col] = (ok & v.notna()) if NAN_FAILS else (ok | v.isna())

df_met['pass_filter'] = flags.all(axis=1)
for c in flags.columns:
    df_met[f'pass_{c}'] = flags[c]

# optional resolution of mutual coincidence pairs
df_met['mutual_resolved_drop'] = False
if CFG['resolve_mutual_pairs'] and 'coinc_mutual' in df_met.columns:
    n_res = 0
    for array, g in df_met.groupby('array'):
        cand = g[g['coinc_mutual'] &
                 (g['coinc_ratio_worst'] > CFG['mutual_ratio_th'])]
        by_name = g.set_index('cell_name')
        seen = set()
        for idx, r in cand.iterrows():
            a, b = r['cell_name'], r['coinc_partner_name']
            if a in seen or b in seen or b not in by_name.index:
                continue
            seen.update([a, b])
            # keep whichever has more spikes
            n_a, n_b = r['n_spikes'], by_name.loc[b, 'n_spikes']
            drop_name = b if n_a >= n_b else a
            m = (df_met['array'] == array) & (df_met['cell_name'] == drop_name)
            df_met.loc[m, 'mutual_resolved_drop'] = True
            # the kept one is admitted even if the coincidence criterion failed
            keep_name = a if drop_name == b else b
            mk = (df_met['array'] == array) & (df_met['cell_name'] == keep_name)
            other = [c for c in flags.columns if c != 'coinc_ratio_worst']
            df_met.loc[mk, 'pass_filter'] = flags.loc[
                df_met.index[mk], other].all(axis=1).values
            n_res += 1
    print(f'resolved {n_res} mutual coincidence pairs')
    df_met.loc[df_met['mutual_resolved_drop'], 'pass_filter'] = False

print(f'{int(df_met["pass_filter"].sum())} of {df_met.shape[0]} units pass '
      f'({100*df_met["pass_filter"].mean():.1f}%)')

# lookup keyed on the identity of a unit within its array
decision = {}
for _, r in df_met.iterrows():
    decision[(int(r['array']), r['cell_name'])] = r


##### WRITE ONE FILE PER ARRAY #####
out_dir = f'{recording_folder(monkey, date, TYPE_REC, DATA_FOLDER)}/spikes_filtered'
ensure_dir_exists(out_dir)
print('\nreading from: spikes_KS4/..._spikes_KS4_unfiltered.nix')
print(f'output directory: {out_dir}')

summary = []
for array in range(1, 17):
    in_p = input_path(monkey, date, array, TYPE_REC, DATA_FOLDER)
    out_p = output_path(monkey, date, array, TYPE_REC, DATA_FOLDER)
    if in_p is None:
        continue
    if os.path.isfile(out_p) and not CFG['overwrite']:
        print(f'--- array {array}: output exists, skipping')
        continue

    try:
        io_in = neo.NixIO(in_p, 'ro')
        blk_in = io_in.read_block()
        sts_in = blk_in.segments[0].spiketrains

        kept = []
        n_no_decision = 0
        for order, st in enumerate(sts_in):
            name = st.annotations.get('nix_name', None)
            rec = decision.get((array, name), None)
            if rec is None:
                n_no_decision += 1
                continue
            if not bool(rec['pass_filter']):
                continue

            # ---- spike times, unchanged ----
            new_st = neo.SpikeTrain(
                times=st.times.copy(),
                t_start=st.t_start,
                t_stop=st.t_stop,
                units=st.units,
                sampling_rate=getattr(st, 'sampling_rate', None),
                name=getattr(st, 'name', None),
            )

            # ---- inherit every annotation from the unfiltered file ----
            for k, v in st.annotations.items():
                new_st.annotations[k] = sanitise_for_nix(v)

            # ---- average waveform, in place of the per-spike waveforms ----
            wf = st.waveforms
            if wf is not None and getattr(wf, 'ndim', 0) >= 2 and wf.shape[0] >= 1:
                wmag = np.asarray(
                    wf.magnitude if hasattr(wf, 'magnitude') else wf, dtype=float)
                if wmag.ndim == 3:          # (spikes, channels, samples)
                    wmag = wmag[:, 0, :]
                avg = np.nanmean(wmag, axis=0)
                new_st.annotations['avg_wf'] = sanitise_for_nix(avg)
                new_st.annotations['avg_wf_zscored'] = sanitise_for_nix(
                    zscore_avg_waveform(avg))
                new_st.annotations['avg_wf_n_spikes'] = int(wmag.shape[0])
                new_st.annotations['avg_wf_n_samples'] = int(wmag.shape[1])
                del wmag
            else:
                new_st.annotations['avg_wf'] = 'NaN'
                new_st.annotations['avg_wf_zscored'] = 'NaN'
                new_st.annotations['avg_wf_n_spikes'] = 0
                new_st.annotations['avg_wf_n_samples'] = 0
            new_st.waveforms = None

            # ---- the computed metrics ----
            for k in METRIC_KEYS:
                if k in rec.index:
                    new_st.annotations[f'flt_{k}'] = sanitise_for_nix(rec[k])

            # ---- the decision, and the rule that produced it ----
            for col, (direc, th) in CRITERIA.items():
                pk = f'pass_{col}'
                if pk in rec.index:
                    new_st.annotations[f'pass_{col}'] = bool(rec[pk])
                new_st.annotations[f'th_{col}'] = f'{direc} {th}'
            new_st.annotations['passed_filter'] = True
            new_st.annotations['train_order_original'] = int(order)
            new_st.annotations['n_units_original'] = int(len(sts_in))

            kept.append(new_st)

        # ---- assemble and write, preserving the original relative order ----
        date_str = datetime.today().strftime('%Y-%m-%d')
        blk_out = neo.Block(
            name=f'Filtered single units, {file_stem(monkey, date, TYPE_REC)} '
                 f'Array{array}',
            description='Units passing the filtering criteria recorded in the '
                        'block annotations. Spike times unchanged; per-spike '
                        'waveforms replaced by the mean waveform and its '
                        'z-scored form (mean over spikes, then that trace '
                        'z-scored) in the unit annotations.',
            date_of_creation=date_str,
            array=array,
        )
        for k, v in blk_in.annotations.items():
            blk_out.annotations[k] = sanitise_for_nix(v)
        blk_out.annotations['filter_criteria'] = str(
            {c: f'{d} {t}' for c, (d, t) in CRITERIA.items()})
        blk_out.annotations['filter_nan_fails'] = bool(NAN_FAILS)
        blk_out.annotations['filter_resolve_mutual_pairs'] = bool(
            CFG['resolve_mutual_pairs'])
        blk_out.annotations['avg_wf_zscore'] = (
            'mean over spikes, then the resulting trace z-scored as a whole')
        blk_out.annotations['n_units_original'] = int(len(sts_in))
        blk_out.annotations['n_units_kept'] = int(len(kept))
        blk_out.annotations['waveforms_stored'] = 'average only'
        blk_out.annotations['date_filtered'] = date_str

        seg = neo.Segment()
        seg.spiketrains = kept
        blk_out.segments = [seg]

        io_out = neo.NixIO(out_p, mode='ow')
        io_out.write_block(blk_out)
        io_out.close()
        io_in.close()

        print(f'--- array {array}: {len(kept)} of {len(sts_in)} units kept'
              + (f', {n_no_decision} without a metrics entry' if n_no_decision else ''),
              flush=True)
        summary.append({'monkey': monkey, 'date': date, 'array': array,
                        'n_original': len(sts_in), 'n_kept': len(kept),
                        'n_no_decision': n_no_decision,
                        'path': out_p})
        del blk_in, blk_out, kept

    except Exception as e:
        print(f'--- array {array} FAILED: {e}')

##### REPORT #####
if not summary:
    print('\nNo arrays written.')
    sys.exit(0)

df_sum = pd.DataFrame(summary)
tot_o, tot_k = df_sum['n_original'].sum(), df_sum['n_kept'].sum()
print(f'\n{len(df_sum)} arrays written')
print(f'{tot_k} of {tot_o} units kept ({100*tot_k/max(tot_o,1):.1f}%)')
if df_sum['n_no_decision'].sum():
    print(f'WARNING: {int(df_sum["n_no_decision"].sum())} units had no metrics '
          'entry and were dropped. Check that the metrics job covered every array.')
print()
print(df_sum[['array', 'n_original', 'n_kept']].to_string(index=False))

sum_dir = f'{DF_FOLDER}/filter_metrics_chosen/write_summary_{TYPE_REC}'
ensure_dir_exists(sum_dir)
df_sum.to_csv(f'{sum_dir}/monkey{monkey}_date_{date}.csv', index=False)


##### READING THE OUTPUT #####
# The files are read with the usual loader by pointing at the new directory.
# A convenience reader, to be added to functions_analysis:
#
# def load_block_filtered(monkey, array, type_rec, date, data_folder=''):
#     if type_rec == 'NATIM':
#         stem = f'macaque{monkey}_TVSD_{date}'
#         folder = f'{data_folder}/{stem}'
#     else:
#         stem = f'macaque{monkey}_{type_rec}_{date}'
#         folder = f'{data_folder}/{stem}'
#     path = f'{folder}/spikes_filtered/{stem}_Array{array}_spikes_filtered.nix'
#     return neo.NixIO(path, 'ro').read_block()
#
# def annotation_value(st, key):
#     """Convert the string 'NaN' back to a numeric NaN."""
#     v = st.annotations.get(key, None)
#     if isinstance(v, str) and v.strip().lower() == 'nan':
#         return np.nan
#     return v
