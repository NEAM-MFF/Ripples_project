# Write filtered spike files for human recordings, one per patient/date,
# keeping only the units that pass the chosen criteria. Stage 2, mirrors
# SUA_write_filtered_nix.py (RS/NATIM), adapted the same way
# SUA_filter_metrics_pkl_create_HUMAN.py adapts stage 1: no arrays, one
# spikes_unfiltered.nix per patient/date, no "KS4" in any filename.
#
#   python SUA_write_filtered_nix_HUMAN.py $SLURM_ARRAY_TASK_ID
#
# INPUT
#   the unfiltered file   {data_folder}/{patient}/spontaneous/spikes/{date}_spikes_unfiltered.nix
#   the metrics           dataframes_human/filter_metrics_HUMAN/{date}.pkl
#   the thresholds        dataframes/filter_metrics_chosen/thresholds.pkl  (the MONKEY one - see below)
#
# OUTPUT
#   {data_folder}/{patient}/spontaneous/spikes_filtered/{date}_spikes_filtered.nix
#
# THRESHOLDS: per instruction, humans use the SAME criteria as RS/NATIM - no
# separate human threshold-selection pass. This reads
# dataframes/filter_metrics_chosen/thresholds.pkl (MONKEY_DF_FOLDER below,
# NOT the human DF_FOLDER), the identical file SUA_write_filtered_nix.py
# reads for RS/NATIM. If that ever needs to diverge - human recordings turn
# out to warrant different cutoffs - point THRESH_FOLDER below at a new
# dataframes_human/filter_metrics_chosen/thresholds.pkl instead, chosen from
# SUA_filter_metrics_pkl_create_HUMAN.py's output the same way F22 chose the
# monkey ones. Not done here.
#
# Everything else - what is and is not carried over, the NaN/NIX 'NaN' string
# convention, avg_wf/avg_wf_zscored instead of per-spike waveforms - is
# unchanged from the RS/NATIM script; see its header for the full rationale.

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
# Same convention as SUA_filter_metrics_pkl_create_HUMAN.py: reads code/ (not
# code_new_filter/), since this stage doesn't depend on
# functions_analysis_new_filter.py's patches and IS the process that produces
# the new filter rather than consuming it.
with open('/CSNG/studekat/ripple_paper_clean_copy/code/params_analysis.yml') as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DATA_FOLDER = params['human_data_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human'
MONKEY_DF_FOLDER = f'{MAIN_FOLDER}/dataframes'  # NEW: thresholds are shared with RS/NATIM, read from here
DATES = params['dates_human']

PATIENT_LIST = ['Patient2', 'Patient3']

if len(sys.argv) < 2:
    print("Usage: SUA_write_filtered_nix_HUMAN.py <SLURM_ARRAY_TASK_ID>")
    sys.exit(1)

task_id = int(sys.argv[1])

pairs_pd = []
for patient in PATIENT_LIST:
    for d in DATES[patient]:
        pairs_pd.append((patient, d))

print(f'{len(pairs_pd)} (patient, date) pairs -> use --array=0-{len(pairs_pd)-1}')
if task_id >= len(pairs_pd):
    print(f'task_id {task_id} out of range.')
    sys.exit(0)

patient, date = pairs_pd[task_id]
print(f'Writing filtered file: HUMAN, patient {patient}, date {date}.')

##### CONFIGURATION ##### (identical to the RS/NATIM script)
CFG = {
    'nan_fails': True,

    'resolve_mutual_pairs': False,
    'mutual_ratio_th':      10.0,

    'overwrite': True,
}

# metrics written to every surviving unit - same list as the RS/NATIM script
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
def spike_path_human(patient, date, data_folder, kind='all'):
    """
    Path to the human spike file. kind is 'all' for every sorted unit
    (spikes_unfiltered.nix) or 'filtered' for the output of this filtering
    (spikes_filtered.nix). Returns None if absent (input_path only).
    """
    base = f'{data_folder}/{patient}/spontaneous'
    suf = {'all': 'spikes_unfiltered', 'good': 'spikes', 'filtered': 'spikes_filtered'}[kind]
    sub = 'spikes_filtered' if kind == 'filtered' else 'spikes'
    return f'{base}/{sub}/{date}_{suf}.nix'


def input_path(patient, date, data_folder):
    p = spike_path_human(patient, date, data_folder, kind='all')
    return p if os.path.isfile(p) else None


def output_path(patient, date, data_folder):
    return spike_path_human(patient, date, data_folder, kind='filtered')


##### HELPERS ##### (identical to the RS/NATIM script)
def sanitise_for_nix(value):
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
    wf = np.asarray(avg_wf, dtype=float)
    if wf.size == 0 or not np.all(np.isfinite(wf)):
        return np.full_like(wf, np.nan)
    s = np.std(wf)
    if s <= 0:
        return np.zeros_like(wf)
    return (wf - np.mean(wf)) / s


def load_thresholds():
    # NEW: reads the MONKEY thresholds file (MONKEY_DF_FOLDER), not a
    # human-specific one - humans use the same criteria as RS/NATIM.
    p = f'{MONKEY_DF_FOLDER}/filter_metrics_chosen/thresholds.pkl'
    with open(p, 'rb') as f:
        t = pickle.load(f)
    return t['criteria'], t.get('nan_fails', CFG['nan_fails'])


def load_metrics(patient, date):
    p = f'{DF_FOLDER}/filter_metrics_HUMAN/{date}.pkl'
    with open(p, 'rb') as f:
        return pickle.load(f)


##### DECIDE WHICH UNITS SURVIVE #####
CRITERIA, NAN_FAILS = load_thresholds()
print('criteria in use:')
for col, (direc, th) in CRITERIA.items():
    print(f'   {col:28s} keep {direc:5s} {th}')
print(f'   NaN fails: {NAN_FAILS}')

df_met = load_metrics(patient, date)
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

# optional resolution of mutual coincidence pairs - no array grouping for
# humans, since there is only one "array" worth of units per patient/date
df_met['mutual_resolved_drop'] = False
if CFG['resolve_mutual_pairs'] and 'coinc_mutual' in df_met.columns:
    n_res = 0
    cand = df_met[df_met['coinc_mutual'] &
                  (df_met['coinc_ratio_worst'] > CFG['mutual_ratio_th'])]
    by_name = df_met.set_index('cell_name')
    seen = set()
    for idx, r in cand.iterrows():
        a, b = r['cell_name'], r['coinc_partner_name']
        if a in seen or b in seen or b not in by_name.index:
            continue
        seen.update([a, b])
        n_a, n_b = r['n_spikes'], by_name.loc[b, 'n_spikes']
        drop_name = b if n_a >= n_b else a
        m = df_met['cell_name'] == drop_name
        df_met.loc[m, 'mutual_resolved_drop'] = True
        keep_name = a if drop_name == b else b
        mk = df_met['cell_name'] == keep_name
        other = [c for c in flags.columns if c != 'coinc_ratio_worst']
        df_met.loc[mk, 'pass_filter'] = flags.loc[
            df_met.index[mk], other].all(axis=1).values
        n_res += 1
    print(f'resolved {n_res} mutual coincidence pairs')
    df_met.loc[df_met['mutual_resolved_drop'], 'pass_filter'] = False

print(f'{int(df_met["pass_filter"].sum())} of {df_met.shape[0]} units pass '
      f'({100*df_met["pass_filter"].mean():.1f}%)')

# lookup keyed on cell_name only - no array dimension for humans
decision = {}
for _, r in df_met.iterrows():
    decision[r['cell_name']] = r


##### WRITE THE FILE #####
in_p = input_path(patient, date, DATA_FOLDER)
out_p = output_path(patient, date, DATA_FOLDER)
out_dir = os.path.dirname(out_p)
ensure_dir_exists(out_dir)
print('\nreading from: spikes/..._spikes_unfiltered.nix')
print(f'output: {out_p}')

summary = []
if in_p is None:
    print(f'Input file not found for patient {patient}, date {date}.')
    sys.exit(0)
if os.path.isfile(out_p) and not CFG['overwrite']:
    print('output exists, skipping')
    sys.exit(0)

try:
    io_in = neo.NixIO(in_p, 'ro')
    blk_in = io_in.read_block()
    sts_in = blk_in.segments[0].spiketrains

    kept = []
    n_no_decision = 0
    for order, st in enumerate(sts_in):
        name = st.annotations.get('nix_name', None)
        rec = decision.get(name, None)
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
            if wmag.ndim == 3:
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
        name=f'Filtered single units, {patient} {date}',
        description='Units passing the filtering criteria recorded in the '
                    'block annotations. Spike times unchanged; per-spike '
                    'waveforms replaced by the mean waveform and its '
                    'z-scored form (mean over spikes, then that trace '
                    'z-scored) in the unit annotations.',
        date_of_creation=date_str,
        patient=patient,
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

    print(f'--- {len(kept)} of {len(sts_in)} units kept'
          + (f', {n_no_decision} without a metrics entry' if n_no_decision else ''),
          flush=True)
    summary.append({'patient': patient, 'date': date,
                    'n_original': len(sts_in), 'n_kept': len(kept),
                    'n_no_decision': n_no_decision,
                    'path': out_p})
    del blk_in, blk_out, kept

except Exception as e:
    print(f'--- FAILED: {e}')

##### REPORT #####
if not summary:
    print('\nNo file written.')
    sys.exit(0)

df_sum = pd.DataFrame(summary)
tot_o, tot_k = df_sum['n_original'].sum(), df_sum['n_kept'].sum()
print(f'\n{tot_k} of {tot_o} units kept ({100*tot_k/max(tot_o,1):.1f}%)')

sum_dir = f'{DF_FOLDER}/filter_metrics_chosen/write_summary_HUMAN'
ensure_dir_exists(sum_dir)
df_sum.to_csv(f'{sum_dir}/{patient}_{date}.csv', index=False)
