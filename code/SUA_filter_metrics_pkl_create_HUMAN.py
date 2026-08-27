# Filtering metrics for every sorted human unit, computed from the unfiltered
# (_spikes_unfiltered.nix) files. Stage 1 of the same two-stage scheme used
# for RS/NATIM (SUA_filter_metrics_pkl_create.py + SUA_write_filtered_nix.py):
# nothing is filtered here, every unit gets a row of metrics, thresholds are
# chosen afterwards and applied in a separate script.
#
#   python SUA_filter_metrics_pkl_create_HUMAN.py $SLURM_ARRAY_TASK_ID
#
# WHAT'S DIFFERENT FROM THE MONKEY (RS/NATIM) VERSION
#   1. No arrays. A monkey recording is split across 16 independently-sorted
#      4-channel chunks; a human recording here is one continuous probe/depth-
#      electrode recording with a single spike file per patient/date. The
#      per-array loop is gone, and coincidence_matrix() is computed once
#      across every unit in that date's file instead of once per array.
#   2. Filenames carry no "KS4" suffix - confirmed by the user - so the path
#      is {data_folder}/{patient}/spontaneous/spikes/{date}_spikes_unfiltered.nix
#      rather than the monkey .../spikes_KS4/..._spikes_KS4_unfiltered.nix.
#   3. "date" here is not a calendar date alone - per the user, it already
#      encodes the full recording identity (e.g.
#      'Patient1_2018_10_22_spontaneous_001'), taken directly from
#      params_analysis['dates_human'][patient]. It is used as-is, unchanged.
#   4. Per-unit identity annotations are human-specific: 'new_electrode_ids'
#      and 'channel_ids' (per SUA_HUMAN_prop_pkl_create.py) instead of the
#      monkey's 'Electrode_ID'/'Array_ID'/'Area'. 'KSLabel'/'Label' are kept,
#      since both pipelines went through the same KS4-family sorter.
#   5. Output root is dataframes_human/ (matching the existing human
#      scripts' convention), not dataframes/ - this mirrors dataframes_human
#      already being the human analogue of the monkey pipeline's plain
#      dataframes/ root. filter_metrics_HUMAN/ and filter_metrics_chosen/
#      live there, same relationship as filter_metrics_RS/NATIM do under
#      dataframes/ for monkeys.
#
# WHAT'S UNCHANGED
# Every metric computation itself (sliding_violation_ratio, coincidence_matrix,
# waveform_shape_heterogeneity, amplitude_distribution_metrics,
# waveform_consistency, burst_amplitude_attenuation, presence_ratio,
# rate_stability) is called exactly as in the RS/NATIM script, from the same
# functions_quality.py - no metric definitions are altered for humans.
#
# ONE OPEN QUESTION, NOT DECIDED HERE: whether the existing
# dataframes/filter_metrics_chosen/thresholds.pkl (chosen on monkey data)
# should be reused for humans, or whether a separate F22-style pass should
# pick human-specific thresholds from this script's output before
# SUA_write_filtered_nix_HUMAN.py is run. That decision happens after this
# script's output exists, not before.

from functions_analysis import *
from functions_quality import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import os
import sys

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml') as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DATA_FOLDER = params['human_data_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_human'  # human analogue of plain dataframes/ for RS/NATIM
DATES = params['dates_human']

PATIENT_LIST = ['Patient2', 'Patient3']

OUT_FOLDER = f'{DF_FOLDER}/filter_metrics_HUMAN'

pairs_pd = []
for patient in PATIENT_LIST:
    for d in DATES[patient]:
        pairs_pd.append((patient, d))

if len(sys.argv) < 2:
    print(f"Usage: SUA_filter_metrics_pkl_create_HUMAN.py <SLURM_ARRAY_TASK_ID>")
    print(f"{len(pairs_pd)} (patient, date) pairs -> use --array=0-{len(pairs_pd)-1}")
    sys.exit(1)

task_id = int(sys.argv[1])
print(f'{len(pairs_pd)} (patient, date) pairs -> use --array=0-{len(pairs_pd)-1}')
if task_id >= len(pairs_pd):
    print(f'task_id {task_id} out of range.')
    sys.exit(0)

patient, date = pairs_pd[task_id]
print(f'Filter metrics: HUMAN, patient {patient}, date {date}.')

##### CONFIGURATION ##### (identical to the RS/NATIM script)
CFG = {
    # refractory
    'censored_s':      DEAD_TIME_S,        # 0.2333 ms, 7 samples at 30 kHz
    'rp_grid_s':       np.arange(0.001, 0.0105, 0.00025),
    'min_spikes':      100,                # below this the metrics are undefined

    # coincidence
    'coinc_window_s':  0.0005,

    # waveform shape
    'wf_max_spikes':   1500,
    'shape_min_spikes': 200,

    # amplitude attenuation within bursts
    'atten_short_s':   0.008,
    'atten_long_lo_s': 0.050,
    'atten_long_hi_s': 0.200,
    'atten_min_pairs': 30,

    # stability
    'presence_bins':   100,
    'stability_bins':  50,

    # ISI-based descriptive measures, saved but not necessarily used as a criterion
    'isi_refractory_s': 0.0015,
    'burst_th_s':       0.008,
}

# annotations to carry through from the file, if present - same generic
# SpikeInterface fields as the RS/NATIM script; the monkey-only identity
# fields (Electrode_ID, Array_ID, Area) are handled separately below
ANNOTATION_KEYS = [
    'num_spikes', 'firing_rate', 'presence_ratio',
    'isi_violations_ratio', 'isi_violations_count',
    'rp_contamination', 'rp_violations', 'sliding_rp_violation',
    'amplitude_cutoff', 'amplitude_median', 'amplitude_cv_median',
    'amplitude_cv_range', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    'firing_range', 'drift_ptp', 'drift_std', 'drift_mad',
    'sd_ratio', 'waveform_SNR', 'line_noise_50Hz', 'line_noise_60Hz',
    'KSLabel', 'Label',
]


##### SPIKE FILE PATH #####
# {data_folder}/{patient}/spontaneous/spikes/{date}_spikes_unfiltered.nix
# 'date' already encodes the full recording identity, per the user.

def spike_path_human(patient, date, data_folder, kind='all'):
    """
    Path to the human spike file. kind is 'all' for every sorted unit
    (spikes_unfiltered.nix), 'good' for the file currently used elsewhere in
    the pipeline (plain spikes.nix, no KS4 suffix for humans), or 'filtered'
    for the output of this filtering (spikes_filtered.nix). Returns None if
    absent.
    """
    base = f'{data_folder}/{patient}/spontaneous'
    suf = {'all': 'spikes_unfiltered', 'good': 'spikes', 'filtered': 'spikes_filtered'}[kind]
    sub = 'spikes' if kind != 'filtered' else 'spikes_filtered'
    p = f'{base}/{sub}/{date}_{suf}.nix'
    return p if os.path.isfile(p) else None


def load_all_units_human(patient, date, data_folder=''):
    """Every sorted unit for one patient/date, or None when the file is absent."""
    path = spike_path_human(patient, date, data_folder, kind='all')
    if path is None:
        print(f'   file not found: expected {data_folder}/{patient}/spontaneous/spikes/{date}_spikes_unfiltered.nix')
        return None
    try:
        return neo.NixIO(path, 'ro').read_block()
    except Exception:
        return None


def read_annotation(st_neo, key):
    """
    One annotation, converted to float where possible.

    The sorting script writes NaN as the string 'NaN', so those are converted
    back. Non-numeric values are returned as strings, which is what identity
    fields such as KSLabel need. Same as the RS/NATIM script.
    """
    v = st_neo.annotations.get(key, None)
    if v is None:
        return np.nan
    if isinstance(v, str):
        if v.strip().lower() in ('nan', 'none', ''):
            return np.nan
        try:
            return float(v)
        except ValueError:
            return v
    try:
        a = np.asarray(v).ravel()
        return float(a[0]) if a.size else np.nan
    except (TypeError, ValueError):
        return str(v)


##### MAIN #####
rows = []

try:
    import time as _time
    _t0_read = _time.time()
    block = load_all_units_human(patient, date, data_folder=DATA_FOLDER)
    if block is None:
        print('File not found, nothing to do.')
        sys.exit(0)

    sts = block.segments[0].spiketrains
    if len(sts) == 0:
        print('No spiketrains in file.')
        sys.exit(0)

    t_start = float(sts[0].t_start.rescale('s').magnitude)
    t_stop = float(sts[0].t_stop.rescale('s').magnitude)
    dur = t_stop - t_start

    _t = {'read': _time.time() - _t0_read}
    _a = _time.time()

    # spike times once, reused by every metric
    st_all = [np.sort(np.asarray(s.times.rescale('s').magnitude, dtype=float))
              for s in sts]

    # cross-unit coincidence needs every unit from this recording at once -
    # for humans that's the whole file, since there is no array split
    frac_c, worst, frac_w, ratio_w, ratio_t = coincidence_matrix(
        st_all, window_s=CFG['coinc_window_s'])
    _t['coinc'] = _time.time() - _a
    _a = _time.time()

    print(f'   {len(sts)} units, span {dur/60:.1f} min', flush=True)

    for cell, st_neo in enumerate(sts):
        try:
            st = st_all[cell]
            d = {}

            # ---------- identity ----------
            d['cell_name'] = st_neo.annotations.get('nix_name', f'unit_{cell}')
            d['patient'] = patient
            d['date'] = date
            d['train_order'] = cell
            d['type_rec'] = 'HUMAN'
            d['new_electrode_ids'] = read_annotation(st_neo, 'new_electrode_ids')
            d['channel_ids'] = read_annotation(st_neo, 'channel_ids')
            for k in ['KSLabel', 'Label']:
                d[k] = read_annotation(st_neo, k)

            d['n_spikes'] = int(len(st))
            d['duration_s'] = dur
            d['FR_computed'] = len(st) / dur if dur > 0 else np.nan

            # ---------- annotations carried through ----------
            for k in ANNOTATION_KEYS:
                if k in ('KSLabel', 'Label'):
                    continue
                d[f'ann_{k}'] = read_annotation(st_neo, k)

            # ---------- refractory, computed here ----------
            sv = sliding_violation_ratio(
                st, t_start, t_stop,
                rp_grid_s=CFG['rp_grid_s'], censored_s=CFG['censored_s'],
                min_spikes=CFG['min_spikes'])
            d['viol_ratio_min'] = sv['ratio_min']
            d['viol_rp_at_min_s'] = sv['rp_at_min']
            d['viol_ratio_1p5ms'] = sv['ratio_1p5ms']
            d['viol_ratio_slope'] = sv['ratio_slope']
            d['n_viol_1p5ms'] = sv['n_viol_1p5ms']
            d['enough_spikes'] = sv['enough_spikes']

            # ---------- coincidence with other units in this recording ----------
            d['coinc_ratio_worst'] = float(ratio_w[cell])
            d['coinc_ratio_total'] = float(ratio_t[cell])
            d['coinc_frac'] = float(frac_c[cell])
            d['coinc_frac_worst'] = float(frac_w[cell])
            d['coinc_partner'] = int(worst[cell])
            d['coinc_partner_name'] = (
                sts[worst[cell]].annotations.get('nix_name', '')
                if worst[cell] >= 0 else '')
            j = int(worst[cell])
            d['coinc_mutual'] = bool(j >= 0 and int(worst[j]) == cell)

            # ---------- waveform-derived ----------
            wf = st_neo.waveforms
            if wf is not None and getattr(wf, 'ndim', 0) == 2 and wf.shape[0] >= 20:
                wmag = np.asarray(
                    wf.magnitude if hasattr(wf, 'magnitude') else wf,
                    dtype=np.float32)
                amps, trough_idx = spike_amplitudes(wmag)
                d['wf_trough_idx'] = trough_idx
                d['wf_n_samples'] = int(wmag.shape[1])

                d.update(waveform_shape_heterogeneity(
                    wmag, max_spikes=CFG['wf_max_spikes'],
                    min_spikes=CFG['shape_min_spikes']))

                ad = amplitude_distribution_metrics(amps)
                d['amp_median'] = ad['amp_median']
                d['amp_cv'] = ad['amp_cv']
                d['amp_mode_separation'] = ad['amp_mode_separation']
                d['amp_cutoff_computed'] = ad['amp_cutoff']

                wc = waveform_consistency(wmag, max_spikes=CFG['wf_max_spikes'])
                d['wf_corr_median'] = wc['wf_corr_median']
                d['wf_corr_p05'] = wc['wf_corr_p05']
                d['wf_snr_computed'] = wc['wf_snr']

                d.update(burst_amplitude_attenuation(
                    st, amps, short_s=CFG['atten_short_s'],
                    long_lo_s=CFG['atten_long_lo_s'],
                    long_hi_s=CFG['atten_long_hi_s'],
                    min_pairs=CFG['atten_min_pairs']))

                d['has_waveforms'] = True
                del wmag, amps
            else:
                d['has_waveforms'] = False
                for k in ['wf_shape_pc1_modesep', 'wf_shape_pc1_bic',
                          'wf_width_cv', 'wf_resid_ratio', 'amp_median',
                          'amp_cv', 'amp_mode_separation',
                          'amp_cutoff_computed', 'wf_corr_median',
                          'wf_corr_p05', 'wf_snr_computed', 'atten_short',
                          'atten_long', 'atten_index', 'p_atten']:
                    d[k] = np.nan
                d['n_short'] = 0
                d['wf_trough_idx'] = -1
                d['wf_n_samples'] = 0

            # ---------- stability ----------
            d['presence_ratio_computed'] = presence_ratio(
                st, t_start, t_stop, n_bins=CFG['presence_bins'])
            d['rate_cv'], d['rate_max_over_med'] = rate_stability(
                st, t_start, t_stop, n_bins=CFG['stability_bins'])

            # ---------- ISI descriptives ----------
            if len(st) >= 2:
                isi = np.diff(st)
                d['isi_frac_below_rp'] = float(
                    np.mean(isi < CFG['isi_refractory_s']))
                d['isi_frac_below_burst'] = float(
                    np.mean(isi < CFG['burst_th_s']))
                d['isi_median_s'] = float(np.median(isi))
                d['isi_cv'] = float(np.std(isi)/np.mean(isi)) \
                    if np.mean(isi) > 0 else np.nan
            else:
                for k in ['isi_frac_below_rp', 'isi_frac_below_burst',
                          'isi_median_s', 'isi_cv']:
                    d[k] = np.nan

            rows.append(d)

        except Exception as e:
            print(f'   unit {cell} failed: {e}')

    _t['units'] = _time.time() - _a
    print(f'   timing: read {_t["read"]:.1f}s  coincidence {_t["coinc"]:.1f}s  '
          f'per-unit metrics {_t["units"]:.1f}s', flush=True)
    del block, st_all

except Exception as e:
    print(f'Patient {patient}, date {date} failed: {e}')

##### SAVING #####
if len(rows) == 0:
    print('No units processed, nothing saved.')
    sys.exit(0)

df = pd.DataFrame(rows)
ensure_dir_exists(OUT_FOLDER)
out_path = f'{OUT_FOLDER}/{date}.pkl'
df.to_pickle(out_path)

n = df.shape[0]
print(f'\nSaved {n} units to {out_path}')
print(f'  with enough spikes for the refractory metrics: '
      f'{int(df["enough_spikes"].sum())} / {n}')
print(f'  with waveforms: {int(df["has_waveforms"].sum())} / {n}')
print(f'  in a mutual coincidence pair: {int(df["coinc_mutual"].sum())} / {n}')
print()
print('COMPUTED METRICS:')
print(df[['viol_ratio_min', 'viol_ratio_slope', 'coinc_ratio_worst',
          'wf_shape_pc1_modesep', 'wf_snr_computed', 'FR_computed',
          'presence_ratio_computed']]
      .describe(percentiles=[.05, .25, .5, .75, .95]).round(4).to_string())
print()
print('ANNOTATIONS CARRIED THROUGH:')
ann_show = [c for c in ['ann_waveform_SNR', 'ann_line_noise_50Hz',
                        'ann_line_noise_60Hz', 'ann_isi_violations_ratio',
                        'ann_sliding_rp_violation', 'ann_presence_ratio']
            if c in df.columns]
print(df[ann_show].describe(percentiles=[.05, .5, .95]).round(4).to_string())
print()
print('NaN counts, computed metrics:')
for c in ['viol_ratio_min', 'coinc_ratio_worst', 'wf_shape_pc1_modesep',
          'wf_snr_computed']:
    n_nan = int(df[c].isna().sum())
    print(f'  {c:24s} {n_nan:5d}')
print()
print('NaN counts, annotations:')
for c in ann_show:
    print(f'  {c:28s} {int(df[c].isna().sum()):5d}')
