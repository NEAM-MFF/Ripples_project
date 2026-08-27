# Filtering metrics for every sorted unit, computed from the unfiltered
# (_spikes_all_units.nix) files. NEW FILTER VERSION.
#
# ONLY CHANGE FROM THE ORIGINAL: params_analysis.yml is now read from
# code_new_filter/, matching the rest of the new-filter pipeline's
# convention. Everything else - metric computation, DF_FOLDER (still plain
# dataframes/, since this stage produces the new filter rather than
# consuming it), the per-array loop - is unchanged.
#
#   python SUA_filter_metrics_pkl_create_new_filter.py $SLURM_ARRAY_TASK_ID RS
#   python SUA_filter_metrics_pkl_create_new_filter.py $SLURM_ARRAY_TASK_ID NATIM
#
# This is stage 1 of a two-stage scheme. Nothing is filtered here: every unit in
# the file gets a row, carrying both the annotations written by the sorting
# pipeline and the metrics computed here. Thresholds are chosen afterwards, in
# the summary notebook, and applied in a separate script that writes new nix
# files. Keeping computation and thresholding apart means a threshold can be
# changed without recomputing anything.
#
# WHY THESE METRICS ARE RECOMPUTED RATHER THAN READ
# The sorting was run in chunks of 4 channels, which limits several of the
# annotations that SpikeInterface wrote:
#   - sync_spike_2/4/8 compare units only within a chunk, so a unit duplicated
#     across a chunk boundary is invisible to them. On this dataset they are
#     near zero for every unit.
#   - drift_ptp/std/mad need a spatial extent that 4 contacts do not provide,
#     and are identically zero.
#   - sliding_rp_violation returns NaN when it cannot establish a contamination
#     level, which on this dataset was frequent, and NaN is indistinguishable
#     from "heavily contaminated" in the stored value.
# The metrics below are computed across the whole array and are defined for
# every unit with enough spikes, so they are usable where the annotations are
# not.
#
# ANNOTATIONS ARE ALSO CARRIED THROUGH
# waveform_SNR, line_noise_50Hz, line_noise_60Hz, presence_ratio and the
# refractory metrics are read from the file and saved alongside. Note that the
# sorting script replaces NaN with the STRING 'NaN' before writing, since NIX
# cannot store NaN, so the reader below converts those back to real NaN. A
# direct numeric comparison against an unconverted annotation would raise a
# TypeError rather than evaluate false.

from functions_analysis import *
from functions_quality import *
import pandas as pd
import numpy as np
import yaml
import pickle
import neo
import os
import sys
import itertools

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=UserWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml') as f:
    params = yaml.safe_load(f)

MAIN_FOLDER = params['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes'

if len(sys.argv) < 3:
    print("Usage: SUA_filter_metrics_pkl_create_new_filter.py <SLURM_ARRAY_TASK_ID> <RS|NATIM>")
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

OUT_FOLDER = f'{DF_FOLDER}/filter_metrics_{TYPE_REC}'

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
print(f'Filter metrics: {TYPE_REC}, monkey {monkey}, date {date}.')

##### CONFIGURATION #####
CFG = {
    # refractory
    'censored_s':      DEAD_TIME_S,        # 0.2333 ms, 7 samples at 30 kHz
    'rp_grid_s':       np.arange(0.001, 0.0105, 0.00025),
    'min_spikes':      100,                # below this the metrics are undefined

    # coincidence
    'coinc_window_s':  0.0005,

    # Waveform shape. 1500 spikes is ample: the mode separation is unchanged
    # from 4000 (single units 1.4-1.6, a narrow-plus-medium merge 9.5, a
    # narrow-plus-wide merge 19.2 at both), while the per-unit cost falls from
    # about 200 ms to 17 ms because the mixture converges far faster.
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

    # ISI-based descriptive measures. Whether the ISI ratio is used as a
    # criterion is decided in the notebook, not here.
    'isi_refractory_s': 0.0015,
    'burst_th_s':       0.008,
}

# annotations to carry through from the file, if present
ANNOTATION_KEYS = [
    'num_spikes', 'firing_rate', 'presence_ratio',
    'isi_violations_ratio', 'isi_violations_count',
    'rp_contamination', 'rp_violations', 'sliding_rp_violation',
    'amplitude_cutoff', 'amplitude_median', 'amplitude_cv_median',
    'amplitude_cv_range', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    'firing_range', 'drift_ptp', 'drift_std', 'drift_mad',
    'sd_ratio', 'waveform_SNR', 'line_noise_50Hz', 'line_noise_60Hz',
    'KSLabel', 'Label', 'Electrode_ID', 'Array_ID', 'Area',
]


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


def load_all_units(monkey, array, type_rec, date, data_folder=''):
    """Every sorted unit for one array, or None when the file is absent."""
    path = spike_path(monkey, date, array, type_rec, data_folder, kind='all')
    if path is None:
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
    fields such as KSLabel need.
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


##### MAIN LOOP #####
rows = []

for array in range(1, 17):
    print(f'--- array {array} ---', flush=True)
    try:
        import time as _time
        _t0_read = _time.time()
        block = load_all_units(monkey, array, TYPE_REC, date,
                               data_folder=DATA_FOLDER)
        if block is None:
            print('   file not found')
            continue

        sts = block.segments[0].spiketrains
        if len(sts) == 0:
            continue

        t_start = float(sts[0].t_start.rescale('s').magnitude)
        t_stop = float(sts[0].t_stop.rescale('s').magnitude)
        dur = t_stop - t_start

        import time as _time
        _t = {'read': _time.time() - _t0_read}
        _a = _time.time()

        # spike times once, reused by every metric
        st_all = [np.sort(np.asarray(s.times.rescale('s').magnitude, dtype=float))
                  for s in sts]

        # cross-unit coincidence needs the whole array at once, which is
        # precisely what the chunked sorting could not do
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
                d['monkey'] = monkey
                d['date'] = date
                d['array'] = array
                d['train_order'] = cell
                d['type_rec'] = TYPE_REC
                for k in ['Electrode_ID', 'Array_ID', 'Area', 'KSLabel', 'Label']:
                    d[k] = read_annotation(st_neo, k)

                d['n_spikes'] = int(len(st))
                d['duration_s'] = dur
                d['FR_computed'] = len(st) / dur if dur > 0 else np.nan

                # ---------- annotations carried through ----------
                for k in ANNOTATION_KEYS:
                    if k in ('Electrode_ID', 'Array_ID', 'Area', 'KSLabel', 'Label'):
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

                # ---------- coincidence with other units on the array ----------
                d['coinc_ratio_worst'] = float(ratio_w[cell])
                d['coinc_ratio_total'] = float(ratio_t[cell])
                d['coinc_frac'] = float(frac_c[cell])
                d['coinc_frac_worst'] = float(frac_w[cell])
                d['coinc_partner'] = int(worst[cell])
                d['coinc_partner_name'] = (
                    sts[worst[cell]].annotations.get('nix_name', '')
                    if worst[cell] >= 0 else '')
                # mutual pairs are almost always one neuron split in two, and
                # the remedy is to keep one rather than drop both
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
                # Saved so the ISI criterion can be evaluated in the notebook.
                # Whether it is used is decided there, not here.
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
        print(f'Array {array} failed: {e}')

##### SAVING #####
if len(rows) == 0:
    print('No units processed, nothing saved.')
    sys.exit(0)

df = pd.DataFrame(rows)
ensure_dir_exists(OUT_FOLDER)
out_path = f'{OUT_FOLDER}/monkey{monkey}_all_arrays_date_{date}.pkl'
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
