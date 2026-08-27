# Firing-metric functions for RS/FS classification from NATIM (evoked) data.
#
# DESIGN RULE: every function here is pure and stateless - it takes spike times
# (in SECONDS, absolute recording clock) plus window definitions and returns
# numbers. Nothing here reads files, and nothing here ever looks at the waveform
# class. The waveform class is the independent variable and must stay
# untouched by these metrics, otherwise the CCG-based validation becomes circular.
#
# All ISI-derived quantities are computed WITHIN contiguous segments only.
# An ISI is never allowed to span a gap between two windows.

import numpy as np
from scipy.stats import poisson


########## WINDOW CONSTRUCTION ##########


def build_baseline_windows(trial_onsets_s, t_start_s, t_stop_s,
                           min_gap_s=1.0, trim_front_s=0.4, trim_back_s=0.1):
    """
    Builds baseline windows from the inter-trial gaps.

    In the TVSD/THINGS rapid serial paradigm the median SOA is ~0.42 s, so there
    is no usable pre-stimulus baseline inside a normal trial. The long gaps
    (fixation breaks) are the only clean spontaneous periods available.

    trim_front_s : discarded after the preceding trial onset, to let the evoked
                   response decay away.
    trim_back_s  : discarded before the next trial onset, to avoid anticipatory
                   activity.

    Returns array of shape (n_windows, 2) with [start, stop] in seconds.
    """
    onsets = np.sort(np.asarray(trial_onsets_s, dtype=float))
    edges = np.concatenate([onsets, [t_stop_s]])

    windows = []
    for i in range(len(onsets)):
        gap_start = edges[i]
        gap_stop = edges[i + 1]
        if (gap_stop - gap_start) <= min_gap_s:
            continue
        w0 = gap_start + trim_front_s
        w1 = gap_stop - trim_back_s
        if w1 > w0:
            windows.append([w0, w1])

    # the period before the very first trial, if it is long enough
    if (onsets[0] - t_start_s) > min_gap_s:
        windows.insert(0, [t_start_s, onsets[0] - trim_back_s])

    if len(windows) == 0:
        return np.zeros((0, 2))
    return np.array(windows, dtype=float)


def build_trial_windows(trial_onsets_s, w_start_s, w_stop_s):
    """
    Windows defined relative to each trial onset, e.g. (0.03, 0.10) for the
    transient. Returns (n_trials, 2) array in absolute seconds.
    """
    onsets = np.asarray(trial_onsets_s, dtype=float)[:, None]
    return np.hstack([onsets + w_start_s, onsets + w_stop_s])


########## SEGMENT-SAFE PRIMITIVES ##########


def spikes_in_windows(spike_times_s, windows):
    """
    Splits spike times into a list of per-window arrays.
    Windows must be non-overlapping for the rate to be interpretable.
    """
    st = np.asarray(spike_times_s, dtype=float)
    out = []
    for w0, w1 in windows:
        lo, hi = np.searchsorted(st, [w0, w1])
        out.append(st[lo:hi])
    return out


def total_window_duration(windows):
    if len(windows) == 0:
        return 0.0
    return float(np.sum(windows[:, 1] - windows[:, 0]))


def isis_within_segments(spike_segments):
    """
    ISIs (seconds) computed inside each segment separately and then pooled.
    This is the crucial detail - concatenating spike times across trials and
    differencing would create spurious long ISIs at every trial boundary,
    which would inflate CV and destroy the burst measures.
    """
    isi_list = [np.diff(seg) for seg in spike_segments if len(seg) >= 2]
    if len(isi_list) == 0:
        return np.array([])
    return np.concatenate(isi_list)


########## RATE METRICS ##########


def firing_rate(spike_segments, windows):
    dur = total_window_duration(windows)
    if dur <= 0:
        return np.nan
    n = sum(len(s) for s in spike_segments)
    return n / dur


def psth(spike_times_s, trial_onsets_s, t_pre_s=0.1, t_post_s=0.35, bin_s=0.005):
    """
    Trial-averaged PSTH. Returns (rate_hz, bin_centres_s).
    """
    st = np.asarray(spike_times_s, dtype=float)
    onsets = np.asarray(trial_onsets_s, dtype=float)
    n_trials = len(onsets)
    if n_trials == 0 or len(st) == 0:
        edges = np.arange(-t_pre_s, t_post_s + bin_s, bin_s)
        return np.zeros(len(edges) - 1), (edges[:-1] + edges[1:]) / 2

    edges = np.arange(-t_pre_s, t_post_s + bin_s, bin_s)
    counts = np.zeros(len(edges) - 1)
    for t0 in onsets:
        lo, hi = np.searchsorted(st, [t0 - t_pre_s, t0 + t_post_s])
        rel = st[lo:hi] - t0
        if len(rel):
            counts += np.histogram(rel, bins=edges)[0]
    rate = counts / (n_trials * bin_s)
    return rate, (edges[:-1] + edges[1:]) / 2


def peak_evoked_rate(rate, centres, search_from_s=0.0, search_to_s=0.25,
                     smooth_bins=3):
    """
    Peak of the smoothed PSTH inside the search window, and its latency.
    """
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        rate_s = np.convolve(rate, k, mode='same')
    else:
        rate_s = rate
    m = (centres >= search_from_s) & (centres <= search_to_s)
    if m.sum() == 0:
        return np.nan, np.nan
    idx = np.argmax(rate_s[m])
    return float(rate_s[m][idx]), float(centres[m][idx])


def modulation_index(fr_evoked, fr_baseline):
    denom = fr_evoked + fr_baseline
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return (fr_evoked - fr_baseline) / denom


########## LATENCY ##########


def response_latency(rate, centres, fr_baseline_hz, baseline_sd_hz,
                     n_sd=3.0, n_consecutive=2, search_to_s=0.25):
    """
    First PSTH bin after onset exceeding baseline + n_sd*SD, required to stay
    above threshold for n_consecutive bins. Returns latency in seconds or nan.
    """
    if not np.isfinite(baseline_sd_hz) or baseline_sd_hz <= 0:
        return np.nan
    th = fr_baseline_hz + n_sd * baseline_sd_hz
    m = (centres >= 0) & (centres <= search_to_s)
    r = rate[m]
    c = centres[m]
    above = r > th
    for i in range(len(above) - n_consecutive + 1):
        if np.all(above[i:i + n_consecutive]):
            return float(c[i])
    return np.nan


def first_spike_latency_stats(spike_times_s, trial_onsets_s,
                              t_min_s=0.02, t_max_s=0.25):
    """
    Per-trial first-spike latency: median and its trial-to-trial SD (jitter).
    Only trials with at least one spike in the window contribute.
    """
    st = np.asarray(spike_times_s, dtype=float)
    lats = []
    for t0 in np.asarray(trial_onsets_s, dtype=float):
        lo, hi = np.searchsorted(st, [t0 + t_min_s, t0 + t_max_s])
        if hi > lo:
            lats.append(st[lo] - t0)
    if len(lats) < 5:
        return np.nan, np.nan, len(lats)
    lats = np.array(lats)
    return float(np.median(lats)), float(np.std(lats)), len(lats)


########## VARIABILITY ##########


def cv_isi(isis):
    if len(isis) < 3:
        return np.nan
    m = np.mean(isis)
    if m <= 0:
        return np.nan
    return float(np.std(isis) / m)


def cv2(spike_segments):
    """
    CV2 = mean over adjacent ISI pairs of 2|I(n+1)-I(n)| / (I(n+1)+I(n)).
    Robust to slow rate drift, which matters here because the evoked rate is
    strongly non-stationary within a trial.
    """
    vals = []
    for seg in spike_segments:
        if len(seg) < 3:
            continue
        I = np.diff(seg)
        num = 2 * np.abs(I[1:] - I[:-1])
        den = I[1:] + I[:-1]
        good = den > 0
        vals.append(num[good] / den[good])
    if len(vals) == 0:
        return np.nan
    allv = np.concatenate(vals)
    if len(allv) < 3:
        return np.nan
    return float(np.mean(allv))


def local_variation(spike_segments):
    """
    Shinomoto's LV = mean of 3*(I(n)-I(n+1))^2 / (I(n)+I(n+1))^2.
    LV ~ 1 for Poisson, < 1 for regular, > 1 for bursty.
    """
    vals = []
    for seg in spike_segments:
        if len(seg) < 3:
            continue
        I = np.diff(seg)
        num = 3 * (I[:-1] - I[1:]) ** 2
        den = (I[:-1] + I[1:]) ** 2
        good = den > 0
        vals.append(num[good] / den[good])
    if len(vals) == 0:
        return np.nan
    allv = np.concatenate(vals)
    if len(allv) < 3:
        return np.nan
    return float(np.mean(allv))


########## BURSTING ##########


def burst_index(isis, th_s=0.008):
    """Fraction of ISIs shorter than th_s."""
    if len(isis) < 10:
        return np.nan
    return float(np.mean(isis < th_s))


def burst_stats(spike_segments, th_s=0.008, min_spikes=2):
    """
    Fraction of spikes belonging to a burst, and mean spikes per burst.
    A burst is a run of >= min_spikes spikes with consecutive ISIs < th_s.
    """
    n_spikes_total = 0
    n_spikes_in_burst = 0
    burst_lengths = []
    for seg in spike_segments:
        n_spikes_total += len(seg)
        if len(seg) < 2:
            continue
        I = np.diff(seg)
        short = I < th_s
        i = 0
        while i < len(short):
            if short[i]:
                j = i
                while j < len(short) and short[j]:
                    j += 1
                length = (j - i) + 1
                if length >= min_spikes:
                    burst_lengths.append(length)
                    n_spikes_in_burst += length
                i = j
            else:
                i += 1
    if n_spikes_total == 0:
        return np.nan, np.nan
    frac = n_spikes_in_burst / n_spikes_total
    spb = float(np.mean(burst_lengths)) if burst_lengths else np.nan
    return float(frac), spb


########## ADAPTATION ##########


def adaptation_ratio(rate, centres, early=(0.03, 0.08), late=(0.12, 0.25)):
    """
    Ratio of late to early PSTH rate.

    NOTE ON INTERPRETATION: with a ~0.42 s SOA the "late" window is short and
    partly contaminated by the response to the preceding trial. This is a
    weaker adaptation measure than one obtained from a long sustained stimulus,
    and it should be reported as a PSTH decay ratio rather than as classical
    spike-frequency adaptation. Low values mean strong decay.
    """
    me = (centres >= early[0]) & (centres < early[1])
    ml = (centres >= late[0]) & (centres < late[1])
    if me.sum() == 0 or ml.sum() == 0:
        return np.nan
    r_early = np.mean(rate[me])
    if r_early <= 0:
        return np.nan
    return float(np.mean(rate[ml]) / r_early)


########## RESPONSIVENESS AND RELIABILITY ##########


def responsiveness(spike_times_s, trial_onsets_s,
                   pre=(-0.10, 0.0), post=(0.03, 0.15)):
    """
    Paired comparison of per-trial pre vs post spike counts.
    Returns (wilcoxon_p, mean_count_pre, mean_count_post).

    The pre window here is NOT a clean baseline (the previous stimulus is only
    ~0.42 s back). It is used only as a within-trial reference for detecting
    onset-locked modulation.
    """
    from scipy.stats import wilcoxon
    st = np.asarray(spike_times_s, dtype=float)
    n_pre, n_post = [], []
    for t0 in np.asarray(trial_onsets_s, dtype=float):
        a, b = np.searchsorted(st, [t0 + pre[0], t0 + pre[1]])
        c, d = np.searchsorted(st, [t0 + post[0], t0 + post[1]])
        n_pre.append(b - a)
        n_post.append(d - c)
    n_pre = np.array(n_pre, dtype=float)
    n_post = np.array(n_post, dtype=float)
    # normalise to rate so unequal window lengths are comparable
    r_pre = n_pre / (pre[1] - pre[0])
    r_post = n_post / (post[1] - post[0])
    if np.all(r_pre == r_post) or len(r_pre) < 10:
        return np.nan, float(np.mean(r_pre)), float(np.mean(r_post))
    try:
        _, p = wilcoxon(r_post, r_pre)
    except ValueError:
        p = np.nan
    return float(p), float(np.mean(r_pre)), float(np.mean(r_post))


def repeat_reliability(spike_times_s, onsets_by_image, window=(0.03, 0.25)):
    """
    Trial-to-trial reliability on the repeated test images.

    onsets_by_image : dict {image_index: array of onsets}
    Returns mean over images of the pairwise correlation between single-trial
    spike counts... with so few repeats (1-5) we instead use the ratio of
    across-image variance to within-image variance of the spike count, which is
    a simple one-way ANOVA style signal-to-noise (higher = more reliable/tuned).
    """
    counts_by_img = []
    st = np.asarray(spike_times_s, dtype=float)
    for img, onsets in onsets_by_image.items():
        if len(onsets) < 2:
            continue
        c = []
        for t0 in onsets:
            a, b = np.searchsorted(st, [t0 + window[0], t0 + window[1]])
            c.append(b - a)
        counts_by_img.append(np.array(c, dtype=float))
    if len(counts_by_img) < 5:
        return np.nan
    within = np.mean([np.var(c, ddof=1) for c in counts_by_img if len(c) > 1])
    means = np.array([np.mean(c) for c in counts_by_img])
    between = np.var(means, ddof=1)
    if within <= 0:
        return np.nan
    return float(between / within)


########## BOOTSTRAP ##########


def bootstrap_ci(values, func, n_boot=200, alpha=0.05, seed=0):
    """
    Percentile bootstrap over a 1-D sample. Used for the metrics where a
    per-unit confidence interval is cheap to get (rates, burst index).
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    n = len(v)
    for i in range(n_boot):
        stats[i] = func(v[rng.integers(0, n, n)])
    lo = np.nanpercentile(stats, 100 * alpha / 2)
    hi = np.nanpercentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


########## SORTING QUALITY ##########
#
# NOTE: the spike files already carry a full SpikeInterface quality metric suite
# as neo annotations (sliding_rp_violation, amplitude_cutoff, sync_spike_*,
# drift_*, sd_ratio, waveform_SNR and others). Those are the reference
# implementations and should be preferred. Use read_quality_annotations below.
#
# The functions in this block recompute a subset of those metrics from spike
# times alone. They are retained only as an independent cross-check and are no
# longer called by the analysis scripts. Note that the sorter dead time in this
# dataset is 0.2333 ms (7 samples at 30 kHz), not the 0.5 ms assumed by the
# original default.


QUALITY_ANNOTATION_KEYS = [
    'num_spikes', 'firing_rate', 'presence_ratio',
    'isi_violations_ratio', 'isi_violations_count',
    'rp_contamination', 'rp_violations', 'sliding_rp_violation',
    'amplitude_cutoff', 'amplitude_median',
    'amplitude_cv_median', 'amplitude_cv_range',
    'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    'firing_range', 'drift_ptp', 'drift_std', 'drift_mad',
    'sd_ratio', 'waveform_SNR',
]


def read_quality_annotations(spike_train, prefix='qm_', keys=None):
    """
    Pull the precomputed SpikeInterface quality metrics off a neo SpikeTrain.

    Values are coerced to plain floats. Missing keys and values that cannot be
    coerced return nan rather than raising, since not every recording
    necessarily carries the full set.

    Returns a dict with keys prefixed, e.g. 'qm_sliding_rp_violation'.
    """
    if keys is None:
        keys = QUALITY_ANNOTATION_KEYS
    ann = spike_train.annotations
    out = {}
    for k in keys:
        v = ann.get(k, np.nan)
        try:
            v = np.asarray(v).ravel()
            v = float(v[0]) if v.size else np.nan
        except (TypeError, ValueError):
            v = np.nan
        out[f'{prefix}{k}'] = v
    return out


def isi_violation_rate(spike_times_s, refractory_s=0.0015, censored_s=0.000233):
    """
    Raw fraction of ISIs falling inside the refractory window.

    NOTE: this quantity scales with the square of the firing rate, so a fixed
    threshold on it discriminates against fast-firing units. It is reported for
    transparency only. Threshold on refractory_contamination instead.

    Returns (fraction, n_violations).
    """
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) < 2:
        return np.nan, 0
    isi = np.diff(st)
    n_viol = int(np.sum((isi < refractory_s) & (isi >= censored_s)))
    return n_viol / len(isi), n_viol


def refractory_contamination(spike_times_s, refractory_s=0.0015,
                             censored_s=0.000233, t_start_s=None, t_stop_s=None):
    """
    Estimated fraction of spikes contributed by other neurons, derived from the
    number of refractory period violations (Hill et al. 2011, the quantity
    reported by Kilosort as ContamPct).

    Solves  n_v = 2 f (1-f) N^2 (t_r - t_c) / T  for the smaller root of f.

    Because the expected violation count is corrected for firing rate and spike
    count, this can be thresholded uniformly across units firing at very
    different rates, which the raw violation fraction cannot.

    Returns f in [0,1]. Returns 1.0 when no real root exists, which means the
    observed violation count is at or above the level expected for a fully
    contaminated cluster.
    """
    st = np.asarray(spike_times_s, dtype=float)
    N = len(st)
    if N < 10:
        return np.nan
    if t_start_s is not None and t_stop_s is not None:
        T = t_stop_s - t_start_s
    else:
        T = st[-1] - st[0]
    if T <= 0:
        return np.nan

    isi = np.diff(st)
    n_v = int(np.sum((isi < refractory_s) & (isi >= censored_s)))
    tau = refractory_s - censored_s
    if tau <= 0:
        return np.nan

    # f^2 - f + n_v*T/(2 N^2 tau) = 0
    c = n_v * T / (2 * N ** 2 * tau)
    disc = 1 - 4 * c
    if disc < 0:
        return 1.0
    return float((1 - np.sqrt(disc)) / 2)


def presence_ratio(spike_times_s, t_start_s, t_stop_s, n_bins=100):
    """
    Fraction of equal-width time bins containing at least one spike. Catches
    units that appear or disappear part-way through a recording, typically
    because of electrode drift.
    """
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) == 0 or t_stop_s <= t_start_s:
        return np.nan
    edges = np.linspace(t_start_s, t_stop_s, n_bins + 1)
    counts = np.histogram(st, bins=edges)[0]
    return float(np.mean(counts > 0))
def sliding_rp_contamination(spike_times_s, t_start_s=None, t_stop_s=None,
                             rp_grid_s=None, contam_grid=None, conf_level=0.9):
    """
    Sliding refractory period metric (Hill et al. 2011; IBL implementation).

    Rather than assuming one refractory period, tests a grid of candidate
    refractory periods and asks, for each, whether the observed violation count
    is low enough to rule out a given contamination level with the required
    confidence. Returns the LOWEST contamination level that can be confidently
    ruled in across all tested refractory periods.

    This is more robust than a fixed-window estimate because the true refractory
    period varies between cell types, and because bursting neurons have unusual
    ISI structure just above the refractory period that a fixed window
    misattributes to contamination.

    Returns
    -------
    min_contam : lowest contamination level passing at conf_level (0 = clean)
    best_rp_s  : refractory period at which that was achieved
    """
    st = np.asarray(spike_times_s, dtype=float)
    N = len(st)
    if N < 50:
        return np.nan, np.nan
    T = (t_stop_s - t_start_s) if t_start_s is not None else (st[-1] - st[0])
    if T <= 0:
        return np.nan, np.nan

    if rp_grid_s is None:
        rp_grid_s = np.arange(0.0005, 0.0105, 0.00025)
    if contam_grid is None:
        contam_grid = np.arange(0.005, 0.355, 0.005)

    isi = np.diff(st)
    rate = N / T

    best = (np.nan, np.nan)
    for c in contam_grid:
        # can we rule out contamination >= c at every refractory period?
        ok = False
        for rp in rp_grid_s:
            n_obs = int(np.sum(isi < rp))
            # expected violations if contamination were exactly c
            n_exp = 2 * rp * N * rate * c * (1 - c)
            if n_exp <= 0:
                continue
            # P(observing <= n_obs | contamination == c). If this is small, the
            # observed count is far below what contamination c would produce, so
            # contamination >= c is rejected.
            conf = 1.0 - poisson.cdf(n_obs, n_exp)
            if conf > conf_level:
                ok = True
                best_rp = rp
                break
        if ok:
            return float(c), float(best_rp)
    # Grid exhausted: contamination could not be ruled below the ceiling.
    # Return the ceiling rather than nan, so this case is not confused with
    # insufficient data and is not silently dropped by dropna().
    return float(contam_grid[-1]), np.nan


def isi_refractory_dip(spike_times_s, rp_s=0.002, ref_lo_s=0.020, ref_hi_s=0.050):
    """
    Ratio of ISI density inside the refractory window to density in a
    reference window well outside it.

    A cleanly isolated unit has near-zero density below 2 ms, giving a ratio
    near 0. A merged cluster has no refractory dip and gives a ratio near 1.
    Independent of the Poisson assumptions in the contamination estimate.
    """
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) < 50:
        return np.nan
    isi = np.diff(st)
    d_rp = np.sum(isi < rp_s) / rp_s
    d_ref = np.sum((isi >= ref_lo_s) & (isi < ref_hi_s)) / (ref_hi_s - ref_lo_s)
    if d_ref <= 0:
        return np.nan
    return float(d_rp / d_ref)


def rate_stability(spike_times_s, t_start_s, t_stop_s, n_bins=50):
    """
    Coefficient of variation of the firing rate across coarse time bins, and the
    max/median ratio. Large values indicate drift or a unit appearing partway
    through, which often accompanies unstable cluster assignment.
    """
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) < 50 or t_stop_s <= t_start_s:
        return np.nan, np.nan
    edges = np.linspace(t_start_s, t_stop_s, n_bins + 1)
    c = np.histogram(st, bins=edges)[0].astype(float)
    m = np.mean(c)
    if m <= 0:
        return np.nan, np.nan
    med = np.median(c)
    return float(np.std(c) / m), float(np.max(c) / med) if med > 0 else np.nan
