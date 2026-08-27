# Sorting quality metrics computed from spike times only.
#
# DESIGN PRINCIPLES
#
# 1. NO NaN FOR CONTAMINATION. Every contamination-like metric returns a defined
#    float for any unit with enough spikes. Where a quantity is unidentifiable,
#    the metric saturates at an interpretable ceiling rather than returning NaN,
#    so that "too contaminated to estimate" is never confused with "missing" and
#    is never silently dropped by dropna(). A separate boolean column records
#    whether the unit had enough data at all.
#
# 2. RATIOS, NOT ROOTS. The primary metric is the ratio of observed refractory
#    violations to the number expected from a rate-matched Poisson process. This
#    is always defined, is corrected for firing rate and spike count, and does
#    not require solving the Hill quadratic, which becomes unidentifiable above
#    50 percent contamination and is the source of the saturation seen in the
#    SpikeInterface output.
#
# 3. CROSS-UNIT COINCIDENCE IS COMPUTED DIRECTLY. Given all spike trains from an
#    array, the fraction of a unit's spikes coincident with spikes from other
#    units is measured rather than assumed. A single neuron cannot produce
#    coincidences with other neurons regardless of its firing pattern, so this
#    is the one metric that separates contamination from genuine bursting.
#
# NOT COMPUTABLE HERE: anything requiring per-spike amplitudes or waveforms
# (amplitude cutoff, within-burst amplitude attenuation, isolation distance,
# L-ratio). The processed files retain only per-unit average waveforms.

import numpy as np


# Sorter dead time for this dataset: 7 samples at 30 kHz.
DEAD_TIME_S = 0.2333e-3


########## PRIMARY: REFRACTORY VIOLATIONS ##########


def violation_ratio(spike_times_s, refractory_s=0.0015,
                    censored_s=DEAD_TIME_S, t_start_s=None, t_stop_s=None):
    """
    Ratio of observed refractory violations to the number expected from a
    rate-matched Poisson process over the same window.

        ratio = n_observed / (N * (t_r - t_c) * N / T)

    Interpretation:
        0.0        no violations, cleanly isolated
        ~0.0-0.1   well isolated
        ~1.0       violations at chance level: the spike train carries no
                   refractory structure at all, i.e. fully multi-unit
        >1.0       more violations than chance, indicating either synchronous
                   contamination or a common artifact

    Always defined for N >= 2 and T > 0. Unbounded above, which is a feature:
    it distinguishes degrees of contamination that the Hill estimate collapses
    to a single saturated value.
    """
    st = np.asarray(spike_times_s, dtype=float)
    N = len(st)
    if N < 2:
        return np.nan
    T = (t_stop_s - t_start_s) if (t_start_s is not None and t_stop_s is not None) \
        else (st[-1] - st[0])
    if T <= 0:
        return np.nan
    tau = refractory_s - censored_s
    if tau <= 0:
        return np.nan

    isi = np.diff(st)
    n_obs = float(np.sum((isi < refractory_s) & (isi >= censored_s)))
    n_exp = tau * N * N / T          # adjacent-ISI expectation for a Poisson
    if n_exp <= 0:                    # train of the same rate and duration
        return np.nan
    return float(n_obs / n_exp)


def sliding_violation_ratio(spike_times_s, t_start_s=None, t_stop_s=None,
                            rp_grid_s=None, censored_s=DEAD_TIME_S,
                            min_spikes=100):
    """
    Violation ratio evaluated across a grid of candidate refractory periods.

    The true refractory period is unknown and differs between cell types, so a
    single fixed window is unreliable. This returns a small summary of the whole
    profile:

    ratio_min   : the most favourable reading across the grid. A unit that is
                  clean at ANY plausible refractory period gets a low value.
                  This is the primary contamination measure.
    rp_at_min   : the refractory period achieving that minimum, i.e. the
                  effective refractory period of the unit.
    ratio_1p5ms : the ratio at a conventional 1.5 ms window, for comparability
                  with the literature.
    ratio_slope : ratio at 4 ms minus ratio at 1 ms. A real neuron has a rising
                  profile (violations accumulate only beyond the refractory
                  period); a fully mixed train has a flat profile near 1.

    Returns a dict. Values are NaN only when the unit has fewer than min_spikes
    spikes, which is recorded separately by the caller as insufficient data
    rather than as contamination.
    """
    st = np.asarray(spike_times_s, dtype=float)
    N = len(st)
    out = {'ratio_min': np.nan, 'rp_at_min': np.nan,
           'ratio_1p5ms': np.nan, 'ratio_slope': np.nan,
           'n_viol_1p5ms': 0, 'enough_spikes': False}
    if N < min_spikes:
        return out
    T = (t_stop_s - t_start_s) if (t_start_s is not None and t_stop_s is not None) \
        else (st[-1] - st[0])
    if T <= 0:
        return out

    if rp_grid_s is None:
        rp_grid_s = np.arange(0.001, 0.0105, 0.00025)

    isi = np.diff(st)
    ratios = np.empty(len(rp_grid_s))
    for i, rp in enumerate(rp_grid_s):
        tau = rp - censored_s
        if tau <= 0:
            ratios[i] = np.inf
            continue
        n_obs = float(np.sum((isi < rp) & (isi >= censored_s)))
        n_exp = tau * N * N / T
        ratios[i] = n_obs / n_exp if n_exp > 0 else np.inf

    finite = np.isfinite(ratios)
    if not finite.any():
        return out

    i_min = int(np.nanargmin(np.where(finite, ratios, np.inf)))
    out['ratio_min'] = float(ratios[i_min])
    out['rp_at_min'] = float(rp_grid_s[i_min])
    out['enough_spikes'] = True

    def _at(target):
        j = int(np.argmin(np.abs(rp_grid_s - target)))
        return float(ratios[j]) if np.isfinite(ratios[j]) else np.nan

    out['ratio_1p5ms'] = _at(0.0015)
    r1, r4 = _at(0.001), _at(0.004)
    out['ratio_slope'] = float(r4 - r1) if np.isfinite(r1) and np.isfinite(r4) else np.nan
    out['n_viol_1p5ms'] = int(np.sum((isi < 0.0015) & (isi >= censored_s)))
    return out


########## ISI HISTOGRAM SHAPE ##########


def isi_dip_ratio(spike_times_s, rp_s=0.002, ref_lo_s=0.020, ref_hi_s=0.050,
                  min_spikes=100):
    """
    Ratio of ISI density inside the refractory window to density in a reference
    window well outside it.

    Makes no assumption about the contaminating process, so it fails on
    different grounds from the violation ratio. Where the two agree the
    classification is trustworthy; where they disagree the unit warrants
    individual inspection.

    0 for a clean unit, order 1 for a fully mixed train.
    """
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) < min_spikes:
        return np.nan
    isi = np.diff(st)
    d_rp = np.sum(isi < rp_s) / rp_s
    d_ref = np.sum((isi >= ref_lo_s) & (isi < ref_hi_s)) / (ref_hi_s - ref_lo_s)
    if d_ref <= 0:
        # no reference mass: the unit fires almost exclusively in the
        # refractory range, which is maximal contamination, not missing data
        return np.inf if d_rp > 0 else 0.0
    return float(d_rp / d_ref)


########## CROSS-UNIT COINCIDENCE ##########


def coincidence_matrix(spike_time_list, window_s=0.0005, chunk=2_000_000):
    """
    For every unit, the fraction of its spikes falling within +/- window_s of a
    spike from any OTHER unit in the list, and the identity of its worst partner.

    A single neuron cannot generate coincidences with other neurons whatever its
    firing regime, so an elevated value indicates either a cluster split across
    two entries, a common electrical artifact, or genuine synchrony. This is the
    one available metric that cannot be mimicked by intrinsic bursting.

    On a Utah array, neighbouring electrodes are ~400 um apart and sample
    largely independent populations, so the baseline expectation is low.

    All spikes are pooled and sorted once, then every spike's neighbours are
    gathered in one vectorised pass rather than a Python loop over spikes. The
    expansion is done in chunks so that peak memory stays bounded on dense
    arrays.

    Parameters
    ----------
    spike_time_list : list of 1-D arrays, one per unit, in seconds
    window_s        : coincidence half-width
    chunk           : maximum number of (spike, neighbour) pairs held at once

    Returns
    -------
    frac_coincident : array, fraction of spikes coincident with any other unit.
                      NOTE this is dominated by chance and scales with the
                      number of units and their rates. Use the ratios below.
    worst_partner   : array of unit indices, -1 where undefined
    frac_with_worst : array, fraction attributable to that single partner
    ratio_worst     : observed coincidences with the worst partner divided by
                      the number expected if the two trains were independent.
                      Rate- and count-corrected, so comparable across units and
                      recordings. 1.0 means chance; large values mean a split
                      cluster or a shared artifact.
    ratio_total     : same, pooled over all partners.
    """
    n_units = len(spike_time_list)
    frac = np.zeros(n_units)
    worst = np.full(n_units, -1, dtype=int)
    frac_worst = np.zeros(n_units)
    ratio_worst = np.zeros(n_units)
    ratio_total = np.zeros(n_units)
    if n_units < 2:
        return frac, worst, frac_worst, ratio_worst, ratio_total

    counts = np.array([len(s) for s in spike_time_list])
    if counts.sum() == 0:
        return frac, worst, frac_worst, ratio_worst, ratio_total

    all_t = np.concatenate([np.asarray(s, dtype=float)
                            for s in spike_time_list])
    all_u = np.concatenate([np.full(len(s), i, dtype=np.int32)
                            for i, s in enumerate(spike_time_list)])
    order = np.argsort(all_t, kind='mergesort')
    all_t = all_t[order]
    all_u = all_u[order]
    n_sp = len(all_t)

    lo = np.searchsorted(all_t, all_t - window_s, side='left')
    hi = np.searchsorted(all_t, all_t + window_s, side='right')
    n_per = (hi - lo).astype(np.int64)          # includes the spike itself

    pair_counts = np.zeros((n_units, n_units), dtype=np.int64)
    hit = np.zeros(n_units, dtype=np.int64)

    # walk the spikes in blocks whose expansion stays under `chunk` pairs
    cum = np.cumsum(n_per)
    start = 0
    while start < n_sp:
        stop = int(np.searchsorted(cum, cum[start - 1] if start else 0) )
        stop = int(np.searchsorted(cum, (cum[start - 1] if start else 0) + chunk))
        stop = max(stop, start + 1)
        stop = min(stop, n_sp)

        np_blk = n_per[start:stop]
        tot = int(np_blk.sum())
        if tot == 0:
            start = stop
            continue

        src = np.repeat(np.arange(start, stop, dtype=np.int64), np_blk)
        offs = np.arange(tot, dtype=np.int64) - np.repeat(
            np.cumsum(np_blk) - np_blk, np_blk)
        dst = np.repeat(lo[start:stop].astype(np.int64), np_blk) + offs

        u_src = all_u[src]
        u_dst = all_u[dst]
        keep = u_src != u_dst                   # drops self and same-unit pairs
        if keep.any():
            s_k, us_k, ud_k = src[keep], u_src[keep], u_dst[keep]
            # each spike counted once per distinct partner unit
            key = s_k.astype(np.int64) * n_units + ud_k
            uniq = np.unique(key)
            u_of = all_u[(uniq // n_units)]
            v_of = (uniq % n_units).astype(np.int64)
            np.add.at(pair_counts, (u_of, v_of), 1)
            # each spike counted once if it has any other-unit neighbour
            hit += np.bincount(all_u[np.unique(s_k)], minlength=n_units)
        start = stop

    # chance expectation for pair (i, j): N_i * 2w * rate_j
    T = float(all_t[-1] - all_t[0]) if n_sp > 1 else 0.0
    for i in range(n_units):
        if counts[i] == 0:
            continue
        frac[i] = hit[i] / counts[i]
        if T > 0:
            exp_i = counts[i] * 2.0 * window_s * counts / T   # vector over j
            exp_i[i] = np.inf                                  # exclude self
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios_i = np.where(exp_i > 0, pair_counts[i] / exp_i, 0.0)
            ratios_i = np.nan_to_num(ratios_i, nan=0.0, posinf=0.0)
            j = int(np.argmax(ratios_i))
            worst[i] = j if ratios_i[j] > 0 else -1
            ratio_worst[i] = float(ratios_i[j])
            frac_worst[i] = pair_counts[i, j] / counts[i]
            tot_exp = float(np.sum(exp_i[np.isfinite(exp_i)]))
            ratio_total[i] = float(pair_counts[i].sum() / tot_exp) if tot_exp > 0 else 0.0
    return frac, worst, frac_worst, ratio_worst, ratio_total


########## STABILITY ##########


def presence_ratio(spike_times_s, t_start_s, t_stop_s, n_bins=100):
    """Fraction of equal-width bins containing at least one spike."""
    st = np.asarray(spike_times_s, dtype=float)
    if len(st) == 0 or t_stop_s <= t_start_s:
        return np.nan
    edges = np.linspace(t_start_s, t_stop_s, n_bins + 1)
    return float(np.mean(np.histogram(st, bins=edges)[0] > 0))


def rate_stability(spike_times_s, t_start_s, t_stop_s, n_bins=50):
    """
    CV of the binned firing rate, and max/median ratio. Elevated values indicate
    drift or a unit appearing partway through the recording.
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


########## AMPLITUDE AND WAVEFORM METRICS ##########
#
# The spike trains carry per-spike waveforms on the peak channel, shape
# (n_spikes, n_samples). This makes available the one test that refractory
# statistics cannot perform: whether short-interval spike pairs show the
# amplitude attenuation characteristic of a genuine burst, or the flat profile
# expected when two different neurons have been merged into one cluster.
#
# Refractory violations detect merged clusters but cannot distinguish them from
# real bursting. Cross-unit coincidence detects oversplitting and shared
# artifacts but is blind to merges, since a merged cluster has no separate
# partner to be coincident with. Amplitude attenuation is the missing axis.


def spike_amplitudes(waveforms, trough_idx=None, search_halfwidth=2):
    """
    Per-spike amplitude at the trough, taken from the peak channel waveform.

    The trough sample is located on the mean waveform, then each spike is
    measured within +/- search_halfwidth samples of it to absorb alignment
    jitter. Returns absolute amplitudes, so larger means bigger spike.

    Parameters
    ----------
    waveforms : (n_spikes, n_samples) array
    trough_idx : sample index of the trough. If None, taken from the mean.

    Returns
    -------
    amps : (n_spikes,) array of absolute trough amplitudes
    trough_idx : the index used
    """
    w = np.asarray(waveforms, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2:
        return np.array([]), -1
    mean_w = np.nanmean(w, axis=0)
    if trough_idx is None:
        trough_idx = int(np.nanargmin(mean_w))
    lo = max(0, trough_idx - search_halfwidth)
    hi = min(w.shape[1], trough_idx + search_halfwidth + 1)
    amps = np.abs(np.nanmin(w[:, lo:hi], axis=1))
    return amps, int(trough_idx)


def burst_amplitude_attenuation(spike_times_s, amps, short_s=0.008,
                                long_lo_s=0.050, long_hi_s=0.200,
                                min_pairs=30):
    """
    THE DISCRIMINATING TEST.

    For consecutive spike pairs, the ratio of the second spike's amplitude to
    the first's, computed separately for short intervals (within a putative
    burst) and for long intervals (a within-unit control).

    A genuine neuron firing a burst shows sodium channel inactivation, so the
    second spike is systematically smaller: ratio_short < 1, typically 0.80 to
    0.95. Two different neurons merged into one cluster have no such
    relationship: ratio_short ~= 1, and importantly ratio_short ~= ratio_long.

    The long-interval control matters because it absorbs drift and any
    systematic asymmetry in the amplitude estimate. The interpretable quantity
    is the difference between the two.

    Returns dict:
        atten_short  : median amplitude ratio for short-interval pairs
        atten_long   : median ratio for long-interval pairs (should be ~1)
        atten_index  : atten_long - atten_short. Positive means real
                       attenuation, i.e. consistent with genuine bursting.
                       Near zero means no attenuation, i.e. consistent with a
                       merged cluster.
        n_short      : number of short-interval pairs contributing
        p_atten      : Mann-Whitney p comparing the two ratio distributions
    """
    st = np.asarray(spike_times_s, dtype=float)
    a = np.asarray(amps, dtype=float)
    out = {'atten_short': np.nan, 'atten_long': np.nan, 'atten_index': np.nan,
           'n_short': 0, 'p_atten': np.nan}
    if len(st) != len(a) or len(st) < 3:
        return out

    isi = np.diff(st)
    # a[:-1] can legitimately be zero (a spike amplitude at the detection
    # floor), which would otherwise raise a RuntimeWarning here even though
    # the result is correctly discarded by the isfinite/positive check right
    # below - silence just this division, not warnings in general.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = a[1:] / a[:-1]
    good = np.isfinite(ratio) & (a[:-1] > 0)

    m_short = good & (isi < short_s)
    m_long = good & (isi >= long_lo_s) & (isi < long_hi_s)

    out['n_short'] = int(m_short.sum())
    if m_short.sum() < min_pairs or m_long.sum() < min_pairs:
        return out

    rs, rl = ratio[m_short], ratio[m_long]
    out['atten_short'] = float(np.median(rs))
    out['atten_long'] = float(np.median(rl))
    out['atten_index'] = float(np.median(rl) - np.median(rs))
    try:
        from scipy.stats import mannwhitneyu
        out['p_atten'] = float(mannwhitneyu(rs, rl, alternative='two-sided').pvalue)
    except Exception:
        pass
    return out


def amplitude_distribution_metrics(amps, min_spikes=100, n_boot_bic=1):
    """
    Shape of the per-spike amplitude distribution.

    A cleanly isolated unit has a roughly unimodal, slightly right-skewed
    amplitude distribution. A merged cluster contains spikes of systematically
    different sizes and is broader or bimodal, which catches merges that
    refractory violations miss when the constituent neurons fire slowly.

    Returns dict:
        amp_median, amp_cv        : location and relative spread
        amp_bimodal_bic_gain      : BIC(1 component) - BIC(2 components).
                                    Positive means two components fit better.
        amp_mode_separation       : distance between the two fitted means,
                                    in units of the pooled SD. Only meaningful
                                    when the BIC gain is positive.
        amp_cutoff                : fraction of the distribution estimated to
                                    fall below the detection threshold, from
                                    the truncation of the lower tail.
    """
    a = np.asarray(amps, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    out = {'amp_median': np.nan, 'amp_cv': np.nan,
           'amp_bimodal_bic_gain': np.nan, 'amp_mode_separation': np.nan,
           'amp_cutoff': np.nan}
    if len(a) < min_spikes:
        return out

    out['amp_median'] = float(np.median(a))
    m = float(np.mean(a))
    out['amp_cv'] = float(np.std(a) / m) if m > 0 else np.nan

    try:
        from sklearn.mixture import GaussianMixture
        x = a.reshape(-1, 1)
        # subsample for speed; amplitude statistics converge quickly
        if len(x) > 20000:
            rng = np.random.default_rng(0)
            x = x[rng.choice(len(x), 20000, replace=False)]
        g1 = GaussianMixture(1, random_state=0).fit(x)
        g2 = GaussianMixture(2, random_state=0, n_init=3).fit(x)
        out['amp_bimodal_bic_gain'] = float(g1.bic(x) - g2.bic(x))
        mu = np.sort(g2.means_.ravel())
        sd = np.sqrt(np.mean(g2.covariances_.ravel()))
        out['amp_mode_separation'] = float((mu[1] - mu[0]) / sd) if sd > 0 else np.nan
    except Exception:
        pass

    # amplitude cutoff: compare the observed lower tail against a Gaussian
    # fitted to the upper half, which the detection threshold does not truncate
    try:
        med = np.median(a)
        upper = a[a >= med]
        sd_up = np.std(upper - med) if len(upper) > 10 else np.nan
        if np.isfinite(sd_up) and sd_up > 0:
            from scipy.stats import norm
            expected_below = norm.cdf(a.min(), loc=med, scale=sd_up)
            out['amp_cutoff'] = float(np.clip(expected_below, 0, 1))
    except Exception:
        pass
    return out


def waveform_consistency(waveforms, amps=None, max_spikes=5000, seed=0):
    """
    Correlation of each spike's waveform with the unit's mean template.

    A cleanly isolated unit gives a tight distribution near 1. A merged cluster
    contains two waveform shapes, so the distribution is broad or bimodal.

    Returns dict:
        wf_corr_median : median correlation to the template
        wf_corr_p05    : 5th percentile, sensitive to a minority second shape
        wf_snr         : mean template peak-to-peak divided by the mean
                         residual SD, a direct isolation measure
    """
    w = np.asarray(waveforms, dtype=float)
    out = {'wf_corr_median': np.nan, 'wf_corr_p05': np.nan, 'wf_snr': np.nan}
    if w.ndim != 2 or w.shape[0] < 20:
        return out

    if w.shape[0] > max_spikes:
        rng = np.random.default_rng(seed)
        w = w[rng.choice(w.shape[0], max_spikes, replace=False)]

    tmpl = np.nanmean(w, axis=0)
    tmpl_c = tmpl - tmpl.mean()
    denom_t = np.sqrt(np.sum(tmpl_c ** 2))
    if denom_t <= 0:
        return out

    wc = w - w.mean(axis=1, keepdims=True)
    denom_w = np.sqrt(np.sum(wc ** 2, axis=1))
    ok = denom_w > 0
    corr = np.full(w.shape[0], np.nan)
    corr[ok] = (wc[ok] @ tmpl_c) / (denom_w[ok] * denom_t)

    c = corr[np.isfinite(corr)]
    if len(c):
        out['wf_corr_median'] = float(np.median(c))
        out['wf_corr_p05'] = float(np.percentile(c, 5))

    resid = w - tmpl
    resid_sd = float(np.nanmean(np.nanstd(resid, axis=0)))
    ptp = float(np.nanmax(tmpl) - np.nanmin(tmpl))
    if resid_sd > 0:
        out['wf_snr'] = float(ptp / resid_sd)
    return out


def waveform_shape_heterogeneity(waveforms, max_spikes=4000, seed=0,
                                 min_spikes=200):
    """
    Heterogeneity of waveform SHAPE within a cluster, independent of amplitude.

    Width is a cell-type property and does not vary with electrode distance, so
    a cluster containing spikes of different shapes almost certainly contains
    more than one neuron. Amplitude varies for benign reasons (distance, drift,
    burst attenuation), which is why shape is the stronger evidence.

    Every spike is amplitude-normalised before any shape measure is taken, so
    nothing here responds to amplitude alone.

    VALIDATED ON SYNTHETIC DATA:
        single narrow unit                 pc1_modesep  1.60
        single wide unit                                1.51
        single unit, wide amplitude spread              1.33
        merged narrow + wide                           18.70
        merged narrow + medium                          9.67
        merged same width, different amplitude          1.78
    The last case is correctly low: that is an amplitude split, caught by
    amp_mode_separation instead. The two measures are complementary.

    wf_width_cv is reported for description only. It is confounded with
    amplitude spread (a single unit with variable amplitude scores the same as
    a genuine narrow-plus-medium merge), so it must not be thresholded.
    A Gaussian mixture on the width distribution was tested and does not
    separate merged from single units at all; it is not computed.

    Returns dict:
        wf_shape_pc1_modesep : separation of two fitted modes on the first
                               principal component of the amplitude-normalised
                               waveforms, in pooled SD. THE USABLE MEASURE.
        wf_shape_pc1_bic     : BIC(1) - BIC(2) on the same component
        wf_width_cv          : CV of per-spike half-width. DESCRIPTIVE ONLY.
        wf_resid_ratio       : SD of the residual to the mean normalised shape,
                               over the shape amplitude
    """
    w = np.asarray(waveforms, dtype=float)
    out = {'wf_shape_pc1_modesep': np.nan, 'wf_shape_pc1_bic': np.nan,
           'wf_width_cv': np.nan, 'wf_resid_ratio': np.nan}
    if w.ndim != 2 or w.shape[0] < min_spikes:
        return out

    if w.shape[0] > max_spikes:
        rng = np.random.default_rng(seed)
        w = w[rng.choice(w.shape[0], max_spikes, replace=False)]

    troughs = np.nanmin(w, axis=1)
    ok = np.isfinite(troughs) & (troughs < 0)
    if ok.sum() < min_spikes:
        return out
    w = w[ok]
    wn = w / np.abs(troughs[ok])[:, None]      # trough now at -1 for every spike

    # half-width: samples below half the trough. Vectorised.
    widths = np.sum(wn < -0.5, axis=1).astype(float)
    widths = widths[widths > 0]
    if len(widths) >= min_spikes:
        m = float(np.mean(widths))
        out['wf_width_cv'] = float(np.std(widths) / m) if m > 0 else np.nan

    # shape PCA, then a two-component mixture on the first component
    try:
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        pc1 = PCA(n_components=1).fit_transform(wn)
        if np.std(pc1) > 0:
            g1 = GaussianMixture(1, random_state=0).fit(pc1)
            g2 = GaussianMixture(2, random_state=0, n_init=2).fit(pc1)
            out['wf_shape_pc1_bic'] = float(g1.bic(pc1) - g2.bic(pc1))
            mu = np.sort(g2.means_.ravel())
            sd = np.sqrt(np.mean(g2.covariances_.ravel()))
            out['wf_shape_pc1_modesep'] = float((mu[1] - mu[0]) / sd) if sd > 0 else np.nan
    except Exception:
        pass

    mean_shape = np.nanmean(wn, axis=0)
    resid_sd = float(np.nanmean(np.nanstd(wn - mean_shape, axis=0)))
    amp = float(np.nanmax(mean_shape) - np.nanmin(mean_shape))
    if amp > 0:
        out['wf_resid_ratio'] = float(resid_sd / amp)
    return out
