# Methods: unit filtering metrics

This document describes the metrics computed by `SUA_filter_metrics_pkl_create.py`
and the reasoning behind each. It is written so the Methods section of the
manuscript can be assembled from it, and so the parameter values quoted here can
be checked against the configuration dictionary in the script.

---

## 1. Design

Filtering is done in two stages. The first computes every candidate metric for
every sorted unit and saves them; nothing is removed. The second applies
thresholds chosen from those distributions and writes new files. Separating the
two means a threshold can be revised without recomputing anything, and it means
the choice of threshold is made against measured distributions rather than
assumed ones.

The input is the unfiltered spike files, which contain every sorted unit before
the sorting pipeline's own criteria are applied. These sit in the `spikes_KS4` subdirectory of the recording folder, named
`{recording}_Array{N}_spikes_KS4_unfiltered.nix`, while the earlier filtering
carries no suffix and is named `{recording}_Array{N}_spikes_KS4.nix`. Working from these rather than
from the already-filtered files matters because the pipeline's thresholds
truncate several distributions at the cut, so a metric it acted on cannot be
recalibrated from its output.

---

## 2. Why several metrics are recomputed rather than read

The spike sorting was performed with Kilosort 4 on groups of four channels at a
time, rather than on all sixty-four channels of an array together. Several
consequences follow, and they determine which stored quality metrics are usable.

Metrics that compare units to one another operate only within a four-channel
group. The synchrony measures written by SpikeInterface are therefore blind to a
unit duplicated across a group boundary, and on this dataset they are close to
zero for every unit. The drift measures require a spatial extent that four
contacts do not provide and are identically zero.

The sliding refractory period metric returns a missing value when it cannot
establish a contamination level at the required confidence. On this dataset it
was missing for a substantial minority of units, and the stored value does not
distinguish that case from a unit with too few spikes to assess.

A neuron detected on two channels that fall in different groups is sorted twice
and appears as two units with nearly identical spike times. This form of
oversplitting is invisible to any metric computed within a group.

The metrics described below are computed across the whole array from the spike
times and the per-spike waveforms, and are defined for every unit with enough
spikes. Where a stored annotation is usable it is carried through unchanged
rather than recomputed; those are listed in section 4.

A further practical point: the sorting script replaces missing values with the
string "NaN" before writing, because the NIX format cannot store a numeric NaN.
The reader converts these back. A numeric comparison against an unconverted
annotation raises an error rather than evaluating false, so this conversion is
not optional.

---

## 3. Computed metrics

### 3.1 Refractory violations

Two quantities are derived from the interspike interval distribution, both
computed over a grid of candidate refractory periods from 1 to 10 ms in 0.25 ms
steps, with a censored period of 0.2333 ms corresponding to the sorter's dead
time of seven samples at 30 kHz.

At each candidate refractory period, the number of observed intervals shorter
than that period is divided by the number expected from a homogeneous Poisson
process of the same rate and duration. **`viol_ratio_min`** is the smallest such
ratio across the grid: a unit that is clean at any plausible refractory period
receives a low value. The scale is interpretable, with zero meaning no
violations and one meaning violations at chance level, that is, a spike train
carrying no refractory structure at all.

This formulation was chosen in preference to solving for a contamination
fraction, as in the Hill relation, because that quantity becomes unidentifiable
above roughly fifty percent contamination and returns no value there. A ratio is
always defined and is unbounded above, so heavily contaminated units receive a
large number rather than a missing one.

**`viol_ratio_slope`** is the ratio at 4 ms minus the ratio at 1 ms. A neuron
accumulates violations only beyond its refractory period, giving a positive
slope, whereas a train composed of several neurons is flat. This is largely
independent of the absolute level and captures the shape of the profile rather
than its magnitude.

Validation on synthetic spike trains: clean units at 3, 20, 42 and 100 Hz all
returned a ratio of zero, confirming rate independence; contamination injected
at 5, 10, 20 and 50 percent returned 0.13, 0.19, 0.36 and 0.72; two merged units
returned 0.46 and four returned 0.74; a Poisson train returned 0.98. The slope
was approximately 0.55 for clean units and 0.08 for four merged units.

### 3.2 Cross-unit coincidence

For every pair of units on an array, the number of spikes falling within
0.5 ms of a spike from the other unit is counted and divided by the number
expected if the two trains were independent, which for units i and j is the
spike count of i multiplied by the coincidence window and the rate of j.
**`coinc_ratio_worst`** is the largest such ratio across all partners on the
array, and the partner achieving it is recorded.

The normalisation is necessary because the raw coincidence fraction scales with
the number of units on the array and with their firing rates, so a fast unit on
a densely sampled array appears coincident with everything.

This metric detects oversplit clusters and shared electrical artifacts. A single
neuron cannot produce coincidences with other neurons whatever its firing
pattern, so an elevated value is strong evidence. It is specifically the failure
mode that the four-channel sorting introduces and that the stored synchrony
metrics cannot see.

It does not detect merged clusters, where two neurons are combined into one
unit: there is no separate partner to be coincident with. That case is covered
by the refractory metrics, which is why both are retained.

Validation on synthetic data: independent units returned approximately 1.1, two
units sharing an artifact returned 22.9, and a single train split in two
returned 41.7.

A unit whose most coincident partner also has it as its own most coincident
partner is flagged as a **mutual pair**. These are almost always one neuron
split in two, and the appropriate remedy is to retain one of the pair rather
than discard both.

### 3.3 Waveform shape heterogeneity

This metric is computed and stored but is not applied as a filtering criterion.
It is retained as a diagnostic, and so that the criterion could be introduced
later without recomputation.

Every spike waveform is divided by its own trough amplitude, so that the measure
responds to shape and not to amplitude. The first principal component of the
normalised waveforms is taken, and a two-component Gaussian mixture is fitted to
it. **`wf_shape_pc1_modesep`** is the separation between the two fitted means
expressed in pooled standard deviations.

Waveform width is a property of the cell type and does not vary with the
distance from the electrode, so a cluster containing more than one spike shape
almost certainly contains more than one neuron. Amplitude, by contrast, varies
for benign reasons including electrode distance, drift, and attenuation within a
burst, which is why the normalisation is applied first.

Validation on synthetic data: a single narrow unit returned 1.60, a single wide
unit 1.51, and a single unit with a wide amplitude spread 1.33; a merged
narrow-plus-wide cluster returned 18.70 and a merged narrow-plus-medium cluster
9.67. A merge of two units with the same width but different amplitudes returned
1.78, correctly low, since that is an amplitude split rather than a shape split
and is captured by the amplitude mode separation instead.

A coefficient of variation of the per-spike half-width is also computed but is
reported as descriptive only: it is confounded with amplitude spread, since a
single unit with variable amplitude scores the same as a genuine merge of
narrow and medium units. A Gaussian mixture on the width distribution was tested
and did not separate merged from single units at all.

### 3.4 Waveform consistency and signal-to-noise

Each spike's waveform is correlated against the unit's mean template.
**`wf_corr_median`** and the fifth percentile of that distribution summarise how
consistent the shapes are; a minority second shape depresses the percentile
while leaving the median intact. **`wf_snr_computed`** is the peak-to-peak
amplitude of the mean template divided by the mean standard deviation of the
residual after subtracting it.

The signal-to-noise ratio stored by the sorting pipeline is also carried
through, so the two can be compared.

### 3.5 Amplitude distribution

The per-spike trough amplitude is located on the mean waveform and measured
within two samples of that position on each spike, to absorb alignment jitter.
From the resulting distribution, the coefficient of variation, the separation of
two fitted mixture components, and an estimate of the fraction truncated by the
detection threshold are computed. The mode separation catches merges of units
that differ in amplitude but not in shape, complementing section 3.3.

### 3.6 Amplitude attenuation within bursts

For consecutive spike pairs, the ratio of the second amplitude to the first is
computed separately for short intervals, below 8 ms, and for long intervals
between 50 and 200 ms. **`atten_index`** is the difference between the two
medians.

A neuron firing a burst shows sodium channel inactivation, so the second spike
is systematically smaller and the index is positive. Merging dilutes this,
because short-interval pairs increasingly fall between different neurons rather
than within one burst. This is the only available measure that can distinguish
an elevated refractory violation rate caused by a merged cluster from one caused
by a genuinely bursting neuron, and it is reported for interpretation rather
than used as a threshold, because the dilution is graded rather than binary.

### 3.7 Stability

**`presence_ratio_computed`** is the fraction of one hundred equal-width bins
spanning the recording that contain at least one spike, and identifies units
lost to drift or appearing part-way through. The coefficient of variation of the
binned firing rate and the ratio of its maximum to its median are also recorded.

### 3.8 Firing rate

Computed directly from the spike count and the recording duration, and reported
alongside the annotated value so the two can be compared. A discrepancy would
indicate that the annotation refers to a different time base than the
spiketrain, which would affect every rate-dependent metric.

---

## 4. Metrics carried through from the sorting

These are read from the file rather than recomputed, and are prefixed `ann_`.

**`waveform_SNR`**, **`presence_ratio`**, **`amplitude_cutoff`**,
**`amplitude_cv_median`**, **`sd_ratio`**, and the refractory measures
**`isi_violations_ratio`**, **`isi_violations_count`**, **`rp_contamination`**,
**`rp_violations`** and **`sliding_rp_violation`**, all computed by
SpikeInterface's quality metrics module.

**`line_noise_50Hz`** and **`line_noise_60Hz`**, computed by the sorting
pipeline. The autocorrelogram of the unit is computed in 2 ms bins over one
hundred lags and correlated with a version of itself shifted by one period of
the mains frequency, 20 ms for 50 Hz and 16.7 ms for 60 Hz. A unit locked to
mains has a periodic autocorrelogram and correlates strongly with its own shift.
Note that this quantity is also elevated for any unit with a smoothly varying
autocorrelogram, since a smooth curve correlates with a small shift of itself
regardless of periodicity, so a moderate value does not necessarily indicate
line locking.

**`KSLabel`**, Kilosort's own good or multi-unit call, retained as an
independent reference rather than as a criterion.

---

## 5. Threshold selection

Thresholds are chosen in a separate notebook, from the distributions of the
metrics above. Three considerations guide the choice.

Where a metric is bimodal, the threshold is placed at the trough between the two
components, in which case the exact value has little influence. Where it is
unimodal, the choice is a convention and is justified by the resulting yield.

Every candidate threshold is assessed for rate confounding, by comparing the
median firing rate of the surviving units against the unfiltered median. A large
shift indicates the criterion is acting as a firing rate filter, which matters
because contamination metrics are intrinsically rate dependent and because the
unit classes under later comparison differ in firing rate.

Criteria correlating strongly with one another are treated as one criterion
rather than several, so that a rule counting how many criteria a unit passes
does not over-weight whatever they share.

Line noise and firing rate are applied as hard criteria, which no other
criterion can outvote: a unit locked to mains is not a partially contaminated
neuron but an unusable one. The remaining criteria are combined by counting how
many a unit passes, which is more robust than a conjunction, since a conjunction
is dominated by whichever criterion happens to be strictest.

Missing values are treated as failing by default. This is recorded explicitly
because it is consequential: the sliding refractory metric stored by the sorting
was missing for a substantial fraction of units, and treating that as a failure
removes them on a basis different from contamination.

---

## 6. On the interspike interval criterion

The sorting pipeline applied one refractory criterion, requiring
`isi_violations_ratio` below 0.9. That quantity is normalised so that 1.0
corresponds to violations at chance level, so a threshold at 0.9 excludes only
units that are close to carrying no refractory structure at all. The median
value on this dataset was 0.44, well below the cut.

The remaining refractory metrics computed by SpikeInterface,
`rp_contamination`, `rp_violations` and `sliding_rp_violation`, were written to
every file and used by nothing.

The raw fraction of intervals below the refractory period is also recorded here,
but is not used as a criterion: the expected number of violations scales with
the square of the firing rate, so a fixed threshold on it discriminates against
fast-firing units. `viol_ratio_min` addresses the same failure mode with that
dependence removed, which is why it is preferred.

---

## 7. Parameter summary

| Parameter | Value |
|---|---|
| Censored period (sorter dead time) | 0.2333 ms |
| Refractory period grid | 1 to 10 ms, 0.25 ms steps |
| Violation profile slope | ratio at 4 ms minus ratio at 1 ms |
| Coincidence window | ±0.5 ms |
| Mutual pair threshold | coincidence ratio above 10 |
| Waveforms subsampled for shape | 4000 spikes |
| Minimum spikes for shape metrics | 200 |
| Minimum spikes for refractory metrics | 100 |
| Burst interval for attenuation | 8 ms |
| Long-interval control for attenuation | 50 to 200 ms |
| Presence ratio bins | 100 |
| Rate stability bins | 50 |

---

## 8. Limitations

Adjacent-channel duplication is detectable but not automatically resolvable. A
mutual coincidence pair identifies which two entries are the same neuron, but
deciding which to retain, or whether to merge them, is a separate operation.

The refractory metrics cannot distinguish a merged cluster from a neuron firing
genuine short-interval doublets. The amplitude attenuation index addresses this
but is graded rather than decisive.

None of these metrics can detect a unit that is well isolated but not a single
neuron in the intended sense, for instance a fibre of passage or a
non-somatic recording. Waveform shape is used elsewhere in the analysis to
identify positive-going and triphasic waveforms for that reason.

The four-channel sorting is the origin of most of the problems these metrics
address. Re-sorting each array as a whole would remove the duplication at
source and would make the standard quality metrics usable, at the cost of a
complete reprocessing run.
