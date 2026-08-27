# Methods: writing the filtered spike files

This document describes `SUA_write_filtered_nix.py`, the second stage of the
two-stage filtering scheme. The first stage, which computes the metrics, is
documented in `METHODS_filter_metrics.md`.

---

## 1. Purpose and placement

The first stage computes every candidate metric for every sorted unit and saves
them without removing anything. This stage applies the chosen thresholds and
writes new files containing only the units that pass.

Separating the two means a threshold can be revised and the files regenerated
without recomputing any metric, and it means the thresholds are chosen against
measured distributions rather than assumed ones. It also means the decision is
auditable: the metric values that produced it are stored alongside the units
that survived it.

---

## 2. Input and output

The input for each array is the unfiltered spike file, which contains every
sorted unit before the sorting pipeline's own criteria were applied, together
with the metrics computed for that recording and the thresholds saved by the
selection notebook. The spike files sit in the `spikes_KS4` subdirectory of the recording folder,
with every sorted unit in `{recording}_Array{N}_spikes_KS4_unfiltered.nix` and
the earlier filtering in `{recording}_Array{N}_spikes_KS4.nix`.

The output is one file per array, written to a `spikes_filtered` directory
placed beside the sorted spike files within the same recording folder, and named
`{recording}_Array{N}_spikes_KS4_filtered.nix` so that the three sets can be
told apart by filename alone.
The recording folder layout is otherwise unchanged, so the existing loaders
reach the new files by substituting one path element. The two recording
paradigms use different naming conventions and both are handled: resting state
recordings are named by animal, paradigm and date, while the natural image
recordings follow the TVSD convention.

Files are written only for arrays where an input file exists. An array whose
units have no entry in the metrics is reported rather than silently skipped,
since that indicates the metric computation did not cover the whole recording.

---

## 3. What is retained

**Spike times** are copied unchanged, together with the start and stop times,
sampling rate and units of the original spiketrain.

**Ordering** is preserved. The surviving units appear in the same relative order
as in the unfiltered file, and each carries `train_order_original`, its index in
that file, so any unit can be traced back to its source. The original unit count
is recorded both per unit and on the block.

**All annotations** from the unfiltered file are inherited unchanged, including
the electrode identifier, the cortical area, the sorter's own label, and every
quality metric written by the sorting pipeline.

**The computed metrics** are added with an `flt_` prefix, so that a file carries
the evidence for its own contents. These include the refractory violation ratio
and profile slope, the cross-unit coincidence ratio and the identity of the most
coincident partner, the recomputed signal-to-noise ratio and presence ratio, the
amplitude statistics, and the within-burst amplitude attenuation.

The waveform shape heterogeneity is stored among these but is **not** used as a
criterion. It is retained as a diagnostic, and because storing it means the
criterion could be added later without recomputing anything.

**The decision and the rule that produced it** are recorded per unit: one
boolean per criterion, and the threshold and direction that criterion applied,
as a `th_` annotation. The full criterion set is also written to the block. A
file therefore records the rule that generated it and does not depend on an
external record to be interpreted.

---

## 4. Waveforms

The per-spike waveforms are **not** written. In their place each unit carries
two vectors.

**`avg_wf`** is the mean waveform across all of that unit's spikes, in the units
of the original recording.

**`avg_wf_zscored`** is that same averaged trace, z-scored. The mean over
spikes is taken first, and the resulting single waveform is then standardised as
a whole, by subtracting its mean and dividing by its standard deviation, so it
has zero mean and unit standard deviation. Individual spikes are never z-scored,
and nothing is normalised per sample across spikes. This is the form used by the
waveform classification and the waveform figures. The convention should be
confirmed against `aux_add_zscored_avg_waveform` before the two are treated as
interchangeable.

The number of spikes contributing to the average and the number of samples are
recorded alongside, so a mean computed from few spikes can be identified.

The saving is substantial. Per-spike waveforms occupy roughly 1.5 GB per array,
while two vectors of ninety samples per unit are negligible; in a test on
representative data the output was under two percent of the input size. Nothing
in the downstream analysis uses per-spike waveforms, which are needed only for
the amplitude and shape metrics computed in the first stage and already
summarised in the annotations.

---

## 5. Missing values and the NIX format

The NIX format cannot store a numeric NaN. The sorting pipeline handles this by
writing the string "NaN" in place of a missing value, and the same convention is
followed here so that the filtered files are consistent with the rest of the
dataset.

This has a consequence that must be respected when reading: a direct numeric
comparison against an unconverted annotation raises an error rather than
evaluating false. A reader that converts the string back to a numeric NaN is
provided at the end of the script and should be used wherever annotations are
compared.

Arrays containing non-finite values, which can arise in a mean waveform computed
from very few spikes, are written with those entries replaced by zero. The
accompanying spike count allows such cases to be identified.

---

## 6. Mutual coincidence pairs

A pair of units where each is the other's most coincident partner is almost
always a single neuron split into two clusters by the four-channel sorting. A
threshold on the coincidence ratio removes both members of such a pair, which
discards a real neuron rather than resolving the duplication.

The script can instead resolve the pair, retaining the member with more spikes
and dropping the other, with both annotated so the decision is visible. This is
disabled by default, so that the script reproduces exactly the selection
reported by the threshold notebook, and should be enabled deliberately. Whether
it is enabled is recorded on the block.

---

## 7. Reproducibility

The thresholds are read from the file written by the selection notebook rather
than duplicated in the script, so the two cannot diverge. Every threshold, the
treatment of missing values, and whether mutual pairs were resolved are recorded
on each output block, together with the date of filtering and the original and
retained unit counts.

Each recording is processed as an independent task, and a summary of the units
retained per array is written for each.

---

## 8. Validation

The writing was tested end to end on synthetic files. Spike times and counts
were unchanged, the relative order of the surviving units was preserved and
matched the recorded original indices, the per-spike waveforms were absent from
the output, the stored mean waveform matched a recomputation exactly, the
z-scored waveform had zero mean and unit standard deviation, inherited
annotations round-tripped unchanged, and the recorded thresholds were readable
from the output. The output was 1.75 percent of the input size.

---

## 9. Limitations

The filtering acts on units, not on the sorting. A cluster split across a
four-channel boundary is detected and can be resolved by discarding one member,
but the two are not merged, so the spikes assigned to the discarded member are
lost rather than recovered.

Units removed for having too few spikes are removed by the firing rate
criterion, but the refractory and shape metrics are also undefined for them.
Applying the rate criterion first means such units are excluded once for a
single stated reason rather than repeatedly by metrics that could not be
computed.

The filtered files record the criteria that produced them, but not the units
that were removed. The metrics from the first stage remain the record of what
was discarded and should be retained alongside.
