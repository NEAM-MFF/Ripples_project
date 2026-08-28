# Pipeline run order

26 small submit scripts, one job per file, numbered in the order you can
submit them. Each file has exactly one active `python` line and a header comment stating what it needs to
have finished first. Where a script's `--array` count depends on how many
recordings exist, the script itself prints the exact count on first run —
double-check the number in the file still matches before you `sbatch` it.

## The two things worth knowing before you start

**One codebase.** All 26 scripts, including the filtering stage (01–07:
compute QC metrics on the raw sort, pick thresholds, write the filtered
spike files) and everything downstream that *consumes* `spikes_filtered/`
(stages 08 onward), run from the same directory, currently (set as you wish):
```
/CSNG/studekat/ripple_paper_clean_copy/code_new_filter
```
Don't run any of stages 08+ before the filtering stage (01–07) has produced
`spikes_filtered/` for the relevant recordings.

**One shared threshold file.** RS, NATIM and HUMAN all read the same
`dataframes/filter_metrics_chosen/thresholds.pkl` (step 04, manual). There
is no separate human threshold file

## Run order and dependencies

```
STAGE 0 — build the filter (must finish first)
  01 filter_metrics_rs      ─┐
  02 filter_metrics_natim    ├─► 04 MANUAL: run F22 notebook, writes thresholds.pkl
  03 filter_metrics_human   ─┘        │
                                       ▼
  05 write_filtered_rs     (needs 01 + 04)
  06 write_filtered_natim  (needs 02 + 04)
  07 write_filtered_human  (needs 03 + 04)

STAGE 1 — RS analysis (needs 05)
  08 RS_prop            ─┬─► 12 RS_shuffle_phases (needs 08)
  09 RS_detect_ripples   │
                         ├─► 13 RS_trigg_spectra_create (needs 09)
                         │        └─► 14 RS_trigg_spectra_merge (needs ALL of 13 done)
                         ├─► 15 RS_trigg_df           (needs 08 + 09)
                         └─► 16 RS_trigg_phase_align  (needs 08 + 09)
  10 RS_delta_env       (needs 05 only, independent of 08/09)
  11 RS_hypnogram       (needs 05 only, independent of 08/09)

STAGE 2 — NATIM analysis (needs 06)
  17 NATIM_prop          ─┬─► 19 NATIM_trigg_df           (needs 17 + 18)
  18 NATIM_detect_ripples ─┴─► 20 NATIM_trigg_phase_align  (needs 17 + 18)
                                    └─► 21 build_NATIM_fig5_cache (needs 17 + 20)
  17 + 18 ──► 22 NATIM_spectra_testing

STAGE 3 — graph (needs BOTH pipelines' unit tables)
  23 graph_SUA_preprocess   (needs 08 AND 17)

STAGE 4 — HUMAN analysis (needs 07)
  24 HUMAN_detect_ripples ─┬─► 26 HUMAN_trigg_phase_align (needs 24 + 25)
  25 HUMAN_prop           ─┘
```

Within a stage, anything not chained above by an arrow can be submitted at
the same time (SLURM will just queue them). `10_RS_delta_env` and
`11_RS_hypnogram`, for instance, only need `05` — you don't have to wait for
`08`/`09` first.

## Chaining jobs automatically (optional)

Instead of watching `squeue` and submitting the next file by hand, you can
chain jobs so SLURM only starts the next one after the previous succeeds:

```bash
jid1=$(sbatch --parsable 01_filter_metrics_rs.sh)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 05_write_filtered_rs.sh)
```

For an array job, `--dependency=afterok:$jid1` waits for *every* task in the
array to finish successfully before the dependent job starts — exactly what
you want before, e.g., `14_RS_trigg_spectra_merge.sh`.

## Chaining jobs automatically (optional)
Python packaged used can be found in requirements.txt file.
