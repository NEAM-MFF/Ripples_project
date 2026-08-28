# Cell-type-specific coupling of single-unit spikes and cortical ripples in macaque and human V1

Analysis of high-frequency ripple-band oscillations in macaque and human V1, their relationship to single-unit spiking, and the mechanisms underlying their generation.

Preprint describing the project: https://doi.org/10.64898/2026.07.02.736041

Dataset used in the project: https://doi.org/10.12751/g-node.xg19b5


## Repository layout

```
code/
├── functions_analysis.py       # shared analysis functions used across the pipeline
├── params_analysis.yml         # dataset paths, recording dates, array layouts, classification thresholds
├── requirements.txt            # exact package versions used to run this pipeline
├── submit_pipeline/            # SLURM submit scripts, numbered in run order (see below)
│   └── 00_HOW_TO_RUN.md         # full run order, dependencies, and job-chaining instructions
├── panel_notebooks/             # one notebook per figure/supplementary figure, produces the paper's plots
└── *.py                         # preprocessing/analysis scripts, run via submit_pipeline/
```

## Running the pipeline

All preprocessing is run and parallelized as SLURM array jobs via the numbered submit scripts in `submit_pipeline/`. Start with `submit_pipeline/00_HOW_TO_RUN.md`, which lays out the full dependency graph — in short:

1. **Build the SNR-based unit filter** (`01`–`07`): compute quality-control metrics for every sorted unit, choose thresholds (one shared threshold file for RS, NATIM, and human data), and write the filtered spike files.
2. **RS (resting-state) analysis** (`08`–`16`): unit properties, ripple detection, ripple-triggered spectra, and phase-alignment for the macaque resting-state recordings.
3. **NATIM (natural-image) analysis** (`17`–`22`): the same pipeline for the natural-image-viewing recordings.
4. **Spatial clustering graph** (`23`): pools RS and NATIM unit tables to build the single-unit-type clustering graph (Figure 5A).
5. **Human analysis** (`24`–`26`): ripple detection and phase-alignment for the blind human V1 recordings.

Each script prints the exact `--array` range it needs on first run — confirm the number in the file still matches before submitting. Jobs can be chained automatically via `sbatch --dependency=afterok:<jobid>` instead of submitting by hand; see `00_HOW_TO_RUN.md` for the exact commands.

The environment used to run this pipeline is recorded in `requirements.txt` (generated via `pip freeze`).

## Figures

All plots are produced by the notebooks in `panel_notebooks/`, named by the figure (`F`) and supplementary figure (`SF`) they generate. Run the relevant preprocessing scripts (above) before opening a notebook — each one reads from the dataframes those scripts produce.

| Notebook | Figures produced |
|---|---|
| `F1_SF1_hypnograms.ipynb` | Figure 1, Supplementary Figure 1 |
| `F1_SF2_ripples.ipynb` | Figure 1, Supplementary Figure 2 |
| `F2_cell_classif.ipynb` | Figure 2 |
| `F3_phase_pref.ipynb` | Figure 3 |
| `F3_SF3_correlation_examples.ipynb` | Figure 3, Supplementary Figure 3 |
| `SF3_phase_pref_EC_EO.ipynb` | Supplementary Figure 3 |
| `SF3_testing_RS_EC_EO.ipynb` | Supplementary Figure 3 (statistics) |
| `SF4_phase_pref_monkeyL.ipynb` / `monkeyN` / `monkeyF` | Supplementary Figure 4 (per animal) |
| `F4_NATIM_cell_classif.ipynb` | Figure 4 |
| `F4_NATIM_phase_pref.ipynb` | Figure 4 |
| `SF5_NATIM_ripples.ipynb` | Supplementary Figure 5 |
| `SF5_testing_NATIM_RS.ipynb` | Supplementary Figure 5 (statistics) |
| `SF6_cort_space.ipynb` | Supplementary Figure 6 |
| `F5_neigh_types_graph.ipynb` | Figure 5A (spatial clustering graph) |
| `F5_SF7_cliques_distribution.ipynb` | Figure 5B, Supplementary Figure 7 |
| `F5_spectral_comparison_RS.ipynb` | Figure 5C, D |
| `F5_model.ipynb` | Figure 5F (LIF population model) |
| `SF8_spectral_comparison_NATIM.ipynb` | Supplementary Figure 8 |
| `F6_HUMAN_cell_classif.ipynb` | Figure 6 |
| `F6_HUMAN_phase_pref.ipynb` | Figure 6 |
| `SF9_HUMAN_ripples.ipynb` | Supplementary Figure 9 |
| `SF10_HUMAN_phase_pref_Patient2.ipynb` / `Patient3` / `_subselect` | Supplementary Figure 10 |
| `aux_shuffle_phase.ipynb` | auxiliary — phase-shuffle null distribution used for the phase-selectivity significance threshold |


