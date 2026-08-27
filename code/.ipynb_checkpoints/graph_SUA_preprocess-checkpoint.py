### Neighbouring unit-type pairs, RS or NATIM, with optional quality filtering ###
#
#   python graph_SUA_preprocess_new_filter.py RS
#   python graph_SUA_preprocess_new_filter.py NATIM
#   python graph_SUA_preprocess_new_filter.py RS pass_k3        # quality-filtered
#
# NEW FILTER VERSION. This script never touches a spike/nix file at all - it
# works entirely off the property dataframes (final_class, channel_order),
# per its own header note about removing the neo block load for speed. So the
# only things that need to change are:
#   1. params_analysis.yml path -> code_new_filter
#   2. DF_FOLDER -> dataframes_new_filter, which automatically repoints
#      PROP_FOLDER ('sua_prop_all' / 'sua_prop_all_NATIM') and OUT_FOLDER at
#      the new-filter tree, since both are built from DF_FOLDER already.
#
# ONE THING TO DECIDE: the optional QUALITY_LEVEL argument (e.g. 'pass_k3')
# joins against f'{DF_FOLDER}/sua_quality_{TYPE_REC}/unit_inclusion_list.pkl'.
# That's one of the dataframes HANDOVER_filtered_data.md marks OLD (built on
# the earlier good-units set), and apply_quality() has no try/except around
# opening it - the second-you point DF_FOLDER at dataframes_new_filter/, that
# file won't exist there (it was never generated for the new pipeline) and
# the script will raise FileNotFoundError immediately if you pass a
# QUALITY_LEVEL. This is very likely fine to just not use going forward: the
# current filtering (SNR/viol_ratio/coinc_ratio thresholds) already IS the
# quality gate now, so units in the new spikes_filtered/ set don't need a
# second quality pass - call this script with no second argument. If you
# actually want a further quality cut on top, that inclusion list would need
# to be rebuilt against the new unit set first, which isn't done here.

from functions_analysis import *
import pandas as pd
import numpy as np
import yaml
import pickle
import itertools
import sys
import os
from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

##### PARAMETERS #####
with open('/CSNG/studekat/ripple_paper_clean_copy/code_new_filter/params_analysis.yml') as f:
    params_analysis = yaml.safe_load(f)

MAIN_FOLDER = params_analysis['main_folder']
DF_FOLDER = f'{MAIN_FOLDER}/dataframes_new_filter'  # NEW
FINAL_CLASSES = params_analysis['final_classes']

TYPE_REC = sys.argv[1].upper() if len(sys.argv) > 1 else 'RS'
QUALITY_LEVEL = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. 'pass_k3' - see note above, likely obsolete now

if TYPE_REC == 'RS':
    MONKEY_LIST = ['L', 'N', 'F']
    PROP_FOLDER = 'sua_prop_all'
elif TYPE_REC == 'NATIM':
    MONKEY_LIST = ['N', 'F']
    PROP_FOLDER = 'sua_prop_all_NATIM'
else:
    print(f'Unknown recording type {TYPE_REC}')
    sys.exit(1)

N_PERM = 1000            # permutations for the null
RNG_SEED = 0
SHUFFLE_SCOPES = ['global', 'within']   # both are computed, see the header

tag = f'{TYPE_REC}' + (f'_{QUALITY_LEVEL}' if QUALITY_LEVEL else '_all')
OUT_FOLDER = f'{DF_FOLDER}/dict_pair_occur_{tag}'

print(f'Recording type: {TYPE_REC}')
print(f'Animals: {MONKEY_LIST}')
print(f'Quality filter: {QUALITY_LEVEL if QUALITY_LEVEL else "none"}')
print(f'Shuffle scopes: {SHUFFLE_SCOPES}')
print(f'Output: {OUT_FOLDER}')

KEY = ['monkey', 'date', 'array', 'cell_name']

# paired class names, same convention as the original
all_pairs_comb = list(itertools.combinations(FINAL_CLASSES, 2))
pairs_merged = ["+".join(sorted(t)) for t in all_pairs_comb]
for cl in FINAL_CLASSES:
    pairs_merged.append("+".join((cl, cl)))
PAIR_INDEX = {p: i for i, p in enumerate(pairs_merged)}
N_PAIRS = len(pairs_merged)

# lookup from an ordered class-index pair to a flat pair index
CLASS_INDEX = {cl: i for i, cl in enumerate(FINAL_CLASSES)}
PAIR_LUT = np.zeros((len(FINAL_CLASSES), len(FINAL_CLASSES)), dtype=int)
for i, a in enumerate(FINAL_CLASSES):
    for j, b in enumerate(FINAL_CLASSES):
        PAIR_LUT[i, j] = PAIR_INDEX["+".join(sorted((a, b)))]


##### LOADING #####
def load_prop_tagged(monkey, type_rec, prop_folder, exclude_noisy=True):
    """Property dataframes for one animal, tagged with monkey and date."""
    dfs = []
    for date in params_analysis['dates'][monkey][type_rec]:
        p = f'{DF_FOLDER}/{prop_folder}/monkey{monkey}_all_arrays_date_{date}.pkl'
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
        except Exception:
            print(f'   missing: {monkey} {date}')
            continue
        d['monkey'] = monkey
        d['date'] = date
        dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    if exclude_noisy:
        df = df[~df['ch_is_noisy_100Hz']]
        df = df[~df['ch_is_noisy_120Hz']]
    return df.reset_index(drop=True)


def apply_quality(df, level):
    """Join the inclusion list and keep units passing `level`."""
    p = f'{DF_FOLDER}/sua_quality_{TYPE_REC}/unit_inclusion_list.pkl'
    with open(p, 'rb') as f:
        df_q = pickle.load(f)
    qcols = [c for c in KEY + ['n_quality_pass', level] if c in df_q.columns]
    d = df.merge(df_q[qcols], on=KEY, how='left', suffixes=('', '_q'))
    n_missing = int(d['n_quality_pass'].isna().sum())
    if n_missing:
        print(f'   {n_missing} units without a quality entry, treated as failing')
    d[level] = d[level].fillna(False).astype(bool)
    return d[d[level]].reset_index(drop=True)


##### PAIR COUNTING #####
def layout_coords(layout):
    """Channel -> (row, col), resolved once instead of inside the pair loop."""
    coords = np.full((64, 2), -1, dtype=int)
    for ch in range(64):
        idx = np.where(layout == ch)
        if len(idx[0]):
            coords[ch] = (idx[0][0], idx[1][0])
    return coords


def count_pairs_array(classes_idx, channels, coords):
    """
    Vectorised replacement for aux_calculate_pairs.

    Returns (all_counts, close_counts), each a length-N_PAIRS integer vector.
    Two channels are neighbours when they differ by less than 2 in both row and
    column, matching the original definition (this includes diagonals).
    """
    n = len(classes_idx)
    all_counts = np.zeros(N_PAIRS, dtype=np.int64)
    close_counts = np.zeros(N_PAIRS, dtype=np.int64)
    if n < 2:
        return all_counts, close_counts

    valid = (channels >= 0) & (channels < 64)
    classes_idx = classes_idx[valid]
    channels = channels[valid]
    n = len(classes_idx)
    if n < 2:
        return all_counts, close_counts

    xy = coords[channels]                     # (n, 2)
    ok = xy[:, 0] >= 0
    classes_idx, xy = classes_idx[ok], xy[ok]
    n = len(classes_idx)
    if n < 2:
        return all_counts, close_counts

    i, j = np.triu_indices(n, k=1)
    d_row = np.abs(xy[i, 0] - xy[j, 0])
    d_col = np.abs(xy[i, 1] - xy[j, 1])
    is_neigh = (d_row < 2) & (d_col < 2)

    pair_ids = PAIR_LUT[classes_idx[i], classes_idx[j]]
    all_counts += np.bincount(pair_ids, minlength=N_PAIRS)
    if is_neigh.any():
        close_counts += np.bincount(pair_ids[is_neigh], minlength=N_PAIRS)
    return all_counts, close_counts


def neighbour_index_pairs(classes_idx, xy, offset):
    """
    Indices of neighbouring unit pairs on one array, expressed as offsets into
    a global label array so that permutations can be applied across all blocks
    at once.

    Returns (i_global, j_global). Only adjacency is stored; the class labels
    themselves are supplied at permutation time.
    """
    n = len(classes_idx)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    i, j = np.triu_indices(n, k=1)
    d_row = np.abs(xy[i, 0] - xy[j, 0])
    d_col = np.abs(xy[i, 1] - xy[j, 1])
    is_neigh = (d_row < 2) & (d_col < 2)
    return i[is_neigh] + offset, j[is_neigh] + offset


def run_permutations(labels_global, block_slices, i_glob, j_glob,
                     scope, n_perm, rng):
    """
    Null distribution of neighbouring-pair counts.

    scope='global': one shuffle of the whole label vector per permutation, so
        classes are exchanged freely between arrays, dates and animals. Only
        the overall class composition is held fixed.

    scope='within': each block is shuffled independently, so every array keeps
        its own class composition.

    Channel positions and the adjacency structure are fixed in both cases; only
    which unit carries which class label changes.
    """
    out = np.zeros((n_perm, N_PAIRS), dtype=np.int64)
    if len(i_glob) == 0:
        return out
    lab = labels_global.copy()
    for p in range(n_perm):
        if scope == 'global':
            rng.shuffle(lab)
        else:
            for a, b in block_slices:
                if b - a > 1:
                    seg = lab[a:b]
                    rng.shuffle(seg)
                    lab[a:b] = seg
        pair_ids = PAIR_LUT[lab[i_glob], lab[j_glob]]
        out[p] = np.bincount(pair_ids, minlength=N_PAIRS)
    return out


##### MAIN #####
all_sums = np.zeros(N_PAIRS, dtype=np.int64)
close_sums = np.zeros(N_PAIRS, dtype=np.int64)
classes_counter = {cl: 0 for cl in FINAL_CLASSES}
rng = np.random.default_rng(RNG_SEED)

# collected across every block, so that a global shuffle is possible
labels_global = []
block_slices = []
i_glob_parts, j_glob_parts = [], []
offset = 0

n_units_used = 0
n_blocks = 0

for monkey in MONKEY_LIST:
    print(f'=== {monkey} ===')
    df = load_prop_tagged(monkey, TYPE_REC, PROP_FOLDER, exclude_noisy=True)
    if df is None:
        print('   no data')
        continue
    print(f'   {df.shape[0]} units loaded')
    if QUALITY_LEVEL:
        df = apply_quality(df, QUALITY_LEVEL)
        print(f'   {df.shape[0]} units after {QUALITY_LEVEL}')

    df = df[df['final_class'].isin(FINAL_CLASSES)]
    df = df[df['channel_order'] > -1]

    layouts = {
        'odd': layout_coords(np.array(params_analysis['layout'][f'{monkey}_odd'])),
        'even': layout_coords(np.array(params_analysis['layout'][f'{monkey}_even'])),
    }
    v_areas = [a in ['V1', 'V2'] for a in params_analysis['areas'][monkey]]

    for (date, array), grp in df.groupby(['date', 'array'], sort=False):
        array = int(array)
        if not (1 <= array <= 16) or not v_areas[array - 1]:
            continue
        coords = layouts['even'] if array % 2 == 0 else layouts['odd']

        cls_idx = grp['final_class'].map(CLASS_INDEX).values.astype(int)
        chans = grp['channel_order'].values.astype(int)

        a_cnt, c_cnt = count_pairs_array(cls_idx, chans, coords)
        all_sums += a_cnt
        close_sums += c_cnt

        for cl in grp['final_class']:
            classes_counter[cl] += 1
        n_units_used += len(grp)
        n_blocks += 1

        # keep the same units the observed count used, in the same order
        valid = (chans >= 0) & (chans < 64)
        cls_v, chans_v = cls_idx[valid], chans[valid]
        xy = coords[chans_v]
        ok = xy[:, 0] >= 0
        cls_v, xy = cls_v[ok], xy[ok]
        if len(cls_v) == 0:
            continue

        ig, jg = neighbour_index_pairs(cls_v, xy, offset)
        i_glob_parts.append(ig)
        j_glob_parts.append(jg)
        labels_global.append(cls_v)
        block_slices.append((offset, offset + len(cls_v)))
        offset += len(cls_v)

labels_global = np.concatenate(labels_global) if labels_global else np.array([], dtype=int)
i_glob = np.concatenate(i_glob_parts) if i_glob_parts else np.array([], dtype=int)
j_glob = np.concatenate(j_glob_parts) if j_glob_parts else np.array([], dtype=int)

print()
print(f'{len(labels_global)} units in the permutation pool, '
      f'{len(i_glob)} neighbouring pairs')

perm_by_scope = {}
for scope in SHUFFLE_SCOPES:
    print(f'running {N_PERM} permutations, scope={scope} ...')
    rng_s = np.random.default_rng(RNG_SEED)
    perm_by_scope[scope] = run_permutations(
        labels_global, block_slices, i_glob, j_glob, scope, N_PERM, rng_s)

# sanity: total neighbouring pairs must be identical in every permutation,
# since only the labels move
tot_obs = int(close_sums.sum())
for scope, perm in perm_by_scope.items():
    tot_perm = perm.sum(axis=1)
    if not np.all(tot_perm == tot_perm[0]):
        print(f'WARNING [{scope}]: permutation totals vary, '
              'the null is not label-preserving')
    elif tot_perm[0] != tot_obs:
        print(f'WARNING [{scope}]: observed total {tot_obs} '
              f'!= permuted total {tot_perm[0]}')
    else:
        print(f'check ok [{scope}]: {tot_obs} neighbouring pairs '
              'in both observed and null')

##### RESULTS #####
all_sums_d = {p: int(all_sums[i]) for p, i in PAIR_INDEX.items()}
close_sums_d = {p: int(close_sums[i]) for p, i in PAIR_INDEX.items()}

# original crude chance model, kept for comparability
pooled_count = {}
for t in all_pairs_comb:
    a, b = sorted(t)
    pooled_count["+".join((a, b))] = classes_counter[a] * classes_counter[b]


def enrichment_table(perm_sums, scope):
    """Observed against the null, one row per class pair."""
    rows = []
    for p, i in PAIR_INDEX.items():
        obs = int(close_sums[i])
        null = perm_sums[:, i]
        exp = float(null.mean())
        sd = float(null.std())
        n_ge = int((null >= obs).sum())
        n_le = int((null <= obs).sum())
        p_two = 2 * min(n_ge + 1, n_le + 1) / (N_PERM + 1)
        rows.append({
            'scope': scope,
            'pair': p,
            'n_all_pairs': int(all_sums[i]),
            'n_close_obs': obs,
            'n_close_exp': round(exp, 2),
            'enrichment': round(obs / exp, 3) if exp > 0 else np.nan,
            'z': round((obs - exp) / sd, 3) if sd > 0 else np.nan,
            'p_perm': min(p_two, 1.0),
            'frac_close': round(obs / all_sums[i], 5) if all_sums[i] > 0 else np.nan,
        })
    d = pd.DataFrame(rows)
    try:
        from statsmodels.stats.multitest import multipletests
        ok = d['p_perm'].notna()
        d.loc[ok, 'p_holm'] = multipletests(d.loc[ok, 'p_perm'], method='holm')[1]
    except Exception:
        pass
    return d.sort_values('enrichment', ascending=False)


tables = {sc: enrichment_table(perm_by_scope[sc], sc) for sc in SHUFFLE_SCOPES}
df_res = pd.concat(tables.values(), ignore_index=True)

for sc in SHUFFLE_SCOPES:
    print()
    print(f'===== scope: {sc} =====')
    print(tables[sc].drop(columns='scope').to_string(index=False))

# side by side, so the scale the effect lives at is visible
if len(SHUFFLE_SCOPES) > 1:
    print()
    print('===== comparison =====')
    print('Enriched under global but not within means the classes occupy the')
    print('same arrays or animals rather than adjacent electrodes.')
    print()
    cmp = tables[SHUFFLE_SCOPES[0]][['pair', 'n_close_obs']].copy()
    for sc in SHUFFLE_SCOPES:
        t = tables[sc].set_index('pair')
        cmp[f'enr_{sc}'] = cmp['pair'].map(t['enrichment'])
        cmp[f'z_{sc}'] = cmp['pair'].map(t['z'])
        cmp[f'p_{sc}'] = cmp['pair'].map(
            t['p_holm'] if 'p_holm' in t.columns else t['p_perm'])
    cmp = cmp.sort_values(f'z_{SHUFFLE_SCOPES[0]}', ascending=False)
    print(cmp.to_string(index=False))

##### SAVING #####
ensure_dir_exists(OUT_FOLDER)
with open(f'{OUT_FOLDER}/all_pairs_count', 'wb') as f:
    pickle.dump(defaultdict(float, all_sums_d), f)
with open(f'{OUT_FOLDER}/close_pairs_count', 'wb') as f:
    pickle.dump(defaultdict(float, close_sums_d), f)
with open(f'{OUT_FOLDER}/classes_counter', 'wb') as f:
    pickle.dump(classes_counter, f)
with open(f'{OUT_FOLDER}/pooled_count', 'wb') as f:
    pickle.dump(pooled_count, f)
for sc in SHUFFLE_SCOPES:
    with open(f'{OUT_FOLDER}/perm_null_{sc}', 'wb') as f:
        pickle.dump(perm_by_scope[sc], f)
    tables[sc].to_csv(f'{OUT_FOLDER}/pair_enrichment_{sc}.csv', index=False)
df_res.to_csv(f'{OUT_FOLDER}/pair_enrichment_both.csv', index=False)
if len(SHUFFLE_SCOPES) > 1:
    cmp.to_csv(f'{OUT_FOLDER}/pair_enrichment_comparison.csv', index=False)

meta = {'type_rec': TYPE_REC, 'quality_level': QUALITY_LEVEL,
        'shuffle_scopes': SHUFFLE_SCOPES,
        'monkeys': MONKEY_LIST, 'n_perm': N_PERM, 'seed': RNG_SEED,
        'n_blocks': n_blocks, 'n_units': n_units_used,
        'classes_counter': classes_counter}
with open(f'{OUT_FOLDER}/meta', 'wb') as f:
    pickle.dump(meta, f)

print()
print(f'saved to {OUT_FOLDER}')
print('  all_pairs_count, close_pairs_count, classes_counter, pooled_count')
for sc in SHUFFLE_SCOPES:
    print(f'  perm_null_{sc}, pair_enrichment_{sc}.csv')
print('  pair_enrichment_both.csv, pair_enrichment_comparison.csv, meta')
