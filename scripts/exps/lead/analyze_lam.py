# scripts/exps/lead/analyze_lam.py
import os, glob, re
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
files = sorted(glob.glob(os.path.join(ROOT_DIR, 'results', 'iterlog_bandit-*lamq*.csv')))
print(f'Found {len(files)} lam-sweep log files')
if not files:
    raise SystemExit('No lam-sweep logs found — check the filename tag.')

frames = []
for f in files:
    base = os.path.basename(f)
    m = re.search(r'iterlog_(.+?)_([\w\d]+)_id(\d+)_(\d+)\.csv', base)
    tag, oracle, seed = m.group(1), m.group(2), int(m.group(4))
    lam = float(re.search(r'lamq([\d.]+)', tag).group(1))
    d = pd.read_csv(f)
    d['lam_rq'] = lam
    d['oracle'] = oracle
    d['seed'] = seed
    d['run_id'] = f'lam{lam}_s{seed}'
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

num = ['rv_mean','rv_max','n_admitted_mols','fail_rv','fail_rq','fail_rs','fail_rsim','top_ds','iter']
for c in num:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# flag implausible docking scores
outliers = df[df['rv_max'] > 15]
if len(outliers):
    print(f'\nWARNING: {len(outliers)} iterations with rv_max > 15 (likely Vina failures)')
    print(outliers[['lam_rq','seed','iter','rv_max']].to_string(index=False))

finals = df.sort_values('iter').groupby('run_id').tail(1)
starts = df.sort_values('iter').groupby('run_id').head(1).set_index('run_id')['top_ds']
finals = finals.set_index('run_id')
finals['improvement'] = finals['top_ds'] - starts

print('\n--- Final Top DS by lam_rq ---')
print(finals.groupby('lam_rq')['top_ds'].agg(['mean','std','min','max','count']).round(3))

print('\n--- Improvement over start by lam_rq ---')
print(finals.groupby('lam_rq')['improvement'].agg(['mean','std']).round(3))

print('\n--- rv_mean / rv_max (pooled) by lam_rq ---')
print(df.groupby('lam_rq')[['rv_mean','rv_max']].agg(['mean','std']).round(3))

print('\n--- Gate outcomes by lam_rq (mean per iteration) ---')
gate = df.groupby('lam_rq')[['fail_rv','fail_rq','fail_rs','fail_rsim','n_admitted_mols']].mean()
print(gate.round(1))

print('\n--- Second-half only (iters 6-10), after convergence ---')
late = df[df['iter'] > 5]
print(late.groupby('lam_rq')[['rv_mean','rv_max','fail_rv','fail_rq','n_admitted_mols']].mean().round(2))

print('\n--- rv_mean trajectory by lam_rq ---')
print(df.pivot_table(index='iter', columns='lam_rq', values='rv_mean', aggfunc='mean').round(2))

print('\n--- Per-run finals (check variance) ---')
print(finals[['lam_rq','seed','top_ds','rv_mean','rv_max']].sort_values(['lam_rq','seed']).to_string(index=False))