import os, glob, re
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
files = sorted(glob.glob(os.path.join(ROOT_DIR, 'results', 'iterlog_*.csv')))
print(f'Found {len(files)} log files')
if not files:
    raise SystemExit('No iterlog files found — did the sweep run?')

frames = []
for f in files:
    d = pd.read_csv(f)
    m = re.search(r'iterlog_(\w+?)_([\w\d]+)_id(\d+)_(\d+)\.csv', os.path.basename(f))
    d['oracle'] = m.group(2)
    d['seed'] = int(m.group(4))
    d['run_id'] = f'{m.group(1)}_{m.group(2)}_s{m.group(4)}'
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

required = {'fail_rv', 'fail_rq', 'fail_rs', 'fail_rsim', 'n_admitted_mols'}
if not required.issubset(df.columns):
    raise SystemExit(f'Missing columns {required - set(df.columns)} — old-schema logs mixed in?')

finals = df.sort_values('iter').groupby('run_id').tail(1)
starts = df.sort_values('iter').groupby('run_id').head(1).set_index('run_id')['top_ds']
finals = finals.set_index('run_id')
finals['improvement'] = finals['top_ds'] - starts

print('\n--- Final Top DS by oracle x strategy ---')
print(finals.groupby(['oracle', 'strategy'])['top_ds'].agg(['mean', 'std', 'count']).round(3))

print('\n--- Improvement over start ---')
print(finals.groupby(['oracle', 'strategy'])['improvement'].agg(['mean', 'std']).round(3))

print('\n--- rv_mean / rv_max (pooled) ---')
print(df.groupby(['oracle', 'strategy'])[['rv_mean', 'rv_max']].agg(['mean', 'std']).round(3))

print('\n--- Admissions: fragments added vs molecules passing gate ---')
adm = df.groupby(['oracle', 'strategy'])[['n_admitted', 'n_admitted_mols']].sum()
adm['frags_per_mol'] = (adm['n_admitted'] / adm['n_admitted_mols']).round(2)
print(adm)

fail_cols = ['fail_rv', 'fail_rq', 'fail_rs', 'fail_rsim']

print('\n--- Binding constraint: share of molecules stopped at each gate (%) ---')
fails = df.groupby(['oracle', 'strategy'])[fail_cols].sum()
fails['passed'] = df.groupby(['oracle', 'strategy'])['n_admitted_mols'].sum()
pct = fails.div(fails.sum(axis=1), axis=0).mul(100).round(1)
print(pct)

print('\n--- Gate-failure shift, bandit minus random (pp) ---')
shift = pct.unstack('strategy')
delta = pd.DataFrame({c: shift[(c, 'bandit')] - shift[(c, 'random')]
                      for c in list(fail_cols) + ['passed']}).round(1)
print(delta)

print('\n--- fail_rv trajectory by iteration (mean per run) ---')
print(df.pivot_table(index=['oracle', 'iter'], columns='strategy',
                     values='fail_rv', aggfunc='mean').round(1))

print('\n--- Downstream gate failures (rq+rs+rsim) by iteration ---')
df['fail_downstream'] = df[['fail_rq', 'fail_rs', 'fail_rsim']].sum(axis=1)
traj = df.pivot_table(index=['oracle', 'iter'], columns='strategy',
                      values='fail_downstream', aggfunc='mean')
traj['delta'] = (traj['bandit'] - traj['random']).round(1)
print(traj.round(1))