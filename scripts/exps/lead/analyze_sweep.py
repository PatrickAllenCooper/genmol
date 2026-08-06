import os, glob, re
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
files = sorted(glob.glob(os.path.join(ROOT_DIR, 'results', 'iterlog_*.csv')))
print(f'Found {len(files)} log files')

frames = []
for f in files:
    d = pd.read_csv(f)
    m = re.search(r'iterlog_(\w+?)_([\w\d]+)_id(\d+)_(\d+)\.csv', os.path.basename(f))
    d['oracle'] = "parp1"
    d['seed'] = int(m.group(4))
    d['run_id'] = f'{m.group(1)}_{m.group(2)}_s{m.group(4)}'
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

finals = df.sort_values('iter').groupby('run_id').tail(1)
starts = df.sort_values('iter').groupby('run_id').head(1).set_index('run_id')['top_ds']
finals = finals.set_index('run_id')
finals['improvement'] = finals['top_ds'] - starts

print('\n--- Final Top DS by oracle x strategy ---')
print(finals.groupby(['oracle', 'strategy'])['top_ds'].agg(['mean', 'std', 'count']).round(3))

print('\n--- Improvement over start by oracle x strategy ---')
print(finals.groupby(['oracle', 'strategy'])['improvement'].agg(['mean', 'std']).round(3))

print('\n--- rv_mean (pooled) by oracle x strategy ---')
print(df.groupby(['oracle', 'strategy'])['rv_mean'].agg(['mean', 'std']).round(3))

print('\n--- rv_max (pooled) by oracle x strategy ---')
print(df.groupby(['oracle', 'strategy'])['rv_max'].agg(['mean', 'std']).round(3))

print('\n--- Admissions by oracle x strategy ---')
print(df.groupby(['oracle', 'strategy'])['n_admitted'].sum())

print('\n--- rv_mean trajectory: bandit minus random, per oracle ---')
traj = df.pivot_table(index=['oracle', 'iter'], columns='strategy', values='rv_mean', aggfunc='mean')
traj['delta'] = traj['bandit'] - traj['random']
print(traj.round(3))