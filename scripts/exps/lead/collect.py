# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Join every run's config.json to its iterlog.csv into one tidy table.

Replaces anal_sweep.py / analyze_sweep.py / analyze_lam.py, which parsed
hyperparameters back out of filenames with

    r'iterlog_(\\w+?)_([\\w\\d]+)_id(\\d+)_(\\d+)\\.csv'

That regex cannot match the tags run.py emits for bandit runs
(`bandit-ucb2.0-cap150-lamq1.0`) because `\\w` matches neither `-` nor `.`, so
both scripts raise AttributeError on m.group(2) as soon as one bandit log is
present. Reading config.json instead means the schema can grow new axes without
touching any parsing code.

Emits one row per (run, iteration), plus a `finals` view with one row per run.

Usage:
    python scripts/exps/lead/collect.py --results-dir /scratch/.../results \\
        -o runs.parquet
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

# Docking failures come back as 99.9, which reward_vina negates and clips to 0.
# Anything above this in rv_max means Vina returned nonsense.
RV_SANITY_MAX = 15.0

ITERLOG_COLUMNS = ['iter', 'strategy', 'rv_mean', 'rv_max', 'n_admitted',
                   'n_admitted_mols', 'fail_rv', 'fail_rq', 'fail_rs',
                   'fail_rsim', 'top_ds']


def load_run(run_dir):
    """Return (per-iteration DataFrame, warning list) for one run directory."""
    warnings = []
    config_path = os.path.join(run_dir, 'config.json')
    iterlog_path = os.path.join(run_dir, 'iterlog.csv')
    status_path = os.path.join(run_dir, 'status.json')

    if not os.path.exists(config_path):
        return None, [f'{run_dir}: no config.json, skipping']
    with open(config_path) as f:
        config = json.load(f)

    if not os.path.exists(iterlog_path):
        return None, [f'{run_dir}: no iterlog.csv, skipping']
    df = pd.read_csv(iterlog_path)
    if df.empty:
        return None, [f'{run_dir}: empty iterlog, skipping']

    missing = set(ITERLOG_COLUMNS) - set(df.columns)
    if missing:
        return None, [f'{run_dir}: iterlog missing columns {sorted(missing)}, skipping']

    # status.json is written last and only on a clean finish, so its absence is
    # how a walltime/OOM kill is distinguished from a completed run.
    completed = os.path.exists(status_path)
    if not completed:
        warnings.append(f'{run_dir}: no status.json -- run did not finish, '
                        f'marked completed=False')
    elif len(df) != int(config.get('num_iter', len(df))):
        warnings.append(f'{run_dir}: status.json present but iterlog has '
                        f'{len(df)} rows for num_iter={config.get("num_iter")}')

    # A run's rows must be strictly increasing in iter. Interleaved rows meant
    # two processes shared one file -- the exact corruption in the committed
    # results_v1_unshaped/ logs (iter = 1,2,1,3,2,4,...).
    if not df['iter'].is_monotonic_increasing:
        warnings.append(f'{run_dir}: iter column is not monotonic -- '
                        f'concurrent writers? treating as untrusted')
        df['untrusted'] = True
    else:
        df['untrusted'] = False

    for key, value in config.items():
        if key in ('strategy',):
            continue                    # already a column in the iterlog
        df[key] = value
    df['completed'] = completed
    df['run_dir'] = run_dir
    return df, warnings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_results = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'results')
    parser.add_argument('--results-dir', default=default_results)
    parser.add_argument('-o', '--output', default='runs.parquet',
                        help='.parquet or .csv')
    parser.add_argument('--finals-output', default=None,
                        help='optional one-row-per-run summary')
    parser.add_argument('--include-incomplete', action='store_true',
                        help='keep runs with no status.json (default: drop)')
    args = parser.parse_args()

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.results_dir, '*'))
                      if os.path.isdir(d))
    if not run_dirs:
        sys.exit(f'No run directories under {args.results_dir}')

    frames, warnings = [], []
    for run_dir in run_dirs:
        df, warns = load_run(run_dir)
        warnings.extend(warns)
        if df is not None:
            frames.append(df)

    if not frames:
        sys.exit(f'Found {len(run_dirs)} directories but no usable runs')

    df = pd.concat(frames, ignore_index=True)

    n_all = df['run_id'].nunique()
    if not args.include_incomplete:
        df = df[df['completed']]
    n_kept = df['run_id'].nunique() if len(df) else 0

    # One row per run: the last iteration, plus improvement over the start.
    ordered = df.sort_values(['run_id', 'iter'])
    finals = ordered.groupby('run_id').tail(1).copy()
    finals['improvement'] = finals['top_ds'] - finals['start_ds']

    suspect = int((df['rv_max'] > RV_SANITY_MAX).sum())
    untrusted = int(df['untrusted'].sum())

    def write(frame, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if path.endswith('.csv'):
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)

    write(df, args.output)
    if args.finals_output:
        write(finals, args.finals_output)

    for w in warnings:
        print(f'WARNING: {w}', file=sys.stderr)

    print(f'runs found          : {n_all}')
    print(f'runs kept           : {n_kept}'
          + ('' if args.include_incomplete else ' (incomplete dropped)'))
    print(f'iteration rows      : {len(df)}')
    print(f'rv_max > {RV_SANITY_MAX} rows  : {suspect} (likely Vina failures)')
    print(f'untrusted rows      : {untrusted} (non-monotonic iter)')
    print(f'wrote               : {args.output}')
    if args.finals_output:
        print(f'wrote               : {args.finals_output}')

    if n_kept:
        cols = [c for c in ['strategy', 'oracle_name'] if c in finals.columns]
        if cols:
            print('\n--- improvement over start, by '
                  + ' x '.join(cols) + ' ---')
            print(finals.groupby(cols)['improvement']
                  .agg(['mean', 'std', 'count']).round(3))


if __name__ == '__main__':
    main()
