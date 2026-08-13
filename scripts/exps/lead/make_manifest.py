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

"""Emit one stage of the bandit-vs-random grid as a JSONL manifest.

Each line is one run: a flat dict of run.py flags. hpc/slurm_genmol_grid.sh
shards the file by line index across workers, so the manifest is the single
source of truth for what the grid contains.

The bandit-only axes (ucb_c, lam_*, q_alpha, q_init) do not apply to the random
arm, so the stages below are unions of sub-designs rather than one product.

Staged rather than full-factorial for two reasons grounded in the code:

  * A run makes 2 * num_gen * num_iter = 2000 bandit pulls, while cut() admits
    200-320 new fragments per iteration. pop_cap is what actually sets the arm
    count, so the interesting regime is BELOW the 100/150/200 originally
    proposed -- hence {0, 25, 50, 100, 200}.
  * The shortfall term is bounded by lam_rq against rv in roughly [0, 13], so at
    lam_rq=200 the reward is essentially -200 * (QED shortfall) and the UCB
    bonus (~1.7 at c=2) is numerically irrelevant. That cell measures greedy QED
    avoidance, not a bandit, so it is a single anchor rather than a full row.

Usage:
    python scripts/exps/lead/make_manifest.py --stage 1a -o manifest_1a.jsonl
    python scripts/exps/lead/make_manifest.py --stage 1b \\
        --anchor best_1a.json -o manifest_1b.jsonl
"""

import argparse
import itertools
import json
import os
import sys

# Targets used for the screening stages. parp1 is DS-bound (fail_rq climbs from
# ~0 to ~30 per 100 as the search proceeds); jak2 is QED-bound from the first
# iteration (fail_rq 41-67 per 100). Screening lam_rq on parp1 alone would
# understate it.
SCREEN_TARGETS = ['parp1', 'jak2']
ALL_TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']

POP_CAPS = [0, 25, 50, 100, 200]        # 0 = uncapped, reproduces main
UCB_CS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]  # 0 = greedy, 8 -> approaches random
LAM_RQS = [0.0, 1.0, 2.0, 3.0, 10.0]
Q_ALPHAS = [0.1, 0.4, 0.0]               # 0.0 = true sample average

SCREEN_SEEDS = list(range(5))
CONFIRM_SEEDS = list(range(10))

DEFAULTS = dict(
    oracle_name='parp1', start_mol_idx=0, sim_thr=0.4, seed=0,
    num_gen=100, num_iter=10, gamma=0.0,
    lam_rq=1.0, lam_rs=1.0, lam_rsim=1.0,
    strategy='bandit', ucb_c=2.0, pop_cap=150,
    q_alpha=0.4, epsilon=0.15, q_init='zero', q_optimistic=12.0,
)


def run(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


def stage_pilot(anchor):
    """Smoke test and per-run timing. Deliberately tiny.

    Submit this same manifest to both aa100 and ah200 to compare wall-clock.
    The reproducibility check is a manual `--force` re-run of one of these runs
    followed by a diff of molecules.csv, not a manifest line: an identical
    config is an identical run_id by construction.
    """
    return [run(oracle_name=target, strategy=strategy, seed=0, num_iter=2)
            for target in SCREEN_TARGETS
            for strategy in ['bandit', 'random']]


def stage_1a(anchor):
    """Structure screen: what arm count and exploration rate does UCB1 need?"""
    return [run(oracle_name=t, pop_cap=cap, ucb_c=c, seed=s,
                strategy='bandit', lam_rq=1.0)
            for t, cap, c, s in itertools.product(
                SCREEN_TARGETS, POP_CAPS, UCB_CS, SCREEN_SEEDS)]


def stage_1b(anchor):
    """Shaping screen at the 1a winner."""
    return [run(oracle_name=t, seed=s, strategy='bandit',
                pop_cap=anchor['pop_cap'], ucb_c=anchor['ucb_c'],
                lam_rq=lam, q_alpha=qa)
            for t, lam, qa, s in itertools.product(
                SCREEN_TARGETS, LAM_RQS, Q_ALPHAS, SCREEN_SEEDS)]


def stage_1c(anchor):
    """Random baselines. pop_cap applies to both arms, so it is swept here too."""
    return [run(oracle_name=t, strategy='random', pop_cap=cap, seed=s)
            for t, cap, s in itertools.product(
                SCREEN_TARGETS, POP_CAPS, SCREEN_SEEDS)]


def stage_1d(anchor):
    """lam_rq=200 anchor: the degenerate limit, one row not a full crossing."""
    return [run(oracle_name=t, seed=s, strategy='bandit', lam_rq=200.0,
                pop_cap=anchor['pop_cap'], ucb_c=anchor['ucb_c'])
            for t, s in itertools.product(SCREEN_TARGETS, SCREEN_SEEDS)]


def stage_2(anchor):
    """Mechanism ablations: Q-initialisation, horizon, and an egreedy arm."""
    out = [run(oracle_name=t, seed=s, strategy='bandit',
               pop_cap=anchor['pop_cap'], ucb_c=anchor['ucb_c'],
               lam_rq=anchor.get('lam_rq', 1.0),
               q_alpha=anchor.get('q_alpha', 0.4),
               q_init=qi, num_iter=ni)
           for t, qi, ni, s in itertools.product(
               SCREEN_TARGETS, ['zero', 'admit_rv'], [10, 30], SCREEN_SEEDS)]
    out += [run(oracle_name=t, seed=s, strategy='egreedy',
                pop_cap=anchor['pop_cap'],
                lam_rq=anchor.get('lam_rq', 1.0),
                q_alpha=anchor.get('q_alpha', 0.4))
            for t, s in itertools.product(SCREEN_TARGETS, SCREEN_SEEDS)]
    return out


def stage_3(anchor):
    """Confirmation: the winning bandit vs egreedy vs random, all targets."""
    out = []
    for target, idx, seed in itertools.product(
            ALL_TARGETS, [0, 1, 2], CONFIRM_SEEDS):
        common = dict(oracle_name=target, start_mol_idx=idx, seed=seed,
                      pop_cap=anchor['pop_cap'])
        out.append(run(strategy='bandit', ucb_c=anchor['ucb_c'],
                       lam_rq=anchor.get('lam_rq', 1.0),
                       q_alpha=anchor.get('q_alpha', 0.4),
                       q_init=anchor.get('q_init', 'zero'), **common))
        out.append(run(strategy='egreedy',
                       lam_rq=anchor.get('lam_rq', 1.0),
                       q_alpha=anchor.get('q_alpha', 0.4), **common))
        out.append(run(strategy='random', **common))
    return out


STAGES = {
    'pilot': stage_pilot, '1a': stage_1a, '1b': stage_1b, '1c': stage_1c,
    '1d': stage_1d, '2': stage_2, '3': stage_3,
}

# Stages parameterised by the winner of an earlier stage.
NEEDS_ANCHOR = {'1b', '1d', '2', '3'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=sorted(STAGES))
    parser.add_argument('-o', '--output', default='-',
                        help='JSONL destination; - for stdout')
    parser.add_argument('--anchor', default=None,
                        help='JSON file with the winning config from an earlier '
                             'stage (needs at least pop_cap and ucb_c)')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='override the target list for this stage')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='override the seed list for this stage')
    args = parser.parse_args()

    global SCREEN_TARGETS, ALL_TARGETS, SCREEN_SEEDS, CONFIRM_SEEDS
    if args.targets:
        SCREEN_TARGETS = ALL_TARGETS = list(args.targets)
    if args.seeds:
        SCREEN_SEEDS = CONFIRM_SEEDS = list(args.seeds)

    anchor = None
    if args.anchor:
        with open(args.anchor) as f:
            anchor = json.load(f)
    if args.stage in NEEDS_ANCHOR:
        if anchor is None:
            parser.error(f'--stage {args.stage} needs --anchor '
                         f'(the winning config from the previous stage)')
        for key in ('pop_cap', 'ucb_c'):
            if key not in anchor:
                parser.error(f'anchor file is missing required key {key!r}')

    runs = STAGES[args.stage](anchor)

    # run.py is idempotent on run_id, so duplicate lines would be wasted
    # scheduling rather than corrupt data -- still worth catching here.
    seen, unique = set(), []
    for cfg in runs:
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    n_dupes = len(runs) - len(unique)

    lines = [json.dumps(cfg, sort_keys=True) for cfg in unique]
    if args.output == '-':
        print('\n'.join(lines))
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'stage {args.stage}: wrote {len(unique)} runs to {args.output}'
              + (f' ({n_dupes} duplicates dropped)' if n_dupes else ''),
              file=sys.stderr)


if __name__ == '__main__':
    main()
