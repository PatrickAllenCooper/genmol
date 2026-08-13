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

"""Regression tests for the defects that made a parallel grid unsafe.

Each test corresponds to something that was actually broken:

  * Output paths were keyed on (oracle, start_mol_idx, sim_thr, seed) only, so
    every strategy/pop_cap/ucb_c/lam_* variant appended into one CSV with no
    column identifying the configuration. The committed
    results_v1_unshaped/iterlog_bandit_parp1_id0_0.csv still shows the damage:
    its iter column reads 1,2,1,3,2,4,3,5,6,4.
  * --seed named a file and seeded no RNG, so replicates were unreproducible.
  * DockingVina scanned docking/tmp/tmpN with a check-then-create loop on a
    shared path; os.makedirs has no exist_ok, so concurrent tasks past the
    first died with FileExistsError out of __init__.
  * predict() rebuilt its result list by appending over sorted(keys), so one
    dead worker shifted every later score onto the wrong molecule -- and the
    zip() in update_population truncated instead of raising.

Runs without rdkit or openbabel installed: only the heavy imports are stubbed,
and everything under test is stdlib + numpy.

    python scripts/exps/lead/test_grid_safety.py
"""

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEAD = REPO / 'scripts' / 'exps' / 'lead'
sys.path.insert(0, str(REPO))

_FAILURES = []


def check(label, cond):
    print(('PASS  ' if cond else 'FAIL  ') + label)
    if not cond:
        _FAILURES.append(label)


class _Any:
    """Stands in for anything we never actually call."""

    def __getattr__(self, item):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_heavy_imports():
    for name in ['rdkit', 'rdkit.Chem', 'openbabel', 'openbabel.pybel',
                 'genmol', 'genmol.sampler', 'genmol.utils',
                 'genmol.utils.utils_chem', 'sascorer', 'scripts',
                 'scripts.exps', 'scripts.exps.lead',
                 'scripts.exps.lead.docking',
                 'scripts.exps.lead.docking.docking']:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['rdkit'].Chem = _Any()
    sys.modules['rdkit'].DataStructs = _Any()
    sys.modules['rdkit.Chem'].DataStructs = _Any()
    sys.modules['rdkit.Chem'].AllChem = _Any()
    sys.modules['rdkit.Chem'].QED = _Any()
    sys.modules['rdkit.Chem'].RDConfig = types.SimpleNamespace(RDContribDir='.')
    sys.modules['openbabel'].pybel = sys.modules['openbabel.pybel']
    sys.modules['genmol.sampler'].Sampler = _Any()
    sys.modules['genmol.utils.utils_chem'].cut = lambda smiles: set()
    sys.modules['scripts.exps.lead.docking.docking'].DockingVina = _Any()


BASE_ARGS = dict(
    oracle_name='parp1', start_mol_idx=0, sim_thr=0.4, seed=0,
    model_path='model.ckpt', num_gen=100, num_iter=10, gamma=0,
    lam_rq=1.0, lam_rs=1.0, lam_rsim=1.0, strategy='bandit',
    ucb_c=2.0, pop_cap=150, q_alpha=0.4, epsilon=0.15,
    q_init='zero', q_optimistic=12.0, out_dir='/tmp/x',
    num_sub_proc=None, force=False,
)


def test_run_id(lead_run):
    print('\n--- run_id separates every scientific axis ---')

    def args(**over):
        return argparse.Namespace(**{**BASE_ARGS, **over})

    base = lead_run.compute_run_id(args())

    # Axes that must produce a distinct run directory.
    for field, value in [('ucb_c', 0.5), ('pop_cap', 25), ('lam_rq', 10.0),
                         ('lam_rs', 0.0), ('lam_rsim', 0.0), ('sim_thr', 0.5),
                         ('num_gen', 50), ('num_iter', 30), ('gamma', 0.3),
                         ('q_alpha', 0.1), ('q_init', 'admit_rv'),
                         ('epsilon', 0.3), ('strategy', 'random'),
                         ('seed', 7), ('start_mol_idx', 1),
                         ('oracle_name', 'jak2')]:
        check(f'{field} changes run_id',
              lead_run.compute_run_id(args(**{field: value})) != base)

    # The random arm used to tag its iterlog with the bare string "random", so
    # runs differing only in pop_cap truncated each other (mode 'w').
    check('random arm: pop_cap disambiguated',
          lead_run.compute_run_id(args(strategy='random', pop_cap=100))
          != lead_run.compute_run_id(args(strategy='random', pop_cap=200)))

    # Relocating output or swapping the checkpoint path must NOT rename a run,
    # or resuming would re-run everything.
    for field, value in [('out_dir', '/scratch/other'),
                         ('model_path', '/abs/model.ckpt'),
                         ('num_sub_proc', 8)]:
        check(f'{field} does not change run_id',
              lead_run.compute_run_id(args(**{field: value})) == base)

    check('run_id is deterministic', lead_run.compute_run_id(args()) == base)
    check('run_id is filesystem-safe',
          all(c.isalnum() or c in '_-.' for c in base))


def test_seeding(lead_run):
    print('\n--- --seed actually seeds the RNGs ---')
    import random
    import numpy as np

    def draw():
        return (random.random(), float(np.random.rand()),
                tuple(random.sample(range(100), 3)))

    lead_run.seed_everything(123)
    first = draw()
    lead_run.seed_everything(123)
    check('same seed reproduces random + numpy', draw() == first)
    lead_run.seed_everything(124)
    check('different seed diverges', draw() != first)


def test_affinity_alignment(dockmod):
    print('\n--- docking results stay aligned with their SMILES ---')
    DockingVina = dockmod.DockingVina
    sentinel = dockmod.DOCK_FAILED

    complete = {i: -(7.0 + i) for i in range(10)}
    check('complete batch passes through in order',
          DockingVina.collect_affinities(complete, 10)
          == [-(7.0 + i) for i in range(10)])

    gapped = {i: -(7.0 + i) for i in range(10) if i != 3}
    out = DockingVina.collect_affinities(gapped, 10)
    check('dead worker: length preserved', len(out) == 10)
    check('dead worker: sentinel at the right index', out[3] == sentinel)
    check('dead worker: later scores not shifted',
          all(out[i] == -(7.0 + i) for i in range(10) if i != 3))

    # Witness that the old rebuild really did corrupt the mapping.
    old = [gapped[k] for k in sorted(gapped)]
    check('old sorted-key rebuild misaligned (regression witness)',
          len(old) == 9 and old[3] == -11.0)

    check('all workers dead -> all sentinels',
          DockingVina.collect_affinities({}, 5) == [sentinel] * 5)
    check('empty batch stays empty', DockingVina.collect_affinities({}, 0) == [])


def test_temp_dirs(dockmod):
    print('\n--- concurrent DockingVina instances do not collide ---')
    DockingVina = dockmod.DockingVina
    os.environ.setdefault('TMPDIR', os.environ.get('TEMP', '/tmp'))

    instances = [DockingVina('parp1') for _ in range(8)]
    dirs = [inst.temp_dir for inst in instances]
    check('8 instances get 8 distinct temp dirs', len(set(dirs)) == 8)
    check('every temp dir exists', all(os.path.isdir(d) for d in dirs))
    check('no temp dir inside the repo checkout',
          not any(str(REPO).lower() in d.lower() for d in dirs))

    explicit = os.path.join(os.environ['TMPDIR'], 'genmol_explicit_test')
    inst = DockingVina('parp1', temp_dir=explicit)
    check('explicit temp_dir honoured', inst.temp_dir == explicit)
    check('explicit temp_dir not owned', inst._owns_temp_dir is False)
    del inst
    check('explicit temp_dir survives teardown', os.path.isdir(explicit))
    os.rmdir(explicit)

    check('num_sub_proc defaults to 10', DockingVina('parp1').num_sub_proc == 10)
    check('num_sub_proc honours the argument',
          DockingVina('parp1', num_sub_proc=8).num_sub_proc == 8)
    os.environ['GENMOL_NUM_SUB_PROC'] = '6'
    check('num_sub_proc honours GENMOL_NUM_SUB_PROC',
          DockingVina('parp1').num_sub_proc == 6)
    del os.environ['GENMOL_NUM_SUB_PROC']
    # exhaustiveness=1 means qvina runs one Monte Carlo task and --cpu is inert.
    check('num_cpu_dock is 1', DockingVina('parp1').num_cpu_dock == 1)

    half_built = DockingVina.__new__(DockingVina)
    try:
        half_built.__del__()
        check('__del__ on a half-built instance does not mask the real error', True)
    except Exception as exc:
        check(f'__del__ on a half-built instance raised {exc!r}', False)


def test_manifest_ids(lead_run):
    print('\n--- every manifest entry maps to its own run directory ---')
    import json
    import subprocess

    make = _load('make_manifest', LEAD / 'make_manifest.py')
    anchor = {'pop_cap': 50, 'ucb_c': 1.0, 'lam_rq': 1.0,
              'q_alpha': 0.4, 'q_init': 'zero'}
    extra = dict(model_path='model.ckpt', out_dir='/x',
                 num_sub_proc=None, force=False)

    grand, all_ids = 0, set()
    for stage in ['pilot', '1a', '1b', '1c', '1d', '2', '3']:
        configs = make.STAGES[stage](anchor)
        ids = [lead_run.compute_run_id(argparse.Namespace(**{**cfg, **extra}))
               for cfg in configs]
        check(f'stage {stage}: {len(configs)} configs -> no run_id collisions',
              len(ids) == len(set(ids)))
        grand += len(configs)
        all_ids |= set(ids)
    print(f'      {grand} configs, {len(all_ids)} unique run directories '
          f'({grand - len(all_ids)} repeats across stages, skipped on resubmit)')


def main():
    _stub_heavy_imports()
    lead_run = _load('lead_run', LEAD / 'run.py')
    dockmod = _load('dockmod', LEAD / 'docking' / 'docking.py')

    test_run_id(lead_run)
    test_seeding(lead_run)
    test_affinity_alignment(dockmod)
    test_temp_dirs(dockmod)
    test_manifest_ids(lead_run)

    print()
    if _FAILURES:
        print(f'{len(_FAILURES)} FAILURE(S):')
        for f in _FAILURES:
            print(f'  - {f}')
        return 1
    print('All checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
