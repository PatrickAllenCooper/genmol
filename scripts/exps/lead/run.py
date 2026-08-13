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


import os
import sys
sys.path.append(os.path.realpath('.'))

from time import time
from collections import defaultdict
import random
import argparse
import hashlib
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem, QED, RDConfig
from scripts.exps.lead.docking.docking import DockingVina
from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import math


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Argparse flags that name where output goes rather than what is computed.
# Excluded from the run_id hash so relocating results does not rename the run.
_NON_SCIENTIFIC_FLAGS = frozenset({'out_dir', 'model_path', 'num_sub_proc', 'force'})


def compute_run_id(args):
    """Stable short hash over every flag that can change the result.

    Output paths used to be keyed on (oracle, start_mol_idx, sim_thr, seed)
    only, so every combination of strategy/pop_cap/ucb_c/lam_* at a given
    target and seed appended into one CSV with no column identifying the
    configuration. Hashing the whole namespace makes each grid point its own
    directory and keeps provenance in config.json rather than in a filename.
    """
    payload = {k: v for k, v in sorted(vars(args).items())
               if k not in _NON_SCIENTIFIC_FLAGS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]
    return f'{args.strategy}_{args.oracle_name}_id{args.start_mol_idx}_s{args.seed}_{digest}'


def seed_everything(seed):
    """--seed previously only named a file; nothing was ever seeded."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.run_id = compute_run_id(args)

        # df = pd.read_csv('scripts/exps/lead/docking/actives.csv')
        # Get the directory where the current script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(script_dir, 'docking', 'actives.csv'))
        df = df[df['target'] == self.args.oracle_name]
        self.start_smiles = df['smiles'].iloc[self.args.start_mol_idx]
        start_mol = Chem.MolFromSmiles(self.start_smiles)
        self.start_fp = AllChem.GetMorganFingerprintAsBitVect(start_mol, 2, 2048)
        self.start_prop = df['DS'].iloc[self.args.start_mol_idx]
        print(f'Start SMILES:\t{self.start_smiles}')
        print(f'Start DS:\t{self.start_prop}')

        # One directory per grid point. The results/ dir is gitignored, so it
        # may not exist on a fresh clone -- the guard that used to handle that
        # was dropped when the iterlog was added, making run.py crash at startup.
        self.run_dir = os.path.join(self.args.out_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.fname = os.path.join(self.run_dir, 'molecules.csv')
        self.iter_log_path = os.path.join(self.run_dir, 'iterlog.csv')
        self.status_path = os.path.join(self.run_dir, 'status.json')
        print(f'\033[92m{self.run_dir}\033[0m')

        with open(os.path.join(self.run_dir, 'config.json'), 'w') as f:
            json.dump({**vars(self.args),
                       'run_id': self.run_id,
                       'start_smiles': self.start_smiles,
                       'start_ds': float(self.start_prop)}, f, indent=2, default=str)

        self.predictor = DockingVina(self.args.oracle_name,
                                     num_sub_proc=self.args.num_sub_proc)
        self.population = [(self.start_prop, frag) for frag in cut(self.start_smiles)]
        print(f'Initial population: {len(self.population)} frags')
        self.sampler = Sampler(self.args.model_path)

        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        for prop, frag in self.population:
            self.Q[frag] = max(self.Q[frag], prop)
        self.alpha = self.args.q_alpha
        self.c = self.args.ucb_c
        self.t = 0

        with open(self.iter_log_path, 'w') as f:
            f.write('iter,strategy,rv_mean,rv_max,n_admitted,n_admitted_mols,'
                    'fail_rv,fail_rq,fail_rs,fail_rsim,top_ds\n')

    def q_update(self, frag, r):
        """Q update honouring --q_alpha (0 selects a true sample average).

        UCB1 is defined against a sample mean; the fixed EMA at 0.4 has an
        effective memory of ~2.5 pulls, which is very noisy when arms only
        receive ~10 pulls.
        """
        if self.alpha > 0:
            self.Q[frag] += self.alpha * (r - self.Q[frag])
        else:
            self.Q[frag] += (r - self.Q[frag]) / max(self.N[frag], 1)

    def init_new_frag(self, frag, admit_rv):
        """Seed Q for a fragment entering the population for the first time.

        Seed fragments get Q = the start molecule's docking score (~7-10), but
        newly discovered ones used to start at 0.0 from the defaultdict and
        needed ~4 pulls to reach parity. Since the population cap evicts the
        well-estimated seed fragments as soon as better-scoring molecules
        appear, the bandit ended up choosing almost entirely among arms it had
        no estimate for. q_init makes that a measurable axis.
        """
        if frag in self.N and self.N[frag] > 0:
            return                      # already has real statistics
        if self.args.q_init == 'zero':
            return                      # historical behaviour
        if self.args.q_init == 'admit_rv':
            self.Q[frag] = max(self.Q[frag], float(admit_rv))
        elif self.args.q_init == 'optimistic':
            self.Q[frag] = max(self.Q[frag], float(self.args.q_optimistic))
    
    def reward_vina(self, smiles_list):
        reward = - np.array(self.predictor.predict(smiles_list))
        reward = np.clip(reward, 0, None)
        return reward
    
    def reward_qed(self, mols):
        return [QED.qed(m) for m in mols]
    
    def reward_sa(self, mols):
        return [(10 - sascorer.calculateScore(m)) / 9 for m in mols]
    
    def reward_sim(self, mols):
        mol_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
        return DataStructs.BulkTanimotoSimilarity(self.start_fp, mol_fps)
        
    def reward(self, smiles_list):
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        rv = self.reward_vina(smiles_list)
        rq = self.reward_qed(mols)
        rs = self.reward_sa(mols)
        rsim = self.reward_sim(mols)
        return rv, rq, rs, rsim
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return mols[idx][0]
    
    def select_frag(self, candidates, exclude=None):
        if exclude is not None:
            candidates = [f for f in candidates if f != exclude]
        if not candidates:
            return None
        if self.args.strategy == 'egreedy':
            if random.random() < self.args.epsilon:
                return random.choice(candidates)
            return max(candidates, key=lambda f: self.Q[f])
        unpulled = [f for f in candidates if self.N[f] == 0]
        if unpulled:
            return random.choice(unpulled)
        logt = math.log(max(self.t, 2))
        return max(candidates,
                   key=lambda f: self.Q[f] + self.c * math.sqrt(logt / self.N[f]))
    
    def update_population(self, smiles_list, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        fails = {'rv': 0, 'rq': 0, 'rs': 0, 'rsim': 0}
        n_admitted_mols = 0
        n_added = 0
        for rv, rq, rs, rsim, smiles in zip(rv_list, rq_list, rs_list, rsim_list, smiles_list):
            if not rv > self.start_prop:
                fails['rv'] += 1
            elif not rq >= 0.6:
                fails['rq'] += 1
            elif not rs >= 6/9:
                fails['rs'] += 1
            elif not rsim >= self.args.sim_thr:
                fails['rsim'] += 1
            else:
                n_admitted_mols += 1
                frags = {frag for frag in cut(smiles)}
                n_added += len(frags)
                for frag in frags:
                    self.init_new_frag(frag, rv)
                self.population.extend([(rv, frag) for frag in frags])
        self.population.sort(reverse=True)
        if self.args.pop_cap > 0:
            seen, capped = set(), []
            for prop, frag in self.population:
                if frag not in seen:
                    seen.add(frag)
                    capped.append((prop, frag))
                if len(capped) >= self.args.pop_cap:
                    break
            self.population = capped
        return fails, n_admitted_mols, n_added

    def generate(self):
        frags = list({frag for _, frag in self.population})
        # random.sample(frags, 2) raises ValueError below 2 fragments, and
        # select_frag returns None, which attach() cannot consume.
        if len(frags) < 2:
            return None, (None, None)
        for _ in range(1000):
            if self.args.strategy == 'random':
                frag1, frag2 = random.sample(frags, 2)
            else:
                frag1 = self.select_frag(frags)
                frag2 = self.select_frag(frags, exclude=frag1)
            if frag1 is None or frag2 is None:
                return None, (None, None)
            smiles = Chem.MolToSmiles(self.attach(frag1, frag2))
            if smiles is None: continue
            smiles = self.sampler.mask_modification(smiles, min_len=50, gamma=self.args.gamma)
            if smiles is not None:
                smiles = sorted(smiles.split('.'), key=len)[-1]
            return smiles, (frag1, frag2)
        return None, (None, None)
            
    def record(self, smiles_list, prop_list):
        with open(self.fname, 'a') as f:
            for i in range(len(smiles_list)):
                str = f'{smiles_list[i]},'
                for props in prop_list: str += f'{props[i]},'
                str += '\n'
                f.write(str)
                
    def update_bandit(self, frag_pairs, prop_list):
        rv_list, rq_list, rs_list, rsim_list = prop_list
        for (frag1, frag2), rv, rq, rs, rsim in zip(
                frag_pairs, rv_list, rq_list, rs_list, rsim_list):
            if frag1 is None or frag2 is None:
                continue
            shortfall = (self.args.lam_rq * max(0.0, 0.6 - rq) / 0.6
                         + self.args.lam_rs * max(0.0, 6/9 - rs) / (6/9)
                         + self.args.lam_rsim * max(0.0, self.args.sim_thr - rsim)
                           / max(self.args.sim_thr, 1e-6))
            r = rv - shortfall
            for frag in (frag1, frag2):
                self.N[frag] += 1
                self.t += 1
                self.q_update(frag, r)

    def run(self):
        t_start = time()
        for i in range(self.args.num_iter):
            gen_results = [self.generate() for _ in range(self.args.num_gen)]
            smiles_list = [s for s, _ in gen_results]
            frag_pairs = [p for _, p in gen_results]

            prop_list = self.reward(smiles_list)
            rv = prop_list[0]

            n_before = len(self.population)
            # egreedy also needs Q maintained; only the random arm skips it.
            if self.args.strategy != 'random':
                self.update_bandit(frag_pairs, prop_list)
            fails, n_admitted_mols, n_added = self.update_population(smiles_list, prop_list)

            self.record(smiles_list, prop_list)

            arms = {frag for _, frag in self.population}
            n_unpulled = sum(1 for f in arms if self.N[f] == 0)
            print(f'  arms={len(arms)} unpulled={n_unpulled} t={self.t}')

            with open(self.iter_log_path, 'a') as f:
                f.write(f'{i+1},{self.args.strategy},{np.mean(rv):.4f},'
                        f'{np.max(rv):.4f},{n_added},{n_admitted_mols},'
                        f"{fails['rv']},{fails['rq']},{fails['rs']},{fails['rsim']},"
                        f'{self.population[0][0]:.4f}\n')

            print(f'[Iter {i+1:03d}] Top DS: {self.population[0][0]} | '
                  f'rv mean={np.mean(rv):.2f} max={np.max(rv):.2f} | '
                  f'admitted={n_added} mols={n_admitted_mols} | '
                  f"fail rv={fails['rv']} rq={fails['rq']} rs={fails['rs']} rsim={fails['rsim']}")
        elapsed = time() - t_start
        # Written last, and only on a clean finish. A task killed by walltime or
        # OOM leaves no status.json, which is how the launcher tells a finished
        # run from a truncated one when resuming.
        with open(self.status_path, 'w') as f:
            json.dump({'run_id': self.run_id,
                       'completed': True,
                       'num_iter': self.args.num_iter,
                       'elapsed_sec': round(elapsed, 2),
                       'final_top_ds': float(self.population[0][0]),
                       'start_ds': float(self.start_prop),
                       'improvement': float(self.population[0][0]) - float(self.start_prop)},
                      f, indent=2)
        print(f'{elapsed:.2f} sec elapsed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--oracle_name',      type=str,   default='parp1',
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
    parser.add_argument('-i', '--start_mol_idx',    type=int,   default=0, choices=[0, 1, 2])
    parser.add_argument('-d', '--sim_thr',          type=float, default=0.4)
    parser.add_argument('-s', '--seed',             type=int,   default=0)
    parser.add_argument('-m', '--model_path',       type=str,   default='model.ckpt')
    parser.add_argument('--num_gen',                type=int,   default=100)
    parser.add_argument('--num_iter',               type=int,   default=10)
    parser.add_argument('--gamma',                  type=float, default=0)
    parser.add_argument('--lam_rq',   type=float, default=1.0)
    parser.add_argument('--lam_rs',   type=float, default=1.0)
    parser.add_argument('--lam_rsim', type=float, default=1.0)
    parser.add_argument('--strategy', type=str, default='bandit',
                        choices=['random', 'bandit', 'egreedy'])
    parser.add_argument('--ucb_c', type=float, default=2.0)
    parser.add_argument('--pop_cap', type=int, default=150)
    # Previously read via getattr(args, 'q_alpha', 0.4) with no flag behind it.
    parser.add_argument('--q_alpha', type=float, default=0.4,
                        help='EMA rate for the Q update; 0 selects a true sample average')
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='exploration rate for --strategy egreedy')
    parser.add_argument('--q_init', type=str, default='zero',
                        choices=['zero', 'admit_rv', 'optimistic'],
                        help='Q seed for a newly discovered fragment; zero is historical')
    parser.add_argument('--q_optimistic', type=float, default=12.0,
                        help='Q seed used when --q_init optimistic')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(ROOT_DIR, 'results'))
    parser.add_argument('--num_sub_proc', type=int, default=None,
                        help='docking worker processes; defaults to $GENMOL_NUM_SUB_PROC or 10')
    parser.add_argument('--force', action='store_true',
                        help='re-run even if a completed status.json already exists')
    args = parser.parse_args()

    seed_everything(args.seed)

    run_id = compute_run_id(args)
    status = os.path.join(args.out_dir, run_id, 'status.json')
    if os.path.exists(status) and not args.force:
        print(f'SKIP {run_id}: already completed ({status})')
        sys.exit(0)

    GenMolOpt(args).run()