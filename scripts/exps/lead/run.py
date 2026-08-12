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


class GenMolOpt():
    def __init__(self, args):
        super().__init__()
        self.args = args

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

        self.predictor = DockingVina(self.args.oracle_name)
        self.population = [(self.start_prop, frag) for frag in cut(self.start_smiles)]
        print(f'Initial population: {len(self.population)} frags')
        self.sampler = Sampler(self.args.model_path)

        self.fname = f'results/{self.args.oracle_name}_id{self.args.start_mol_idx}_' + \
                     f'thr{self.args.sim_thr}_{self.args.seed}.csv'
        print(f'\033[92m{self.fname}\033[0m')
        self.fname = os.path.join(ROOT_DIR, self.fname)


        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        for prop, frag in self.population:
            self.Q[frag] = max(self.Q[frag], prop)
        self.alpha = getattr(self.args, 'q_alpha', 0.4)
        self.c = self.args.ucb_c
        self.t = 0  

        tag = self.args.strategy
        if self.args.strategy == 'bandit':
            tag += (f'-ucb{self.args.ucb_c}-cap{self.args.pop_cap}'
                    f'-lamq{self.args.lam_rq}')
        self.iter_log_path = os.path.join(
            ROOT_DIR, 'results',
            f'iterlog_{tag}_{self.args.oracle_name}_'
            f'id{self.args.start_mol_idx}_{self.args.seed}.csv'
        )
        with open(self.iter_log_path, 'w') as f:
            f.write('iter,strategy,rv_mean,rv_max,n_admitted,n_admitted_mols,'
                    'fail_rv,fail_rq,fail_rs,fail_rsim,top_ds\n')
    
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
        for _ in range(1000):
            if self.args.strategy == 'bandit':
                frag1 = self.select_frag(frags)
                frag2 = self.select_frag(frags, exclude=frag1)
            else:
                frag1, frag2 = random.sample(frags, 2)
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
                self.Q[frag] += self.alpha * (r - self.Q[frag])

    def run(self):
        t_start = time()
        for i in range(self.args.num_iter):
            gen_results = [self.generate() for _ in range(self.args.num_gen)]
            smiles_list = [s for s, _ in gen_results]
            frag_pairs = [p for _, p in gen_results]

            prop_list = self.reward(smiles_list)
            rv = prop_list[0]

            n_before = len(self.population)
            if self.args.strategy == 'bandit':
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
        print(f'{time() - t_start:.2f} sec elapsed')

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
    parser.add_argument('--strategy', type=str, default='bandit', choices=['random', 'bandit'])
    parser.add_argument('--ucb_c', type=float, default=2.0)
    parser.add_argument('--pop_cap', type=int, default=150)
    args = parser.parse_args()

    GenMolOpt(args).run()