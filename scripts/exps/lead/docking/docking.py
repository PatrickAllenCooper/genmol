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


# This file has been modified from MOOD.
#
# Source:
# https://github.com/SeulLee05/MOOD/blob/main/scorer/docking.py
#
# The license for the original version of this file can be
# found in LICENSE/3rd_party/LICENSE_MOOD.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import os
import tempfile
from shutil import rmtree
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Docking failure sentinel. reward_vina() negates and clips this to 0.
DOCK_FAILED = 99.9


def default_scratch_root():
    """Node-local scratch if SLURM gave us one, else the system temp dir.

    Never the repo checkout: it is shared between concurrent tasks.
    """
    for var in ('SLURM_TMPDIR', 'TMPDIR'):
        path = os.environ.get(var)
        if path and os.path.isdir(path):
            return path
    return tempfile.gettempdir()


class DockingVina(object):
    def __init__(self, target, temp_dir=None, num_sub_proc=None):
        super().__init__()

        if target == 'fa7':
            self.box_center = (10.131, 41.879, 32.097)
            self.box_size = (20.673, 20.198, 21.362)
        elif target == 'parp1':
            self.box_center = (26.413, 11.282, 27.238)
            self.box_size = (18.521, 17.479, 19.995)
        elif target == '5ht1b':
            self.box_center = (-26.602, 5.277, 17.898)
            self.box_size = (22.5, 22.5, 22.5)
        elif target == 'jak2':
            self.box_center = (114.758, 65.496, 11.345)
            self.box_size= (19.033, 17.929, 20.283)
        elif target == 'braf':
            self.box_center = (84.194, 6.949, -7.081)
            self.box_size = (22.032, 19.211, 14.106)
        
        self.vina_program = os.path.join(ROOT_DIR, 'docking/qvina02')
        self.receptor_file = os.path.join(ROOT_DIR, f'docking/{target}.pdbqt')
        self.exhaustiveness = 1
        # At exhaustiveness=1 qvina runs a single Monte Carlo task, so --cpu
        # cannot be utilised and each worker is effectively one core. Size the
        # pool against the allocation rather than the old hardcoded 10.
        if num_sub_proc is None:
            num_sub_proc = int(os.environ.get('GENMOL_NUM_SUB_PROC', 0)) or 10
        self.num_sub_proc = max(1, int(num_sub_proc))
        self.num_cpu_dock = 1
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100

        # The old check-then-create scan over docking/tmp/tmpN was a TOCTOU race
        # on a path shared by every concurrent task: os.makedirs has no
        # exist_ok, and the break sits after it, so the losing process raised an
        # uncaught FileExistsError out of __init__. mkdtemp is atomic and gives
        # each instance a private directory on node-local scratch.
        if temp_dir is None:
            self._owns_temp_dir = True
            self.temp_dir = tempfile.mkdtemp(prefix='genmol_dock_',
                                             dir=default_scratch_root())
        else:
            self._owns_temp_dir = False
            os.makedirs(temp_dir, exist_ok=True)
            self.temp_dir = temp_dir
        print(f'Docking tmp dir: {self.temp_dir}')

    def gen_3d(self, smi, ligand_mol_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_mol_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' %(self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' %(self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print(f'gen_3d unexpected error: {smi}')
                return_dict[idx] = DOCK_FAILED
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(f'docking unexpected error: {smi}')
                return_dict[idx] = DOCK_FAILED
                continue
            if len(affinity_list)==0:
                affinity_list.append(DOCK_FAILED)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        """
            input SMILES list
            output affinity list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()

        return self.collect_affinities(return_dict, len(smiles_list))

    @staticmethod
    def collect_affinities(return_dict, n):
        """Rebuild the affinity list positionally, tolerating missing indices.

        The previous implementation appended over sorted(return_dict.keys()).
        A worker killed outright (OOM, external signal) never writes its index,
        and that rebuild then returned a SHORT list in which every score past
        the gap silently belonged to the previous molecule -- and the zip() in
        update_population truncated instead of raising. Indexing into a
        preallocated list keeps SMILES and scores aligned by construction.
        """
        affinity_list = [DOCK_FAILED] * n
        missing = 0
        for idx in range(n):
            if idx in return_dict:
                affinity_list[idx] = return_dict[idx]
            else:
                missing += 1
        if missing:
            print(f'WARNING: {missing}/{n} docking results missing '
                  f'(worker died); scored as {DOCK_FAILED}')
        return affinity_list
    
    def __del__(self):
        # getattr, not self.temp_dir: a failure earlier in __init__ leaves the
        # attribute unset, and an AttributeError raised during teardown buries
        # the real traceback.
        temp_dir = getattr(self, 'temp_dir', None)
        if getattr(self, '_owns_temp_dir', False) and temp_dir \
                and os.path.exists(temp_dir):
            rmtree(temp_dir, ignore_errors=True)
