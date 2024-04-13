#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import cycle, product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


# n_core = 32
# n_thread = 32 * 2
n_core = 6
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class SingleBodyFreelyMoving01(Problem):
    def get_name(self):
        return '01_single_body_freely_moving'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/01_single_body_freely_moving.py' + backend

        # Base case info
        self.case_info = {
            '2d': (dict(
                rb_evolve="2d",
                ), '2D formulation'),

            '3d_rot': (dict(
                rb_evolve="3d_rot",
                ), '3d rotation formulation'),

            '3d_quat': (dict(
                rb_evolve="3d_quat",
                ), '3d quaternion formulation'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_variation_of_rotation_matrix()

    def plot_variation_of_rotation_matrix(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t = data[name]['t']
            R_0_val = data[name]['R_0_val']

            plt.plot(t, R_0_val, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('R[0] value')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('R_0_val_vs_t.pdf'))
        plt.clf()
        plt.close()


class FourBodiesFreelyMoving02(Problem):
    def get_name(self):
        return '02_four_bodies_freely_moving'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/02_four_bodies_freely_moving.py' + backend

        # Base case info
        self.case_info = {
            '2d': (dict(
                rb_evolve="2d",
                ), '2D formulation'),

            '3d_rot': (dict(
                rb_evolve="3d_rot",
                ), '3d rotation formulation'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_variation_of_rotation_matrix()

    def plot_variation_of_rotation_matrix(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        for name in self.case_info:
            t = data[name]['t']
            R_0_val = data[name]['R_0_val']

            plt.plot(t, R_0_val, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('R[0] value')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('R_0_val_vs_t.pdf'))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    PROBLEMS = [
        SingleBodyFreelyMoving01,
        FourBodiesFreelyMoving02
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
    # Extra notes
    # Peng-Nan Sun, An accurate FSI-SPH

# contact_force_normal_x[0::2], contact_force_normal_y[0::2], contact_force_normal_z[0::2]
# contact_force_normal_x[1::2], contact_force_normal_y[1::2], contact_force_normal_z[1::2]
# au_contact, av_contact, aw_contact

        # vyas_2021_rebound_kinematics_3d_compare_flipped(),  # Done
