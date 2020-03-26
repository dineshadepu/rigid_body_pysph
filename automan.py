#!/usr/bin/env python
import os
from itertools import cycle, product

from automan.api import PySPHProblem as Problem
from automan.api import (Automator, Simulation, filter_cases,
                         )
from pysph.solver.utils import load
import numpy as np
import matplotlib
from scipy.signal import savgol_filter
matplotlib.use('agg')


class Case1(Problem):
    def get_name(self):
        return 'case1'

    def setup(self):
        self.plotdata = {}
        self.cl = {}
        self.cd = {}
        self.st = {}
        get_path = self.input_path
        self.umax = 1.0
        self.rho = 1000
        self.re = re = [200]
        self.dc = 2.0
        self.nx = 30
        self.tf = 1.
        pfreq = 400
        cmd = 'python case_1.py ' + backend

        _case_info = {
            'donothing': dict(io_method='donothing'),
            'mirror': dict(io_method='mirror'),
            'hybrid': dict(io_method='hybrid'),
            'mod_donothing': dict(io_method='mod_donothing'),
            'characteristic': dict(io_method='characteristic'),
        }

        self.case_info = {
            f'{method}_re_{r}_nx_30': dict(
                re=r, **_case_info[method],
                dc=self.dc, nx=self.nx, tf=self.tf, pfreq=self.nx*10
            )
            for method, r in product(_case_info, re)
        }

        # cases with different reynolds number
        self.case_info['hybrid_re_20_nx_40'] = dict(
                re=20, io_method='hybrid',
                dc=self.dc, nx=40, tf=100, pfreq=400
            )
        self.case_info['mod_donothing_re_20_nx_40'] = dict(
                re=20, io_method='mod_donothing',
                dc=self.dc, nx=40, tf=100, pfreq=400
            )
        self.case_info['hybrid_re_20_nx_20'] = dict(
                re=20, io_method='hybrid',
                dc=self.dc, nx=20, tf=100, pfreq=200
            )
        self.case_info['hybrid_re_20_nx_30'] = dict(
                re=20, io_method='hybrid',
                dc=self.dc, nx=30, tf=100, pfreq=300
            )
        self.case_info['mod_donothing_re_20_nx_20'] = dict(
                re=20, io_method='mod_donothing',
                dc=self.dc, nx=20, tf=100, pfreq=200
            )
        self.case_info['mod_donothing_re_20_nx_30'] = dict(
                re=20, io_method='mod_donothing',
                dc=self.dc, nx=30, tf=100, pfreq=300
            )

        self.cases = [
            Simulation(
                get_path(name), cmd, job_info=dict(n_core=12, n_thread=48),
                cache_nnps=None, **scheme_opts(Kwargs)
            ) for name, Kwargs in self.case_info.items()
        ]

        for case in self.cases:
            self.case_info[case.name]['case'] = case

    def run(self):
        self.make_output_dir()
        self._plot_cl_all_methods_re_200()
        self._get_cdcl_values()
        self._get_St_values()
        self._plot_convergence_re20()
        self._plot_cd_cl_hybrid_all()
        self._table_compare_cd_cl_st_all()
        self._plot_particles()
