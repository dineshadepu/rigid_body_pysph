# -*- coding: utf-8 -*-
"""2D rigid body equations
"""
from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
import numpy as np
from math import sqrt
from pysph.sph.equation import Group
from pysph.sph.scheme import Scheme
from compyle.api import (elementwise, annotate, wrap, declare)
from compyle.low_level import (address)
from pysph.sph.wc.linalg import (mat_mult, mat_vec_mult, dot)
from numpy import sin, cos


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = d_m[d_idx]*self.gx
        d_fy[d_idx] = d_m[d_idx]*self.gy
        d_fz[d_idx] = d_m[d_idx]*self.gz
