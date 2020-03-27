"""Freely rotating rigid plate hits another steady rigid plate. (7 seconds)

This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import (get_particle_array_rigid_body,
                              get_particle_array)
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import add_bool_argument
from pysph.sph.scheme import SchemeChooser
from rigid_body import (setup_rigid_body_unconstrained_dynamics,
                        setup_rigid_body_collision_dynamics,
                        RigidBodyScheme)
from rigid_body_setup import (set_angular_momentum)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_3d_block, show_3d

def rotate_body(x, y, theta):
    ra = theta * np.pi / 180
    x1 = x * np.cos(ra) - y * np.sin(ra)
    y1 = x * np.sin(ra) + y * np.cos(ra)
    return x1, y1


class Case2(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

    def create_scheme(self):
        rbs = RigidBodyScheme(rigid_bodies=['body1', 'body2'], boundaries=None,
                              dim=self.dim, orientation="quaternion",
                              principal_moi=True, kn=self.kn, mu=self.mu,
                              en=self.en)
        s = SchemeChooser(default='rbs', rbs=rbs)
        return s

    def create_particles(self):
        dx = 1.0 / 9.0
        x, y = np.mgrid[-0.5:0.7:dx, 0:0.2:dx]
        x = x.ravel()
        y = y.ravel()
        x, y = rotate_body(x, y, 45)
        x = x - 0.2
        y = y - 0.8
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        rad_s = np.ones_like(x) * dx / 2.
        body1 = get_particle_array(name='body1', x=x, y=y, h=h, m=m,
                                   rad_s=rad_s)
        body_id = np.zeros(len(x), dtype=int)
        body1.add_property('body_id', type='int')
        body1.body_id[:] = body_id[:]
        setup_rigid_body_collision_dynamics(body1, rad_s)
        add_properties(body1, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')


        dx = 1.0 / 9.0
        x, y = np.mgrid[0.3:1.3:dx, 0:0.2:dx]
        x = x.ravel()
        y = y.ravel()
        # x, y = rotate_body(x, y, -45)
        y = y - 1.
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        rad_s = np.ones_like(x) * dx
        body2 = get_particle_array(name='body2', x=x, y=y, h=h, m=m,
                                   rad_s=rad_s)
        body_id = np.zeros(len(x), dtype=int)
        body2.add_property('body_id', type='int')
        body2.body_id[:] = body_id[:]

        self.scheme.setup_properties([body1, body2])

        setup_rigid_body_collision_dynamics(body2, rad_s)

        add_properties(body2, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')

        body1.omega[2] = -3.


        # this must run
        set_angular_momentum([body1, body2])

        return [body1, body2]

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        tf = 5.
        dt = 1e-3
        scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['body1']
        viewer.scalar = 'x'
        b.show_legend = True
        ''')


if __name__ == '__main__':
    app = Case2()
    app.run()
