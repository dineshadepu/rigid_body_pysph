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
from pysph.tools.geometry import get_3d_block, get_2d_tank, show_3d

def rotate_body(x, y, theta):
    ra = theta * np.pi / 180
    x1 = x * np.cos(ra) - y * np.sin(ra)
    y1 = x * np.sin(ra) + y * np.cos(ra)
    return x1, y1


class Case3(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.gy = -9.81
        self.dim = 2

    def create_scheme(self):
        rbs = RigidBodyScheme(rigid_bodies=['body'], boundaries=['tank'],
                              dim=self.dim, orientation="quaternion",
                              principal_moi=True, kn=self.kn, mu=self.mu,
                              en=self.en, gy=self.gy)
        s = SchemeChooser(default='rbs', rbs=rbs)
        return s

    def create_particles(self):
        nx, ny = 10, 10
        dx = 1.0 / (nx - 1)
        x, y = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j]
        x = x.flat
        y = y.flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx / 2.


        body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                   rad_s=rad_s)
        body_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int')
        body.body_id[:] = body_id[:]

        self.scheme.setup_properties([body])

        setup_rigid_body_collision_dynamics(body, rad_s)

        add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')



        body.vc[0] = -3.0
        body.vc[1] = -3.0
        body.omega[2] = 1.0


        # this must run
        set_angular_momentum([body])

        # Create the tank.
        x, y = get_2d_tank(dx, base_center=[1., -2.], length=5.0, height=5.0,
                           num_layers=3)
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx

        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        tank = get_particle_array(name='tank', x=x, y=y, h=h, m=m,
                                  rad_s=rad_s)
        # tank.total_mass[0] = np.sum(m)


        return [body, tank]

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
        b = particle_arrays['body']
        viewer.scalar = 'x'
        b.show_legend = True
        ''')


if __name__ == '__main__':
    app = Case3()
    app.run()
