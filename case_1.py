"""A cube, translating and rotating freely without the influence of gravity.
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


def create_an_unsymmetric_body(dx, l, b, h):
    # create a rigid body
    x, y, z = get_3d_block(dx, l, b, h, center=[l/2., b/2., h/2.])
    fltr = (x > l/4.) & (y < b/4.) & (z < h/6.)

    x = x[~fltr]
    y = y[~fltr]
    z = z[~fltr]

    return x, y, z


class Case1(Application):
    def initialize(self):
        self.rho0 = 10.0
        self.hdx = 1.3
        self.dx = 0.02
        self.dy = 0.02
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 3

    def create_scheme(self):
        rbs = RigidBodyScheme(rigid_bodies=['body'], boundaries=None, dim=3,
                              orientation="quaternion",
                              principal_moi=True,
                              kn=self.kn, mu=self.mu, en=self.en)
        s = SchemeChooser(default='rbs', rbs=rbs)
        return s

    def configure_scheme(self):
        scheme = self.scheme
        kernel = CubicSpline(dim=self.dim)
        dt = 1e-3
        tf = 5.
        # tf = dt scheme.configure()
        scheme.configure_solver(kernel=kernel, integrator_cls=EPECIntegrator,
                                dt=dt, tf=tf)

    def create_particles(self):
        nx, ny, nz = 10, 10, 10
        dx = self.dx
        # x, y, z = create_an_unsymmetric_body(self.dx * 2., 0.5, 0.5, 1.)

        x, y, z = np.mgrid[0:1:nx * 1j, 0:1:ny * 1j, 0:1:nz * 1j]
        x = x.flat
        y = y.flat
        z = z.flat

        m = np.ones_like(x) * dx * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx/4.

        body = get_particle_array(name='body', x=x, y=y, z=z, h=h, m=m)

        body_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int')
        body.body_id[:] = body_id[:]

        # setup_rigid_body_unconstrained_dynamics(body, principal_moi=self.principal_moi)
        self.scheme.setup_properties([body])

        setup_rigid_body_collision_dynamics(body, rad_s)

        add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')

        # body.vc[0] = 0.5
        # body.vc[1] = 0.5
        body.omega[2] = 5.

        # this must run
        set_angular_momentum([body])

        # from IPython import embed; embed()

        return [body]


if __name__ == '__main__':
    app = Case1()
    app.run()

    # set_trace()
