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
                        setup_child_body,
                        RigidBodyScheme,
                        SumUpExternalForces,
                        SumUpExternalForcesChild,
                        RK2StepRigidBodyQuaternions,
                        RK2StepRigidBodyQuaternionsOptimized,
                        ChildBodyStep)

from rigid_body_setup import (set_angular_momentum)
from pysph.examples.solid_mech.impact import add_properties
from pysph.tools.geometry import get_3d_block, show_3d


# PySPH base and carray imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group

from pysph.sph.wc.linalg import (mat_mult, mat_vec_mult, dot)
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision)

from pysph.examples.elliptical_drop import EllipticalDrop as EDScheme


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

        setup_rigid_body_unconstrained_dynamics(body,
                                                principal_moi=False)

        setup_rigid_body_collision_dynamics(body, rad_s)

        add_properties(body, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')

        # body.vc[0] = 0.5
        # body.vc[1] = 0.5
        body.omega[1] = 5.

        # this must run
        set_angular_momentum([body])

        # create child body
        x, y, z = np.mgrid[-2. * dx:1+2. * dx:nx * 1j, -2. * dx:1+2. * dx:ny * 1j, -2. * dx:1+2. * dx:nz * 1j]
        x = x.flat
        y = y.flat
        z = z.flat

        rad_s = np.ones_like(x) * dx/4.
        child_body = get_particle_array(name='child_body', x=x, y=y, z=z, h=h, m=m)
        body_id = np.zeros(len(x), dtype=int)
        child_body.add_property('body_id', type='int')
        child_body.body_id[:] = body_id[:]
        setup_child_body(child_body, body)

        setup_rigid_body_collision_dynamics(child_body, rad_s)

        add_properties(child_body, 'tang_velocity_z', 'tang_disp_y',
                       'tang_velocity_x', 'tang_disp_x', 'tang_velocity_y',
                       'tang_disp_z')

        # from IPython import embed; embed()

        return [body, child_body]

    def create_equations(self):
        equations = [
            # Group(equations=[BodyForce(dest='body', sources=None, gx=0.0, gy=0.0, gz=0.0)]),
            Group(equations=[BodyForce(dest='child_body', sources=None, gx=0.0, gy=0., gz=-9.81)]),
            # Group(equations=[
                # RigidBodyCollision(dest='body', sources=['body', 'child_body'], kn=10000.0, mu=0.5),
                # RigidBodyCollision(dest='child_body', sources=['body', 'child_body'], kn=10000.0, mu=0.5)
            # ],),
            Group(equations=[
                SumUpExternalForces(dest='body', sources=None)
            ],
            ),
            Group(equations=[
                SumUpExternalForcesChild(dest='child_body', sources=None)
            ],)]
        return equations

    def create_solver(self):
        kernel = CubicSpline(dim=self.dim)
        dt = 1e-4
        tf = 1.

        integrator = EPECIntegrator(
            body=RK2StepRigidBodyQuaternions(),
            child_body=ChildBodyStep())

        solver = Solver(kernel=kernel, dim=self.dim, integrator=integrator,
                        dt=dt, tf=tf)

        return solver



if __name__ == '__main__':
    app = Case1()
    app.run()
