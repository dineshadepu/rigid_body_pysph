"""A cube translating and rotating freely without the influence of gravity.
This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from rigid_body_pysph.rigid_body_3d import (
    RigidBody3DScheme,
    setup_rigid_body,
    set_linear_velocity,
    set_angular_velocity,
    get_master_and_slave_rb,
    add_contact_properties_body_master)
from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def initialize(self):
        self.rho0 = 2700.0
        self.hdx = 1.0
        self.dx = 0.1
        self.dy = 0.1
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

        self.dt = 1e-3
        self.tf = np.pi

    def create_particles(self):
        from pysph.tools.geometry import get_2d_block
        dx = self.dx
        # ===========================================
        # Create rigid body particle array
        # ===========================================
        x, y = get_2d_block(dx, 1., 1.)
        x = x.flat
        y = y.flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                  rho=self.rho0,
                                  rad_s=rad_s,
                                  E=69 * 1e9,
                                  nu=0.3)
        body_id = np.zeros(len(x), dtype=int)
        dem_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)
        # body.add_constant('total_no_bodies', [1])

        # setup the properties
        setup_rigid_body(body, self.dim)

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        set_linear_velocity(body, np.array([1., 1., 0.]))
        set_angular_velocity(body, np.array([0., 0., 2. * np.pi]))

        body_master, body_slave = get_master_and_slave_rb(body)
        add_contact_properties_body_master(body_master, 6, 3)
        body_master.rad_s[:] = 1. / 2.
        body_master.add_output_arrays(['R'])
        # ===========================================
        # Create rigid body particle array ends
        # ===========================================

        return [body_master, body_slave]

    def create_scheme(self):
        rb3d_ms = RigidBody3DScheme(rigid_bodies_master=['body_master'],
                                    rigid_bodies_slave=['body_slave'],
                                    boundaries=None, dim=self.dim)
        s = SchemeChooser(default='rb3d_ms', rb3d_ms=rb3d_ms)
        return s

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=10)

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[:]
        t, total_energy = [], []
        x, y = [], []
        R = []
        ang_mom = []
        for sd, body in iter_output(files, 'body_master'):
            _t = sd['t']
            # print(_t)
            t.append(_t)
            total_energy.append(0.5 * np.sum(body.m[:] * (body.u[:]**2. +
                                                          body.v[:]**2.)))
            R.append(body.R[0])
            # print("========================")
            # print("R is", body.R)
            # # print("ang_mom x is", body.ang_mom_x)
            # # print("ang_mom y is", body.ang_mom_y)
            # print("ang_mom z is", body.ang_mom_z)
            # # print("omega x is", body.omega_x)
            # # print("omega y is", body.omega_y)
            # print("omega z is", body.omega_z)
            # print("moi global master ", body.inertia_tensor_inverse_global_frame)
            # # print("moi body master ", body.inertia_tensor_inverse_body_frame)
            # # print("moi global master ", body.inertia_tensor_global_frame)
            # # print("moi body master ", body.inertia_tensor_body_frame)
            # # x.append(body.xcm[0])
            # # y.append(body.xcm[1])
            # # print(body.ang_mom_z[0])
            ang_mom.append(body.ang_mom_z[0])
        # print(ang_mom)

        import matplotlib
        import os
        # matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t=t, R_0_val=R)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        # plt.plot(t, total_energy, "-", label='Simulated')
        # plt.plot(t, ang_mom, "-", label='Angular momentum')
        plt.plot(t, R, "-", label='R[0]')

        plt.xlabel('t')
        plt.ylabel('ang energy')
        plt.legend()
        fig = os.path.join(self.output_dir, "ang_mom_vs_t.png")
        plt.savefig(fig, dpi=300)
        # plt.show()

        # plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = Case0()
    app.run()
    app.post_process(app.info_filename)
