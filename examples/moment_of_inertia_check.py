"""We check the current moment of inertia code by computing it on different
bodies

https://en.wikipedia.org/wiki/List_of_moments_of_inertia
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
from pysph.tools.geometry import get_2d_block
from pysph_rfc_new.geometry import create_circle_1


def square_body():
    dx = 0.001
    # ===========================================
    # Create rigid body particle array
    # ===========================================
    rho = 2000.
    x, y = get_2d_block(dx, 1., 1.)
    x = x.flat
    y = y.flat
    m = np.ones_like(x) * dx * dx * rho
    h = np.ones_like(x) * 1.2 * dx
    # radius of each sphere constituting in cube
    rad_s = np.ones_like(x) * dx
    body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                rho=rho,
                                rad_s=rad_s,
                                E=69 * 1e9,
                                nu=0.3)
    body_id = np.zeros(len(x), dtype=int)
    dem_id = np.zeros(len(x), dtype=int)
    body.add_property('body_id', type='int', data=body_id)
    body.add_property('dem_id', type='int', data=dem_id)
    # body.add_constant('total_no_bodies', [1])

    # setup the properties
    setup_rigid_body(body, 2)

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
    print("A square body should have a izz of 333.33 and a total mass of 2000")
    print("The computed values are ")
    print("mass is", body_master.m_b)
    print("izz is", body_master.izz)

    return [body_master, body_slave]


def circular_body():
    dx = 0.001
    # ===========================================
    # Create rigid body particle array
    # ===========================================
    rho = 2000.
    x, y = create_circle_1(1., dx)
    x = x.flat
    y = y.flat
    m = np.ones_like(x) * dx * dx * rho
    h = np.ones_like(x) * 1.2 * dx
    # radius of each sphere constituting in cube
    rad_s = np.ones_like(x) * dx
    body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                              rho=rho,
                              rad_s=rad_s,
                              E=69 * 1e9,
                              nu=0.3)
    body_id = np.zeros(len(x), dtype=int)
    dem_id = np.zeros(len(x), dtype=int)
    body.add_property('body_id', type='int', data=body_id)
    body.add_property('dem_id', type='int', data=dem_id)
    # body.add_constant('total_no_bodies', [1])

    # setup the properties
    setup_rigid_body(body, 2)

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
    print("A circular body should have a izz of 196 and a total mass of 1570")
    print("The computed values are ")
    print("mass is", body_master.m_b)
    print("izz is", body_master.izz)

    return [body_master, body_slave]


square_body()
circular_body()
