"""Rigid body related equations.
"""
from pysph.base.reduce_array import parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
import numpy as np
import numpy
from math import sqrt
from pysph.sph.equation import Group
from pysph.sph.scheme import Scheme, add_bool_argument
from rigid_body_setup import (
    set_total_mass,
    set_center_of_mass,
    set_moment_of_inertia,
    set_mi_in_body_frame_optimized,
    set_body_frame_position_vectors,

    normalize_R_orientation,
    normalize_q_orientation,
    quaternion_to_matrix,
    quaternion_multiplication)

from compyle.api import (elementwise, annotate, wrap, declare)
from compyle.low_level import (address)
from pysph.sph.wc.linalg import (mat_mult, mat_vec_mult, dot)
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision)
from numpy import sin, cos

import numpy as np


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        cm = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        cm = dst.cm
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            frc[i3] += fx[j]
            frc[i3+1] += fy[j]
            frc[i3+2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - cm[i3]
            dy = y[j] - cm[i3+1]
            dz = z[j] - cm[i3+2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3+1] += (dz * fx[j] - dx * fz[j])
            trq[i3+2] += (dx * fy[j] - dy * fx[j])


def setup_rigid_body_unconstrained_dynamics(pa, principal_moi):
    """
    This function will add all the properties regarding unconstrained
    rigid body dynamics.

    The following schemes will be implemented

    1. Motion with Direction cosine matrices
    2. using quaternions
    3. Using principal moment of inertia for both dcm and quaternion
    """
    body_id = pa.body_id

    nb = np.max(body_id) + 1

    # first add all the rigid body properties
    # Every other rigid body scheme or implementation must
    # have to be based on these following properties
    pa.add_constant("total_mass", np.zeros(nb))
    # moment of inetria inverse in body frame
    pa.add_constant("mib", np.zeros(9*nb))
    # moment of inetria inverse in global frame
    pa.add_constant("mig", np.zeros(9*nb))
    # moment of inetria in global frame
    pa.add_constant("moig", np.zeros(9*nb))

    # moment of inetria inverse in principal body frame
    pa.add_constant("mibp", np.zeros(3*nb))

    pa.add_constant("cm", np.zeros(3*nb))
    pa.add_constant("vc", np.zeros(3*nb))
    pa.add_constant("omega", np.zeros(3*nb))
    pa.add_constant("ang_mom", np.zeros(3*nb))
    pa.add_constant("force", np.zeros(3*nb))
    pa.add_constant("torque", np.zeros(3*nb))

    pa.add_constant("cm0", np.zeros(3*nb))
    pa.add_constant("vc0", np.zeros(3*nb))
    pa.add_constant("omega0", np.zeros(3*nb))
    pa.add_constant("ang_mom0", np.zeros(3*nb))

    # position of particles in local frame
    pa.add_property('dx0')
    pa.add_property('dy0')
    pa.add_property('dz0')

    # total no of rigid bodies
    pa.add_constant("nb", nb)

    # if the rigid body uses DCM then we need following property
    pa.add_constant("R", [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb)
    pa.add_constant("R0", [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb)

    # if the rigid body uses quaternion
    pa.add_constant("q", [1., 0., 0., 0.] * nb)
    pa.add_constant("q0", [1., 0., 0., 0.] * nb)

    # Find total mass
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_moment_of_inertia(pa)

    if principal_moi == True:
        set_mi_in_body_frame_optimized(pa)

    set_body_frame_position_vectors(pa)


def setup_rigid_body_collision_dynamics(pa, rad_s):
    """
    This function will add all the properties regarding collision of
    rigid body.

    The following schemes will be implemented

    1. DEM 3d
    """
    pa.add_property('fx')
    pa.add_property('fy')
    pa.add_property('fz')

    pa.add_property('rad_s')
    pa.rad_s[:] = rad_s

    pa.add_output_arrays(['fx', 'fy', 'fz'])


def setup_child_body(child_body, body):
    body_id = child_body.body_id

    nb = np.max(body_id) + 1
    # total no of rigid bodies
    child_body.add_constant("nb", nb)


    # position of particles in local frame
    child_body.add_property('dx0')
    child_body.add_property('dy0')
    child_body.add_property('dz0')


    child_body.constants['total_mass'] = body.constants['total_mass']
    child_body.constants['cm'] = body.constants['cm']
    child_body.constants['vc'] = body.constants['vc']
    child_body.constants['R'] = body.constants['R']
    child_body.constants['omega'] = body.constants['omega']
    child_body.constants['force'] = body.constants['force']
    child_body.constants['torque'] = body.constants['torque']

    set_body_frame_position_vectors(child_body)


class SumUpExternalForcesChild(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        cm = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        cm = dst.cm
        body_id = dst.body_id

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            frc[i3] += fx[j]
            frc[i3+1] += fy[j]
            frc[i3+2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = x[j] - cm[i3]
            dy = y[j] - cm[i3+1]
            dz = z[j] - cm[i3+2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3+1] += (dz * fx[j] - dx * fz[j])
            trq[i3+2] += (dx * fy[j] - dy * fx[j])


class RK2StepRigidBodyRotationMatrices(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                # save the current angular momentum
                dst.ang_mom0[3*i+j] = dst.ang_mom[3*i+j]
                # dst.omega0[3*i+j] = dst.omega[3*i+j]

            # save the current orientation
            for j in range(9):
                dst.R0[9*i+j] = dst.R[9*i+j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm[i3+j] + dtb2 * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]
            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dtb2

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.mib[i9:i9+9].reshape(3, 3))
            dst.mig[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]
            # move angular momentum to t + dt/2.
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + dst.torque[i3:i3+3] * dtb2

            tmp = dst.ang_mom[i3:i3+3]

            omega = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = omega[:]

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[i3+0] + dx
        d_y[d_idx] = d_cm[i3+1] + dy
        d_z[d_idx] = d_cm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vc[i3+0] + du
        d_v[d_idx] = d_vc[i3+1] + dv
        d_w[d_idx] = d_vc[i3+2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm0[i3+j] + dt * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]
            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # update the moment of inertia
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(R, dst.mib[i9:i9+9].reshape(3, 3))
            dst.mig[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()[:]
            # move angular momentum to t + dt/2.
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + dst.torque[i3:i3+3] * dt

            tmp = dst.ang_mom[i3:i3+3]

            omega = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = omega[:]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_cm, d_vc, d_R, d_omega, d_body_id):
        # some variables to update the positions seamlessly
        bid, i9, i3 = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9+0] * d_dx0[d_idx] + d_R[i9+1] * d_dy0[d_idx] +
              d_R[i9+2] * d_dz0[d_idx])
        dy = (d_R[i9+3] * d_dx0[d_idx] + d_R[i9+4] * d_dy0[d_idx] +
              d_R[i9+5] * d_dz0[d_idx])
        dz = (d_R[i9+6] * d_dx0[d_idx] + d_R[i9+7] * d_dy0[d_idx] +
              d_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_cm[i3+0] + dx
        d_y[d_idx] = d_cm[i3+1] + dy
        d_z[d_idx] = d_cm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vc[i3+0] + du
        d_v[d_idx] = d_vc[i3+1] + dv
        d_w[d_idx] = d_vc[i3+2] + dw


class RK2StepRigidBodyQuaternions(RK2StepRigidBodyRotationMatrices):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                # save the current angular momentum
                dst.ang_mom0[3*i+j] = dst.ang_mom[3*i+j]
                # dst.omega0[3*i+j] = dst.omega[3*i+j]

            # save the current orientation
            for j in range(4):
                dst.q0[4*i+j] = dst.q[4*i+j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i4 = 4 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm[i3+j] + dtb2 * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # change in quaternion
            delta_quat = np.array([0., 0., 0., 0.])
            # angular velocity magnitude
            omega_magn = sqrt(dst.omega[i3]**2 + dst.omega[i3+1]**2 +
                              dst.omega[i3+2]**2)
            axis_rot = np.array([0., 0., 0.])
            if omega_magn > 1e-12:
                axis_rot = dst.omega[i3:i3+3] / omega_magn
            delta_quat[0] = cos(omega_magn * dtb2 * 0.5)
            delta_quat[1] = axis_rot[0] * sin(omega_magn * dtb2 * 0.5)
            delta_quat[2] = axis_rot[1] * sin(omega_magn * dtb2 * 0.5)
            delta_quat[3] = axis_rot[2] * sin(omega_magn * dtb2 * 0.5)

            res = np.array([0., 0., 0., 0.])
            quaternion_multiplication(dst.q[i4:i4+4], delta_quat, res)
            dst.q[i4:i4+4] = res

            # normalize the orientation
            normalize_q_orientation(dst.q[i4:i4+4])

            # update the moment of inertia
            quaternion_to_matrix(dst.q[i4:i4+4], dst.R[i9:i9+9])
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.T
            tmp = np.matmul(R, dst.mib[i9:i9+9].reshape(3, 3))
            dst.mig[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()
            # move angular momentum to t + dt/2.
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + dst.torque[i3:i3+3] * dtb2

            tmp = dst.ang_mom[i3:i3+3]

            omega = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = omega[:]

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i4 = 4 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm0[i3+j] + dt * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # change in quaternion
            delta_quat = np.array([0., 0., 0., 0.])
            # angular velocity magnitude
            omega_magn = sqrt(dst.omega[i3]**2 + dst.omega[i3+1]**2 +
                              dst.omega[i3+2]**2)
            axis_rot = np.array([0., 0., 0.])
            if omega_magn > 1e-12:
                axis_rot = dst.omega[i3:i3+3] / omega_magn
            delta_quat[0] = cos(omega_magn * dt * 0.5)
            delta_quat[1] = axis_rot[0] * sin(omega_magn * dt * 0.5)
            delta_quat[2] = axis_rot[1] * sin(omega_magn * dt * 0.5)
            delta_quat[3] = axis_rot[2] * sin(omega_magn * dt * 0.5)

            res = np.array([0., 0., 0., 0.])
            quaternion_multiplication(dst.q0[i4:i4+4], delta_quat, res)
            dst.q[i4:i4+4] = res

            # normalize the orientation
            normalize_q_orientation(dst.q[i4:i4+4])

            # update the moment of inertia
            quaternion_to_matrix(dst.q[i4:i4+4], dst.R[i9:i9+9])
            R = dst.R[i9:i9+9].reshape(3, 3)
            R_t = R.T
            tmp = np.matmul(R, dst.mib[i9:i9+9].reshape(3, 3))
            dst.mig[i9:i9+9] = (np.matmul(tmp, R_t)).ravel()
            # move angular momentum to t + dt/2.
            dst.ang_mom[i3:i3+3] = dst.ang_mom0[i3:i3+3] + dst.torque[i3:i3+3] * dt

            tmp = dst.ang_mom[i3:i3+3]

            omega = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = omega[:]


class RK2StepRigidBodyRotationMatricesOptimized(RK2StepRigidBodyRotationMatrices):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                # save the current angular momentum
                # dst.ang_mom0[3*i+j] = dst.ang_mom[3*i+j]
                dst.omega0[3*i+j] = dst.omega[3*i+j]

            # save the current orientation
            for j in range(9):
                dst.R0[9*i+j] = dst.R[9*i+j]

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm[i3+j] + dtb2 * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # convert the angular velocity and torque to body frame
            ob = np.matmul(R.transpose(), dst.omega[i3:i3+3])
            tb = np.matmul(R.transpose(), dst.torque[i3:i3+3])
            mibp_i = dst.mibp[i3:i3+3]

            ob_dot = np.array([0., 0., 0.])
            ob_dot[0] = mibp_i[0] * (tb[0] - (mibp_i[2] - mibp_i[1]) *
                                     ob[2] * ob[1])
            ob_dot[1] = mibp_i[1] * (tb[1] - (mibp_i[0] - mibp_i[2]) *
                                     ob[0] * ob[2])
            ob_dot[2] = mibp_i[2] * (tb[2] - (mibp_i[1] - mibp_i[0]) *
                                     ob[1] * ob[0])

            # convert the rate of change of angular velocity from
            # body frame to global frame
            og_dot = np.matmul(R, ob_dot)

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dtb2
            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # increment the angular velocity to next time step
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + og_dot * dtb2

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm0[i3+j] + dt * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3+2], dst.omega[i3+1]],
                                  [dst.omega[i3+2], 0, -dst.omega[i3+0]],
                                  [-dst.omega[i3+1], dst.omega[i3+0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # convert the angular velocity and torque to body frame
            ob = np.matmul(R.transpose(), dst.omega[i3:i3+3])
            tb = np.matmul(R.transpose(), dst.torque[i3:i3+3])
            mibp_i = dst.mibp[i3:i3+3]

            ob_dot = np.array([0., 0., 0.])
            ob_dot[0] = mibp_i[0] * (tb[0] - (mibp_i[2] - mibp_i[1]) *
                                     ob[2] * ob[1])
            ob_dot[1] = mibp_i[1] * (tb[1] - (mibp_i[0] - mibp_i[2]) *
                                     ob[0] * ob[2])
            ob_dot[2] = mibp_i[2] * (tb[2] - (mibp_i[1] - mibp_i[0]) *
                                     ob[1] * ob[0])

            # convert the rate of change of angular velocity from
            # body frame to global frame
            og_dot = np.matmul(R, ob_dot)

            # update the orientation to next time step
            dst.R[i9:i9+9] = dst.R0[i9:i9+9] + r_dot * dt
            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9+9])

            # increment the angular velocity to next time step
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + og_dot * dt


class RK2StepRigidBodyQuaternionsOptimized(RK2StepRigidBodyQuaternions):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                # save the current angular momentum
                # dst.ang_mom0[3*i+j] = dst.ang_mom[3*i+j]
                dst.omega0[3*i+j] = dst.omega[3*i+j]

            # save the current orientation
            for j in range(4):
                dst.q0[4*i+j] = dst.q[4*i+j]

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            i4 = 4 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm[i3+j] + dtb2 * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc[i3+j] + dtb2 * dst.force[i3+j] / dst.total_mass[i]

            # change in quaternion
            delta_quat = np.array([0., 0., 0., 0.])
            # angular velocity magnitude
            omega_magn = sqrt(dst.omega[i3]**2 + dst.omega[i3+1]**2 +
                              dst.omega[i3+2]**2)
            axis_rot = np.array([0., 0., 0.])
            if omega_magn > 1e-12:
                axis_rot = dst.omega[i3:i3+3] / omega_magn
            delta_quat[0] = cos(omega_magn * dtb2 * 0.5)
            delta_quat[1] = axis_rot[0] * sin(omega_magn * dtb2 * 0.5)
            delta_quat[2] = axis_rot[1] * sin(omega_magn * dtb2 * 0.5)
            delta_quat[3] = axis_rot[2] * sin(omega_magn * dtb2 * 0.5)

            # convert the angular velocity and torque to body frame
            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)
            ob = np.matmul(R.transpose(), dst.omega[i3:i3+3])
            tb = np.matmul(R.transpose(), dst.torque[i3:i3+3])
            mibp_i = dst.mibp[i3:i3+3]

            ob_dot = np.array([0., 0., 0.])
            ob_dot[0] = mibp_i[0] * (tb[0] - (mibp_i[2] - mibp_i[1]) *
                                     ob[2] * ob[1])
            ob_dot[1] = mibp_i[1] * (tb[1] - (mibp_i[0] - mibp_i[2]) *
                                     ob[0] * ob[2])
            ob_dot[2] = mibp_i[2] * (tb[2] - (mibp_i[1] - mibp_i[0]) *
                                     ob[1] * ob[0])

            # convert the rate of change of angular velocity from
            # body frame to global frame
            og_dot = np.matmul(R, ob_dot)
            # increment the angular velocity to next time step
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + og_dot * dtb2

            # using the computed net change rotation of quaternion
            # update the orientation to next time step
            res = np.array([0., 0., 0., 0.])
            quaternion_multiplication(dst.q[i4:i4+4], delta_quat, res)
            dst.q[i4:i4+4] = res

            # normalize the orientation
            normalize_q_orientation(dst.q[i4:i4+4])

            # get the rotation matrix from quaternion
            quaternion_to_matrix(dst.q[i4:i4+4], dst.R[i9:i9+9])

    def py_stage2(self, dst, t, dt):
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            i4 = 4 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.cm[i3+j] = dst.cm0[i3+j] + dt * dst.vc[i3+j]
                dst.vc[i3+j] = dst.vc0[i3+j] + dt * dst.force[i3+j] / dst.total_mass[i]

            # change in quaternion
            delta_quat = np.array([0., 0., 0., 0.])
            # angular velocity magnitude
            omega_magn = sqrt(dst.omega[i3]**2 + dst.omega[i3+1]**2 +
                              dst.omega[i3+2]**2)
            axis_rot = np.array([0., 0., 0.])
            if omega_magn > 1e-12:
                axis_rot = dst.omega[i3:i3+3] / omega_magn
            delta_quat[0] = cos(omega_magn * dt * 0.5)
            delta_quat[1] = axis_rot[0] * sin(omega_magn * dt * 0.5)
            delta_quat[2] = axis_rot[1] * sin(omega_magn * dt * 0.5)
            delta_quat[3] = axis_rot[2] * sin(omega_magn * dt * 0.5)

            # convert the angular velocity and torque to body frame
            # Currently the orientation is at time t
            R = dst.R[i9:i9+9].reshape(3, 3)
            ob = np.matmul(R.transpose(), dst.omega[i3:i3+3])
            tb = np.matmul(R.transpose(), dst.torque[i3:i3+3])
            mibp_i = dst.mibp[i3:i3+3]

            ob_dot = np.array([0., 0., 0.])
            ob_dot[0] = mibp_i[0] * (tb[0] - (mibp_i[2] - mibp_i[1]) *
                                     ob[2] * ob[1])
            ob_dot[1] = mibp_i[1] * (tb[1] - (mibp_i[0] - mibp_i[2]) *
                                     ob[0] * ob[2])
            ob_dot[2] = mibp_i[2] * (tb[2] - (mibp_i[1] - mibp_i[0]) *
                                     ob[1] * ob[0])

            # convert the rate of change of angular velocity from
            # body frame to global frame
            og_dot = np.matmul(R, ob_dot)
            # increment the angular velocity to next time step
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + og_dot * dt

            # using the computed net change rotation of quaternion
            # update the orientation to next time step
            res = np.array([0., 0., 0., 0.])
            quaternion_multiplication(dst.q0[i4:i4+4], delta_quat, res)
            dst.q[i4:i4+4] = res

            # normalize the orientation
            normalize_q_orientation(dst.q[i4:i4+4])

            # get the rotation matrix from quaternion
            quaternion_to_matrix(dst.q[i4:i4+4], dst.R[i9:i9+9])


class ChildBodyStep(RK2StepRigidBodyRotationMatrices):
    def py_initialize(self, dst, t, dt):
        pass

    def py_stage1(self, dst, t, dt):
        pass

    def py_stage2(self, dst, t, dt):
        pass


class RigidBodyScheme(Scheme):
    def __init__(self, rigid_bodies, boundaries, dim, orientation, principal_moi, kn,
                 child_rigid_bodies=None, mu=0.5, en=1.0, gx=0.0, gy=0.0, gz=0.0,
                 debug=False):
        self.rigid_bodies = rigid_bodies
        self.boundaries = boundaries
        self.child_rigid_bodies = child_rigid_bodies
        self.dim = dim
        self.orientation = orientation
        self.principal_moi = principal_moi
        self.kn = kn
        self.mu = mu
        self.en = en
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.debug = debug

    def setup_properties(self, particles, clean=True):
        rigid_body_names = self.rigid_bodies
        rigid_body_arrays = []
        for pa_array in particles:
            if pa_array.name in rigid_body_names:
                setup_rigid_body_unconstrained_dynamics(pa_array,
                                                        principal_moi=self.principal_moi)

    def add_user_options(self, group):
        group.add_argument(
            "--orientation", action="store", dest="orientation",
            default="DCM",
            type=str,
            help="Orientation of rigid body"
        )

        add_bool_argument(
            group, 'principal_moi', dest='principal_moi', default=False,
            help='Use principal moment of inertia'
        )

    def consume_user_options(self, options):
        _vars = ['orientation', 'principal_moi']
        data = dict((var, self._smart_getattr(options, var))
                    for var in _vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EPECIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        rigid_bodies = self.rigid_bodies
        for body in rigid_bodies:
            if body not in steppers:
                if self.orientation == "quaternion" and self.principal_moi == True:
                    steppers[body] = RK2StepRigidBodyQuaternionsOptimized()

                if self.orientation == "quaternion" and self.principal_moi == False:
                    steppers[body] = RK2StepRigidBodyQuaternions()

                if self.orientation == "DCM" and self.principal_moi == True:
                    steppers[body] = RK2StepRigidBodyRotationMatricesOptimized()

                if self.orientation == "DCM" and self.principal_moi == False:
                    steppers[body] = RK2StepRigidBodyRotationMatrices()


        if self.child_rigid_bodies is not None:
            child_rigid_bodies = self.child_rigid_bodies
            for body in child_rigid_bodies:
                if body not in steppers:
                    steppers[body] = ChildBodyStep()


        cls = integrator_cls if integrator_cls is not None else EPECIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        equations = []
        g1 = []
        if self.boundaries is not None:
            all = self.rigid_bodies + self.boundaries
        else:
            all = self.rigid_bodies

        if self.child_rigid_bodies is not None:
            all = all + self.child_rigid_bodies

        for name in self.rigid_bodies:
            g1.append(
                BodyForce(dest=name, sources=None, gx=self.gx, gy=self.gy,
                          gz=self.gz))
        equations.append(Group(equations=g1, real=False))

        g2 = []
        if self.child_rigid_bodies is not None:
            rigid_bodies = self.rigid_bodies + self.child_rigid_bodies
        else:
            rigid_bodies = self.rigid_bodies

        for name in rigid_bodies:
            g2.append(
                RigidBodyCollision(dest=name, sources=all, kn=self.kn,
                                   mu=self.mu, en=self.en))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        for name in self.rigid_bodies:
            g3.append(SumUpExternalForces(dest=name, sources=None))
        equations.append(Group(equations=g3, real=False))

        g4 = []
        if self.child_rigid_bodies is not None:
            for name in self.child_rigid_bodies:
                g4.append(SumUpExternalForces(dest=name, sources=None))

            equations.append(Group(equations=g4, real=False))

        return equations
