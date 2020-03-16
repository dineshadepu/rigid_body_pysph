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
from pysph.sph.scheme import Scheme
# from pysph.sph.rigid_body_setup import (
#     setup_rotation_matrix_rigid_body, setup_quaternion_rigid_body,
#     setup_rotation_matrix_rigid_body_optimized,
#     setup_quaternion_rigid_body_optimized)
from compyle.api import (elementwise, annotate, wrap, declare)
from compyle.low_level import (address)
from pysph.sph.wc.linalg import (mat_mult, mat_vec_mult, dot)
from numpy import sin, cos

import numpy as np
from scipy.spatial.transform import Rotation as Rot


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


def setup_rigid_body(pa, body_id, principal_moi):
    """
    This function will add all the properties regarding unconstrained
    rigid body dynamics.

    The following schemes will be implemented

    1. Motion with Direction cosine matrices
    2. using quaternions
    3. Using principal moment of inertia for both dcm and quaternion
    """
    pa.add_property('body_id', body_id, dtype=int)

    nb = np.max(body_id) + 1

    # first add all the rigid body properties
    # Every other rigid body scheme or implementation must
    # have to be based on these following properties
    pa.add_constant("total_mass", np.zeros(nb))
    pa.add_constant("moib_inv", np.zeros(9*nb))
    pa.add_constant("moig_inv", np.zeros(9*nb))
    pa.add_constant("principal_moib_inv", np.zeros(3*nb))

    pa.add_constant("cm", np.zeros(3*nb))
    pa.add_constant("vc", np.zeros(3*nb))
    pa.add_constant("omega", np.zeros(3*nb))
    pa.add_constant("force", np.zeros(3*nb))
    pa.add_constant("torque", np.zeros(3*nb))

    # total no of rigid bodies
    pa.add_constant("nb", nb)

    # if the rigid body uses DCM then we need following property
    pa.add_constant("R", np.zeros(9*nb))

    # if the rigid body uses quaternion
    pa.add_constant("q", np.zeros(4*nb))

    # Find total mass
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_moment_of_inertia(pa)

    # set_basic_rigid_body_properties(pa)


def normalize_q_orientation(q):
    norm_q = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q[:] = q[:] / norm_q


def set_total_mass(pa):
    # left limit of body i
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.total_mass[i] = np.sum(pa.m[fltr])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    # loop over all the bodies
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        pa.cm[3 * i] = np.sum(pa.m[fltr] * pa.x[fltr]) / pa.total_mass[i]
        pa.cm[3 * i + 1] = np.sum(pa.m[fltr] * pa.y[fltr]) / pa.total_mass[i]
        pa.cm[3 * i + 2] = np.sum(pa.m[fltr] * pa.z[fltr]) / pa.total_mass[i]


def set_moment_of_inertia(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.moib_inv[9 * i:9 * i + 9] = I_inv[:]


def set_mi_in_body_frame(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.mib[9 * i:9 * i + 9] = I_inv[:]


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]
        for j in fltr:
            pa.dx0[j] = pa.x[j] - cm_i[0]
            pa.dy0[j] = pa.y[j] - cm_i[1]
            pa.dz0[j] = pa.z[j] - cm_i[2]


def set_mi_in_body_frame_rot_mat_optimized(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        # find the eigen vectors and eigen values of the moi
        vals, R = np.linalg.eigh(I.reshape(3, 3))
        # find the determinant of R
        determinant = np.linalg.det(R)
        if determinant == -1.:
            R[:, 0] = -R[:, 0]

        # recompute the moment of inertia about the new coordinate frame
        # if flipping of one of the axis due the determinant value
        R = R.ravel()

        if determinant == -1.:
            I = np.zeros(9)
            for j in fltr:
                dx = pa.x[j] - cm_i[0]
                dy = pa.y[j] - cm_i[1]
                dz = pa.z[j] - cm_i[2]

                dx0 = (R[0] * dx + R[3] * dy + R[6] * dz)
                dy0 = (R[1] * dx + R[4] * dy + R[7] * dz)
                dz0 = (R[2] * dx + R[5] * dy + R[8] * dz)

                # Ixx
                I[0] += pa.m[j] * (
                    (dy0)**2. + (dz0)**2.)

                # Iyy
                I[4] += pa.m[j] * (
                    (dx0)**2. + (dz0)**2.)

                # Izz
                I[8] += pa.m[j] * (
                    (dx0)**2. + (dy0)**2.)

                # Ixy
                I[1] -= pa.m[j] * (dx0) * (dy0)

                # Ixz
                I[2] -= pa.m[j] * (dx0) * (dz0)

                # Iyz
                I[5] -= pa.m[j] * (dy0) * (dz0)

            I[3] = I[1]
            I[6] = I[2]
            I[7] = I[5]

            # set the inverse inertia values
            vals = np.array([I[0], I[4], I[8]])

        pa.mibp[3 * i:3 * i + 3] = 1. / vals
        pa.R[9 * i:9 * i + 9] = R


def rotation_mat_to_quat(R, q):
    """This code is taken from
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    """
    q[0] = np.sqrt(R[0] + R[4] + R[8]) / 2
    q[1] = (R[7] - R[5]) / (4. * q[0])
    q[2] = (R[2] - R[6]) / (4. * q[0])
    q[3] = (R[3] - R[1]) / (4. * q[0])


def set_mi_in_body_frame_quaternion_optimized(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    # no of bodies
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]

        I = np.zeros(9)
        for j in fltr:
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

            # Izz
            I[8] += pa.m[j] * (
                (pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        # find the eigen vectors and eigen values of the moi
        vals, R = np.linalg.eigh(I.reshape(3, 3))
        # find the determinant of R
        determinant = np.linalg.det(R)
        if determinant == -1.:
            R[:, 0] = -R[:, 0]

        # recompute the moment of inertia about the new coordinate frame
        # if flipping of one of the axis due the determinant value
        R = R.ravel()

        if determinant == -1.:
            I = np.zeros(9)
            for j in fltr:
                dx = pa.x[j] - cm_i[0]
                dy = pa.y[j] - cm_i[1]
                dz = pa.z[j] - cm_i[2]

                dx0 = (R[0] * dx + R[3] * dy + R[6] * dz)
                dy0 = (R[1] * dx + R[4] * dy + R[7] * dz)
                dz0 = (R[2] * dx + R[5] * dy + R[8] * dz)

                # Ixx
                I[0] += pa.m[j] * (
                    (dy0)**2. + (dz0)**2.)

                # Iyy
                I[4] += pa.m[j] * (
                    (dx0)**2. + (dz0)**2.)

                # Izz
                I[8] += pa.m[j] * (
                    (dx0)**2. + (dy0)**2.)

                # Ixy
                I[1] -= pa.m[j] * (dx0) * (dy0)

                # Ixz
                I[2] -= pa.m[j] * (dx0) * (dz0)

                # Iyz
                I[5] -= pa.m[j] * (dy0) * (dz0)

            I[3] = I[1]
            I[6] = I[2]
            I[7] = I[5]

            # set the inverse inertia values
            vals = np.array([I[0], I[4], I[8]])

        pa.mibp[3 * i:3 * i + 3] = 1. / vals

        # get the quaternion from the rotation matrix
        r = Rot.from_dcm(R.reshape(3, 3))
        q_tmp = r.as_quat()
        q = np.zeros(4)
        q[0] = q_tmp[3]
        q[1] = q_tmp[0]
        q[2] = q_tmp[1]
        q[3] = q_tmp[2]

        normalize_q_orientation(q)
        pa.q[4 * i:4 * i + 4] = q

        # also set the rotation matrix
        pa.R[9 * i:9 * i + 9] = R


def set_body_frame_position_vectors_optimized(pa):
    """Save the position vectors w.r.t body frame"""
    nb = pa.nb[0]
    # loop over all the bodies
    for i in range(nb):
        fltr = np.where(pa.body_id == i)[0]
        cm_i = pa.cm[3 * i:3 * i + 3]
        R_i = pa.R[9 * i:9 * i + 9]
        for j in fltr:
            dx = pa.x[j] - cm_i[0]
            dy = pa.y[j] - cm_i[1]
            dz = pa.z[j] - cm_i[2]

            pa.dx0[j] = (R_i[0] * dx + R_i[3] * dy + R_i[6] * dz)
            pa.dy0[j] = (R_i[1] * dx + R_i[4] * dy + R_i[7] * dz)
            pa.dz0[j] = (R_i[2] * dx + R_i[5] * dy + R_i[8] * dz)


def setup_rotation_matrix_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)


def setup_quaternion_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using quaternion."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)


def setup_rotation_matrix_rigid_body_optimized(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame_rot_mat_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)


def setup_quaternion_rigid_body_optimized(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame_quaternion_optimized(pa)
    set_body_frame_position_vectors_optimized(pa)


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class RK2StepRigidBodyDCM(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                # save the current angular momentum
                # dst.L0[j] = dst.L[j]
                dst.omega0[3*i+j] = dst.omega[3*i+j]

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
            # move angular velocity to t + dt/2.
            # omega_dot is
            tmp = dst.torque[i3:i3+3] - np.cross(
                dst.omega[i3:i3+3], np.matmul(dst.mig[i9:i9+9].reshape(3, 3),
                                              dst.omega[i3:i3+3]))
            omega_dot = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + omega_dot * dtb2

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
            # move angular velocity to t + dt
            # omega_dot is
            tmp = dst.torque[i3:i3+3] - np.cross(
                dst.omega[i3:i3+3], np.matmul(dst.mig[i9:i9+9].reshape(3, 3),
                                              dst.omega[i3:i3+3]))
            omega_dot = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + omega_dot * dt

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


def normalize_q_orientation(q):
    norm_q = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q[:] = q[:] / norm_q


def quaternion_multiplication(p, q, res):
    """Parameters
    ----------
    p   : [float]
          An array of length four
    q   : [float]
          An array of length four
    res : [float]
          An array of length four
    Here `p` is a quaternion. i.e., p = [p.w, p.x, p.y, p.z]. And q is an
    another quaternion.
    This function is used to compute the rate of change of orientation
    when orientation is represented in terms of a quaternion. When the
    angular velocity is represented in terms of global frame
    \frac{dq}{dt} = \frac{1}{2} omega q
    http://www.ams.stonybrook.edu/~coutsias/papers/rrr.pdf
    see equation 8
    """
    res[0] = (p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3])
    res[1] = (p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2])
    res[2] = (p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3])
    res[3] = (p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1])


def scale_quaternion(q, scale):
    q[0] = q[0] * scale
    q[1] = q[1] * scale
    q[2] = q[2] * scale
    q[3] = q[3] * scale


def quaternion_to_matrix(q, matrix):
    matrix[0] = 1. - 2. * (q[2]**2. + q[3]**2.)
    matrix[1] = 2. * (q[1] * q[2] - q[0] * q[3])
    matrix[2] = 2. * (q[1] * q[3] + q[0] * q[2])

    matrix[3] = 2. * (q[1] * q[2] + q[0] * q[3])
    matrix[4] = 1. - 2. * (q[1]**2. + q[3]**2.)
    matrix[5] = 2. * (q[2] * q[3] - q[0] * q[1])

    matrix[6] = 2. * (q[1] * q[3] - q[0] * q[2])
    matrix[7] = 2. * (q[2] * q[3] + q[0] * q[1])
    matrix[8] = 1. - 2. * (q[1]**2. + q[2]**2.)


class RK2StepRigidBodyQuaternions(RK2StepRigidBodyDCM):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.nb[0]):
            for j in range(3):
                # save the center of mass and center of mass velocity
                dst.cm0[3*i+j] = dst.cm[3*i+j]
                dst.vc0[3*i+j] = dst.vc[3*i+j]

                dst.omega0[3*i+j] = dst.omega[3*i+j]

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
            # move angular velocity to t + dt/2.
            # omega_dot is
            tmp = dst.torque[i3:i3+3] - np.cross(
                dst.omega[i3:i3+3], np.matmul(dst.mig[i9:i9+9].reshape(3, 3),
                                              dst.omega[i3:i3+3]))
            omega_dot = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + omega_dot * dtb2

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
            # move angular velocity to t + dt
            # omega_dot is
            tmp = dst.torque[i3:i3+3] - np.cross(
                dst.omega[i3:i3+3], np.matmul(dst.mig[i9:i9+9].reshape(3, 3),
                                              dst.omega[i3:i3+3]))
            omega_dot = np.matmul(dst.mig[i9:i9+9].reshape(3, 3), tmp)
            dst.omega[i3:i3+3] = dst.omega0[i3:i3+3] + omega_dot * dt


class RK2StepRigidBodyRotationMatricesOptimized(RK2StepRigidBodyDCM):
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
