import numpy as np
from scipy.spatial.transform import Rotation as Rot
from math import sqrt


def set_angular_momentum(pa_arrays):
    """
    pa is list of particle arrays
    """
    # loop over all the bodies
    for pa in pa_arrays:
        nb = pa.nb[0]
        for i in range(nb):
            pa.ang_mom[3*i:3*i+3] = np.matmul(pa.moig[9*i:9*i+9].reshape(3, 3),
                                        pa.omega[3*i:3*i+3])


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

        pa.moig[9 * i:9 * i + 9] = I[:]

        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.mib[9 * i:9 * i + 9] = I_inv[:]


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
        # this value will be used only at the beginning of the simulation


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
