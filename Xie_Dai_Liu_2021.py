# "A Lie Group-Based Iterative Algorithm Framework for Numerically Solving Forward Kinematics of Gough–Stewart Platform"
# by Xie, Dai, Liu. Written in 2021. This paper includes a Gauss-Newton iteration and a Levenberg-Marquardt iteration.

import numpy as np
from scipy.linalg import expm


def rigid_body_matrix(pose_params):
    phi, theta, psi, x, y, z = pose_params

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    # Rotation matrix R
    R = np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta,       ctheta * sphi,                     ctheta * cphi]
    ])

    R_T = R.T               # Transpose of R
    v = np.array([x, y, z]) # Translation vector
    t_inv = -R_T @ v        # New translation

    # Construct the 4x4 inverse transformation matrix
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_T
    M_inv[:3, 3] = t_inv

    return M_inv


def square_inverse_kinematics_function(pose_matrix, table_points, base_points):
    """
    :param pose_matrix: np array (inverse pose matrix)
    The Euler angles are phi, theta, psi, (radians) and translations are x, y, z (millimeters).
    :param table_points: The point on the table in the identity pose (np array)
    :param base_points: the point on the base in the identity pose (np array).
    :return: np array of floats (the six leg lengths in millimeters). Leg lengths.
    """
    # Create rotation using ZYX (yaw-pitch-roll) order
    length_squares = []
    for i in range(6):
        base_point = np.concatenate([base_points[i], [1]])
        new_point = (pose_matrix @ base_point)[:3]
        length_squares.append((table_points[i][0] - new_point[0])**2 + (table_points[i][1] - new_point[1])**2
                       + (table_points[i][2] - new_point[2])**2)

    return np.array(length_squares)


def residual_vector(pose_matrix, lengths, TableID, Base):
    length_squares = np.array([a ** 2 for a in lengths])
    F = square_inverse_kinematics_function(pose_matrix, TableID, Base)
    return F - length_squares


def screw_matrix(screw_vec):
    """
    :param screw_vec: an np.array with 6 elements
    :return: 4x4 np.array
    """
    w = np.array(screw_vec[:3])
    v = np.array(screw_vec[3:])
    return np.array([
        [0, -w[2], w[1], v[0]],
        [w[2], 0, -w[0], v[1]],
        [-w[1], w[0], 0, v[2]],
        [0, 0, 0, 0]
    ])


def Jacobian(pose_matrix, TableID, Base):
    """
    :param pose_matrix: np array (inverse pose matrix)
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return: Matrix (np.array)
    """

    Jac = []
    for i in range(6):
        base_point = np.concatenate([Base[i], [1]])
        first = np.cross((pose_matrix @ base_point)[:3], TableID[i])
        second = (TableID[i] - (pose_matrix @ base_point)[:3])
        row = np.concatenate([first, second]) * 2
        Jac.append(row)
    return np.array(Jac)


def PoseFinderGN(init, lengths, TableID, Base):
    """
    :param init: Engineer's pose (array of 6 floats: phi,theta,psi, rotation; x,y,z, translation). Initial guess.
    :param lengths: array of 6 floats. Leg lengths.
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return np array (4x4 pose matrix)
    """
    eps1 = 1e-12
    eps2 = 1e-12
    eps3 = 1e-12
    kmax = 200
    k, T = 0, rigid_body_matrix(init)
    J = Jacobian(T, TableID, Base)
    A = J.T @ J
    r = residual_vector(T, lengths, TableID, Base)
    g = J.T @ r
    # print("g = ", g)
    # print("np.max(np.abs(g)) = ", np.max(np.abs(g)))
    found = (np.max(np.abs(g)) <= eps1)
    while not found and k < kmax:
        k += 1
        print("T = ", T)
        sgn = np.linalg.solve(A, -g)
        alpha = 1
        S_gn = screw_matrix(sgn)
        r1 = np.linalg.norm(residual_vector(expm(-0.5 * alpha * S_gn) @ T, lengths, TableID, Base))
        r2 = np.linalg.norm(residual_vector(expm(-alpha * S_gn) @ T, lengths, TableID, Base))
        accept = (r1 <= np.linalg.norm(r)) and (r2 <= r1)
        while not accept and alpha > eps3:
            alpha /= 2
            r2 = r1
            r1 = np.linalg.norm(residual_vector(expm(-0.5*alpha*S_gn) @ T, lengths, TableID, Base))
            accept = (r1 <= np.linalg.norm(r)) and (r2 <= r1)
        print("alpha: ", alpha)
        T = expm(-alpha * S_gn) @ T
        J = Jacobian(T, TableID, Base)
        A = J.T @ J
        r = residual_vector(T, lengths, TableID, Base)
        g = J.T @ r
        found = (np.max(np.abs(g)) <= eps1)
    return T


import RandomizerRedacted as rr
pose = np.array(rr.MakeEulerPose())
Table = np.array([rr.TableID[i].ToPureVec() for i in range(6)])
Base = np.array([rr.Base[i].ToPureVec() for i in range(6)])
input_lengths = rr.euler_lengths(pose, Table, Base)
identity_lengths = rr.euler_lengths(np.array([0,0,0,0,0,0]), Table, Base)
print("Input Lengths: ", input_lengths)
print("ID Lengths: ", identity_lengths)
computed_pose = PoseFinderGN(np.array([0,0,0,0,0,0]), input_lengths, Table, Base)
print("Computed Pose: ", computed_pose)
print("Pose: ", rigid_body_matrix(pose))

# Not impressed with this work. Very poor proofreading. This paper takes the work of Selig_Li and makes it slower.