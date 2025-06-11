# This is script implements the algorithm from A Geometric Newton-Raphson Method for
# Gough-Stewart Platforms by Selig and Li . We will use Euler angles, because I am not convinced they thought of DQs.

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import expm


def rigid_body_matrix(pose_params):
    phi, theta, psi, x, y, z = pose_params

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    M = np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi, x],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi, y],
        [-stheta, ctheta * sphi, ctheta * cphi, z],
        [0, 0, 0, 1]
    ])

    return M


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


def inverse_kinematics_function(pose_matrix, table_points, base_points):
    """
    :param pose_params: np array [phi, theta, psi, x, y, z]
    The Euler angles are phi, theta, psi, (radians) and translations are x, y, z (millimeters).
    :param table: The point on the table in the identity pose (np array)
    :param base: the point on the base in the identity pose (np array).
    :return: np array of floats (the six squared leg lengths in millimeters). Leg lengths.
    """
    # Create rotation using ZYX (yaw-pitch-roll) order
    lengths = []
    for i in range(6):
        point = np.append(table_points[i], 1)
        new_point = pose_matrix @ point
        point = np.append(base_points[i], 1)
        lengths.append(np.linalg.norm(point - new_point))
    return np.array(lengths)


def inverse_jacobian(pose_matrix, table_points, base_points):
    J = []
    for i in range(6):
        b_tilde = np.append(table_points[i], 1)
        b_prime = np.array(pose_matrix @ b_tilde)[:3]
        J.append(np.concatenate([np.cross(base_points[i], b_prime), (b_prime - base_points[i])]) * 2)
    J = np.array(J)
    return np.linalg.inv(J)


def PoseFinder(init, lengths, TableID, Base):
    """
    :param init: an engineer's pose (array of 6 floats: phi,theta,psi, rotation; x,y,z, translation). Initial guess.
    :param lengths: array of 6 floats. Leg lengths.
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return: An element of the Lie Group SE(3) (4 x 4 np.array of the form ((R, v), (0, 1)))
    """
    M = rigid_body_matrix(init)
    square_lengths = np.array([a**2 for a in lengths])
    DL = square_lengths
    while np.linalg.norm(DL) > 1e-4:
        L0 = inverse_kinematics_function(M, TableID, Base)
        L0 = np.array([a**2 for a in L0])
        DL = L0 - square_lengths
        screw_vec = inverse_jacobian(M, TableID, Base) @ DL * (-1)
        S = screw_matrix(screw_vec)
        M = expm(S) @ M
    return M


import RandomizerRedacted as rr
pose = np.array(rr.MakeEulerPose())
Table = np.array([rr.TableID[i].ToPureVec() for i in range(6)])
Base = np.array([rr.Base[i].ToPureVec() for i in range(6)])
input_lengths = rr.euler_lengths(pose, Table, Base)
identity_lengths = rr.euler_lengths(np.array([0,0,0,0,0,0]), Table, Base)
print("Pose as Matrix: ", rigid_body_matrix(pose))
print("Input Lengths: ", input_lengths)
print("ID Lengths: ", identity_lengths)
print("Computed transformation matrix: ", PoseFinder(np.array([0,0,0,0,0,0]), input_lengths, Table, Base))
