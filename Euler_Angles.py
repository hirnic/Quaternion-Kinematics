# This implements NR method for the FKP using engineer poses (Euler angles)

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import approx_fprime


def inverse_kinematics_function(pose_params, table_points, base_points):
    """
    :param pose_params: np array [phi, theta, psi, x, y, z]
    The Euler angles are phi, theta, psi, (radians) and translations are x, y, z (millimeters).
    :param table: The point on the table in the identity pose (np array)
    :param base: the point on the base in the identity pose (np array).
    :return: np array of floats (the six leg lengths in millimeters). Leg lengths.
    """
    # Create rotation using ZYX (yaw-pitch-roll) order
    phi, theta, psi, x, y, z = pose_params
    rot = Rotation.from_euler('zyx', [psi, theta, phi])
    rotation_matrix = rot.as_matrix()
    trans = np.array([x, y, z])
    lengths = []
    for i in range(6):
        new_point = rotation_matrix.T @ (base_points[i] - trans)
        lengths.append(np.linalg.norm(table_points[i] - new_point))
    return np.array(lengths)


def jacobian_approximator(pose_params, table_points, base_points):
    epsilon = 1e-6
    rows = []
    for i in range(6):
        def func(p):
            return inverse_kinematics_function(p, table_points, base_points)[i]

        row = approx_fprime(pose_params, func, epsilon)
        rows.append(row)
    return np.vstack(rows)


def inverse_jacobian(pose_params, table_points, base_points):
    J = jacobian_approximator(pose_params, table_points, base_points)
    return np.linalg.inv(J)


def PoseFinder(init, lengths, TableID, Base):
    """
    :param init: an engineer's pose (array of 6 floats: phi,theta,psi, rotation; x,y,z, translation). Initial guess.
    :param lengths: array of 6 floats. Leg lengths.
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return: Engineer's pose.
    """

    #Step 1:
    U = init
    n = 1
    DL = lengths
    while np.linalg.norm(DL) > 1e-4:
        # print("Step 2:")
        L0 = inverse_kinematics_function(U, TableID, Base)
        DL = L0 - lengths
        Jp = inverse_jacobian(U, TableID, Base)
        DU = Jp @ DL

        U = U - DU
        print("Iteration " + str(n) + ": " + str(U))
        n += 1
    return U

import RandomizerRedacted as rr
pose = np.array(rr.MakeEulerPose())
Table = np.array([rr.TableID[i].ToPureVec() for i in range(6)])
Base = np.array([rr.Base[i].ToPureVec() for i in range(6)])
input_lengths = rr.euler_lengths(pose, Table, Base)
identity_lengths = rr.euler_lengths(np.array([0,0,0,0,0,0]), Table, Base)
print("Pose: ", pose)
print("Input Lengths: ", input_lengths)
print("ID Lengths: ", identity_lengths)
PoseFinder(np.array([0,0,0,0,0,0]), input_lengths, Table, Base)
