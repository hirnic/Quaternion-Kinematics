# This implements NR method for the FKP using engineer poses (Euler angles)

import numpy as np


def displacement_function(pose_params, point_to_displace):
    """

    :param pose_params: np array containing floats phi, theta, psi, x, y, z, where phi, theta, psi are the x, y, z
    rotation angles, respectively, and x, y, z are the translations, respectively.
    :param point_to_displace: np array with 3 floats for x, y, z spatial coordinates of the point to be displaced
    :return:  np array with 3 floats giving the newly rotated + translated point.
    """
    phi, theta, psi, x, y, z = pose_params
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R = np.array([
        [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
        [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
        [-s_theta, c_theta * s_phi, c_theta * c_phi]
    ])

    t = np.array([x, y, z])

    return R @ point_to_displace + t


def square_inverse_kinematics_function(pose_params, table_points, base_points):
    """
    :param pose_params: np array [phi, theta, psi, x, y, z]
    The Euler angles are phi, theta, psi, (radians) and translations are x, y, z (millimeters).
    :param table_points: The point on the table in the identity pose (np array)
    :param base_points: the point on the base in the identity pose (np array).
    :return: np array of floats (the six leg lengths in millimeters). Leg lengths.
    """
    # Create rotation using ZYX (yaw-pitch-roll) order
    length_squares = []
    for i in range(6):
        new_point = displacement_function(pose_params, table_points[i])
        length_squares.append((base_points[i][0] - new_point[0])**2 + (base_points[i][1] - new_point[1])**2
                       + (base_points[i][2] - new_point[2])**2)

    return np.array(length_squares)


def jacobian(pose_params, table_points, base_points):
    phi, theta, psi, x, y, z = pose_params

    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Jacobian = []
    for i in range(6):
        new_point = displacement_function(pose_params, table_points[i])

        dR_dphi = np.array([
            [0, cpsi * stheta * cphi + spsi * sphi, spsi * cphi - cpsi * stheta * sphi],
            [0, spsi * stheta * cphi - cpsi * sphi, -spsi * stheta * sphi - cpsi * cphi],
            [0, ctheta * cphi, -ctheta * sphi]
        ])

        dR_dtheta = np.array([
            [-cpsi * stheta, cpsi * ctheta * sphi, cpsi * ctheta * cphi],
            [-spsi * stheta, spsi * ctheta * sphi, spsi * ctheta * cphi],
            [-ctheta, -stheta * sphi, -stheta * cphi]
        ])

        dR_dpsi = np.array([
            [-spsi * ctheta, -spsi * stheta * sphi - cpsi * cphi, cpsi * sphi - spsi * stheta * cphi],
            [cpsi * ctheta, cpsi * stheta * sphi - spsi * ctheta, cpsi * stheta * cphi + spsi * sphi],
            [0, 0, 0]
        ])

        ds_sqr_dphi = 2 * (new_point - base_points[i]) @ (dR_dphi @ table_points[i])
        ds_sqr_dtheta = 2 * (new_point - base_points[i]) @ (dR_dtheta @ table_points[i])
        ds_sqr_dpsi = 2 * (new_point - base_points[i]) @ (dR_dpsi @ table_points[i])
        ds_sqr_dt1 = 2 * (new_point - base_points[i])[0]
        ds_sqr_dt2 = 2 * (new_point - base_points[i])[1]
        ds_sqr_dt3 = 2 * (new_point - base_points[i])[2]

        Jacobian.append([ds_sqr_dphi, ds_sqr_dtheta, ds_sqr_dpsi, ds_sqr_dt1, ds_sqr_dt2, ds_sqr_dt3])

    return np.array(Jacobian)


def inverse_jacobian(pose_params, table_points, base_points):
    J = jacobian(pose_params, table_points, base_points)
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
    length_squares = [a**2 for a in lengths]
    DL = length_squares
    while np.linalg.norm(DL) > 1e-4:
        # print("Step 2:")
        L0 = square_inverse_kinematics_function(U, TableID, Base)
        DL = L0 - length_squares
        J = jacobian(U, TableID, Base)
        DU = np.linalg.solve(J, DL)

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
