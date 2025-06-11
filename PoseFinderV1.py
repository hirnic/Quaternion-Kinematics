# This program takes randomly generated leg lengths and tries to recover the pose of the table
# The idea is to use Newton-Raphson with Randomizer.Lengths() as our function to invert

import DQClass
import numpy as np


# This takes the initial guess, the lengths of the legs, the coordinates of the table in identity pose, and the base
# coordinates, and outputs the pose dual quaternion of the table.
def PoseFinder(init, lengths, TableID, Base):
    X, Y = init, DQClass.ZeroDQ()
    while (X - Y).size() > 1e-4:
        Y = X
        Q0 = X.A.w
        Q1 = X.A.x
        Q2 = X.A.y
        Q3 = X.A.z
        t1 = X.B.x
        t2 = X.B.y
        t3 = X.B.z

        # Here we extract the Jacobian Matrix
        Jacobian = []
        lossVec = []
        for n in range(6):
            r = TableID[n]
            A = 2 * (r.y * (Q1 * Q2 - Q0 * Q3) + r.z * (Q1 * Q3 + Q0 * Q2)) + r.x * (1 - 2 * Q2 ** 2 - 2 * Q3 ** 2) + t1
            B = 2 * (r.x * (Q1 * Q2 + Q0 * Q3) + r.z * (Q2 * Q3 - Q0 * Q1)) + r.y * (1 - 2 * Q1 ** 2 - 2 * Q3 ** 2) + t2
            C = 2 * (r.x * (Q1 * Q3 - Q0 * Q2) + r.y * (Q2 * Q3 + Q0 * Q1)) + r.z * (1 - 2 * Q1 ** 2 - 2 * Q2 ** 2) + t3
            s = DQClass.Quaternion(0, A, B, C)
            b = Base[n]

            # Here are partial derivatives involved in the length function derived from formulas
            DADQ1 = 2 * (r.y * (Q2 * Q0 + Q1 * Q3) + r.z * (Q3 * Q0 - Q1 * Q2))
            DADQ2 = 2 * (r.y * (Q0 * Q1 + Q2 * Q3) + r.z * (Q0 ** 2 - Q2 ** 2) - 2 * r.x * Q0 * Q2)
            DADQ3 = 2 * (r.y * (Q3 ** 2 - Q0 ** 2) + r.z * (Q1 * Q0 - Q2 * Q3) - 2 * r.x * Q0 * Q3)
            DBDQ1 = 2 * (r.x * (Q0 * Q2 - Q1 * Q3) + r.z * (Q1 ** 2 - Q0 ** 2) - 2 * r.y * Q0 * Q1)
            DBDQ2 = 2 * (r.x * (Q0 * Q1 - Q2 * Q3) + r.z * (Q0 * Q3 + Q1 * Q2))
            DBDQ3 = 2 * (r.x * (Q0 ** 2 - Q3 ** 2) + r.z * (Q0 * Q2 + Q1 * Q3) - 2 * r.y * Q0 * Q3)
            DCDQ1 = 2 * (r.x * (Q0 * Q3 + Q1 * Q2) + r.y * (Q0 ** 2 - Q1 ** 2) - 2 * r.z * Q0 * Q1)
            DCDQ2 = 2 * (r.x * (Q2 ** 2 - Q0 ** 2) + r.y * (Q0 * Q3 - Q1 * Q2) - 2 * r.z * Q0 * Q2)
            DCDQ3 = 2 * (r.x * (Q0 * Q1 + Q2 * Q3) + r.y * (Q0 * Q2 - Q1 * Q3))
            DSDQ1 = DQClass.Quaternion(0, DADQ1, DBDQ1, DCDQ1)
            DSDQ2 = DQClass.Quaternion(0, DADQ2, DBDQ2, DCDQ2)
            DSDQ3 = DQClass.Quaternion(0, DADQ3, DBDQ3, DCDQ3)

            u = s - b

            DFDQ1 = (2 * u.DotProduct(DSDQ1)) / Q0
            DFDQ2 = (2 * u.DotProduct(DSDQ2)) / Q0
            DFDQ3 = (2 * u.DotProduct(DSDQ3)) / Q0
            DFDT1 = 2 * u.x
            DFDT2 = 2 * u.y
            DFDT3 = 2 * u.z

            lossVec.append(u.norm() ** 2 - lengths[n] ** 2)
            Jacobian.append(np.array([DFDQ1, DFDQ2, DFDQ3, DFDT1, DFDT2, DFDT3]))

        matrix = np.array(Jacobian)
        addend = np.linalg.solve(matrix, lossVec)
        X = X.To6Vec() - addend
        rotation = DQClass.ToQuaternionRotation(X[:3])
        translation = DQClass.Quaternion(0, X[3], X[4], X[5])
        X = DQClass.DQuaternion(rotation, translation)
        # print(X)
    X.B = X.B * X.A * (0.5)
    return X


# import RandomizerRedacted as rr
# identity_pose = DQClass.IdentityDQ()
# Table = rr.TableID
# Base = rr.Base
# identity_lengths = rr.LegLengthsRedacted(identity_pose)
#
# import math
# delta = 100/math.sqrt(6)
# identity_lengths = [x for x in identity_lengths]
# print(identity_lengths)
# # id_lengths = 1184.1607030279968
#
# init = DQClass.DQuaternion(DQClass.Quaternion(1, 0, 0, 0), DQClass.Quaternion(0, 0, 0, 0))
# print("IKP: ", rr.LegLengthsRedacted(init))
#
# for i in range(64):
#     halves = np.array([.5,.5,.5,.5,.5,.5])
#     num = np.array(list(map(int, bin(i)[2:].zfill(6))))
#     wholes = (halves - num) * delta
#     lankth = identity_lengths + wholes
#     print("Length: ", lankth)
#     print("Pose: ", PoseFinder(init, identity_lengths + wholes, Table, Base))
