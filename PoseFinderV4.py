# This file takes a random function f:\R^n \to \R^n and inverts it if possible.
import numpy as np
import math
import DQClass
import Randomizer

Base = Randomizer.Base
TablePositions = Randomizer.TableID
LegLengths = Randomizer.LegLengths()
RandomPose = Randomizer.RandomPose
TLS = [LegLengths[n] ** 2 for n in range(6)]  # True length squares


# function is callable and must be 1D array, init must be a 1D array of the same length, and maxIter is an integer.
def Newton(function, init):
    n = len(init)
    X = init
    Norm = np.linalg.norm(function(X))
    while Norm > 10**(-8):
        JacobianTranspose = []
        Delta = 10**(-10)
        for i in range(n):  # This loop makes the Jacobian using a 4-point finite difference formula
            Y = np.copy(X)
            Y[i] = Y[i] - 2.0 * Delta
            fm2 = function(Y)
            #
            Y[i] = Y[i] + Delta
            fm1 = np.multiply(function(Y), -8.0)
            #
            Y[i] = Y[i] + 2.0 * Delta
            fp1 = np.multiply(function(Y), 8.0)
            #
            Y[i] = Y[i] + Delta
            fp2 = np.multiply(function(Y), -1.0)
            #
            column = (fp2 + fp1 + fm1 + fm2)/(12.0 * Delta)  # Found this at https://web.media.mit.edu/~crtaylor/calculator.html
            JacobianTranspose.append(column)
        #
        try:
            InverseJacobian = np.linalg.inv(np.transpose(JacobianTranspose))
            X = X - InverseJacobian.dot(function(X))
            Norm = np.linalg.norm(function(X))
        except:
            print("The Jacobian is probably singular, sorry.")
            return
    return X


# This is the function for optimization and root finding.
def kinematicFunction(x):
    Pose = DQClass.ToPose(x)
    f = []
    for n in range(6):
        r = [TablePositions[n][i] for i in range(3)]
        point = DQClass.ToQuaternionVector(r)
        s = point.PoseIt(Pose)
        a = (s.x - Base[n][0]) ** 2 + (s.y - Base[n][1]) ** 2 + (s.z - Base[n][2]) ** 2 - TLS[n]
        f.append(a)
    return np.array(f).astype(float)


# Error Check
Init = DQClass.IdentityDQ()
Init = np.array(Init.To6Vec()).astype(float)
T = RandomPose
F = Newton(kinematicFunction, Init)
F1 = DQClass.Quaternion(math.sqrt(1 - F[0]**2 - F[1]**2 - F[2]**2), F[0], F[1], F[2])
F2 = DQClass.Quaternion(0, F[3], F[4], F[5])
difference1 = T.A - F1
difference2 = T.B - F2
absDist1 = math.sqrt(difference1.norm() ** 2 + difference2.norm() ** 2)
print("Absolute Error: ", absDist1)
print("Approximate Pose: ", DQClass.DQuaternion(F1, F2))
