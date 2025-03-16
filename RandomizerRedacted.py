# This file generates a random pose and converts it to a set of leg lengths.
import random
import math
import DQClass

d = 200  # This is the gap between two adjacent table spherical joints in millimeters. I guessed to protect info.
D = 1759  # This is the diameter of the table in millimeters
b = 150  # This is the gap between two adjacent base spherical joints in millimeters. I guessed to protect info.
B = 2011.4  # This is meant to be the diameter of the base in millimeters, but I did not know how to interpret it.
h = 1171.7  # This is meant to be the height of the table top in millimeters, but I am sure I misinterpreted this.

R1 = DQClass.Quaternion(math.cos(math.pi / 6), 0, 0, math.sin(math.pi / 6))
R2 = DQClass.Quaternion(math.cos(math.pi / 3), 0, 0, math.sin(math.pi / 3))

T1 = DQClass.ToVectorQuaternion([d/2, D/2 - math.sqrt(3) * d / 2, h])
T2 = DQClass.ToVectorQuaternion([-d/2, D/2 - math.sqrt(3) * d / 2, h])
T3 = R2 * T1 * R2.conjugate()
T4 = R2 * T2 * R2.conjugate()
T5 = R2 * T3 * R2.conjugate()
T6 = R2 * T4 * R2.conjugate()

B1 = DQClass.ToVectorQuaternion([b/2, B/2 - math.sqrt(3) * B / 2, 0])
B2 = DQClass.ToVectorQuaternion([-b/2, B/2 - math.sqrt(3) * B / 2, 0])
B3 = R2 * B1 * R2.conjugate()
B4 = R2 * B2 * R2.conjugate()
B5 = R2 * B3 * R2.conjugate()
B6 = R2 * B4 * R2.conjugate()

# Here is the list of table coordinates as quaternions
TableID = [T1, T2, T3, T4, T5, T6]
# Here is the list of base coordinates as quaternions
Base = [B1, B2, B3, B4, B5, B6]
Base = [R1 * point * R1.conjugate() for point in Base]

# Here are the limitations on the P3350
# MaxTranslation = 635 mm (X,Y)
# MaxHeave = 694 mm (Z)
# Pitch = 30 degrees
# Yaw = 45 degrees


def MakeRandomPose():
    # Here we generate a random rotation
    theta = random.uniform(-0.5, 0.5)  # Rotation Angle between -30 and 30 degrees
    i = random.uniform(-1, 1)
    j = random.uniform(-1, 1)
    k = random.uniform(-1, 1)
    Q = (DQClass.Quaternion(0, i, j, k).normalization() * math.sin(1 / 2 * theta)
         + DQClass.Quaternion(math.cos(1 / 2 * theta), 0, 0, 0))
    #
    # Here we generate a random translation
    t1 = random.randrange(1, 500)
    t2 = random.randrange(1, 500)
    t3 = random.randrange(1, 500)
    t = DQClass.Quaternion(0, t1, t2, t3)
    #
    RandomPose = DQClass.DQuaternion(Q, t * Q * (1/2))
    return RandomPose


# Here we extract the coordinates under the pose (Q, t)
def TableCoords(pose):
    TablePosition = []
    Q = pose.A
    t = pose.B * pose.A.conjugate() * 2
    for n in range(6):
        r = DQClass.ToVectorQuaternion(TableID[n])
        s = Q * r * Q.conjugate() + t
        TablePosition.append([s.x, s.y, s.z])


def LegLengthsRedacted(pose):
    Q = pose.A
    t = pose.B * pose.A.conjugate() * 2
    Lengths = []
    for n in range(6):
        r = TableID[n]
        s = Q * r * Q.conjugate() + t
        basePoint = Base[n]
        L = (s-basePoint).norm()
        Lengths.append(L)
    return Lengths
