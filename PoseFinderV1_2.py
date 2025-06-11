# This script is meant to generate a list of data points for the initial guesses. The idea is that the max and min leg
# lengths must be no farther apart than the platform geometry allows. Thus, we expect the allowable set of leg lengths
# to be clustered around the diagonal of $\mathbb{R}^6$. We will put a bubble around the diagonal and arrange congruent
# hypercubes vertex to vertex in this bubble going along the main diagonal. This will give us a hyperefficient way to
# look at the input leg lengths, find out which vertex is closest, and determine from a precompiled list of data points
# a reasonable initial guess for the pose. The cool part is that determining the initial point at runtime involves only
# bit shifts, adding, and looking up a pose in a table.

import DQClass
import numpy as np
import math
import PoseFinderV1 as PF

id_lengths = 1184.1607030279968  # Determined by table geometry
diagonal_length = 100
edge_length = diagonal_length/np.sqrt(6)
num_cubes = 11  # Should be odd for the next line to work properly
min_length = id_lengths - num_cubes / 2 * diagonal_length
max_length = id_lengths + num_cubes / 2 * diagonal_length

def sign(x):
    return 1 if x >= 0 else -1


def offline_phase(TableID, Base):

    def index_to_point(index):
        """Convert integer index [0, 63] to 6D point with values in {-1, 1}"""
        if not (0 <= index < 64):
            raise ValueError("Index must be in the range [0, 63]")
        point = []
        for i in reversed(range(6)):  # from most to least significant digit
            digit = (index // (2 ** i)) % 2
            point.append(2 * digit - 1)

        return np.array(point)

    data = []

    for n in range(num_cubes):
        region = []
        special_point = min_length + (n + 0.5) * diagonal_length
        init = DQClass.DQuaternion(DQClass.IdentityQ(), DQClass.Quaternion(0, 0, 0, special_point - id_lengths))
        for i in range(64):
            lengths = index_to_point(i) * edge_length / 2
            lengths = [special_point + lengths[k] for k in range(6)]
            pose = PF.PoseFinder(init, lengths, TableID, Base)
            pose = pose.ToFullVec()
            region.append(pose)
        data.append(region)

    data = np.array(data, dtype=np.float32)
    print(data)
    np.save("data_100dag.npy", data)


def initial_guess(lengths):
    smallest = min(lengths)
    n = max(0, min(num_cubes - 1, math.floor((smallest - min_length) / diagonal_length)))
    def legs_to_point():
        """Convert a 6D array of leg lengths to an input point."""
        special_point = min_length + (n + 0.5) * diagonal_length
        v = [0,0,0,0,0,0]
        for i in range(6):
            v[i] = sign(lengths[i] - special_point)
        return [n, v]

    def point_to_index(point):
        """Convert a 6D point to an integer index [0, 63]"""
        index = 0
        for i, val in enumerate(point[1]):
            digit = (val + 1) >> 1
            index += digit << (5 - i)
        return (n, index)

    return point_to_index(legs_to_point())


def PoseFinder(data_loaded, lengths, TableID, Base):
    index = initial_guess(lengths)
    init = data_loaded[index]
    init = DQClass.ToDualQuaternion(init)
    return PF.PoseFinder(init, lengths, TableID, Base)


import RandomizerRedacted as rr
identity_pose = DQClass.IdentityDQ()
Table = rr.TableID
Base = rr.Base

# offline_phase(Table, Base)

data_loaded = np.load("data_100dag.npy")
for _ in range(100):
    random_pose = rr.MakeRandomPose()
    random_lengths = rr.LegLengthsRedacted(random_pose)
    init = DQClass.ToDualQuaternion(data_loaded[initial_guess(random_lengths)])
    computed_pose = PF.PoseFinder(DQClass.IdentityDQ(), random_lengths, Table, Base)
    print("Random Pose: ", random_pose)
    print("Initial Guess: ", init)
    print("Computed Pose: ", computed_pose)
    if np.linalg.norm(np.array((random_pose - computed_pose).ToFullVec())) > .5:
        print("WRONG!")
        break