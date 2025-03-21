# This file tests the PoseFinderV5
import DQClass
import Randomizer
import RandomizerRedacted as rr
import time

Init = DQClass.IdentityDQ()


def TestV1(iters=1000):  # Quaternion formulas
    import PoseFinderV1 as PF

    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(rr.MakeRandomPose())
        Legs.append(rr.LegLengthsRedacted(Poses[n]))

    PF.PoseFinder(Init, Legs[0], rr.TableID, rr.Base)
    computedPoses = []
    Now = time.time()
    for n in range(iters):
        # print("Iteration:", n)
        F = PF.PoseFinder(Init, Legs[n], rr.TableID, rr.Base)
        computedPoses.append(F)
    print("V1 Average Time: ", (time.time() - Now) / iters)
    absDist = 0
    for n in range(iters):
        absDist += (Poses[n] - computedPoses[n]).size()
    print("V1 Average Error : ", absDist / iters)


def TestV5(iters=1000):  # Lie Algebra fixed frame
    import PoseFinderV5 as PF
    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(Randomizer.MakeRandomPose())
        Legs.append(Randomizer.LegLengths(Poses[n]))

    Now = time.time()
    absDist = 0
    for n in range(iters):
        F = PF.PoseFinder(Init, Legs[n])
        absDist += (Poses[n] - F).size()
    print("V5 Time Taken: ", (time.time() - Now) / iters)
    print("V5 Average Error : ", absDist / iters)


def TestV6(iters=1000):  # Lie Algebra moving frame
    import PoseFinderV6
    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(Randomizer.MakeRandomPose())
        Legs.append(Randomizer.LegLengths(Poses[n]))

    Now = time.time()
    absDistV6 = 0
    for n in range(iters):
        # print("Iteration:", n)
        F = PoseFinderV6.PoseFinder(Init, Legs[n])
        absDistV6 += (Poses[n] - F).size()
    print("V6 Time Taken: ", (time.time() - Now) / iters)
    print("V6 Average Error : ", absDistV6 / iters)


def TestV6_2(iters=1000):  # Lie Algebra moving frame redacted data
    import PoseFinderV6_2 as PF

    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(rr.MakeRandomPose())
        Legs.append(rr.LegLengthsRedacted(Poses[n]))

    PF.PoseFinder(Init, Legs[0], rr.TableID, rr.Base)
    computedPoses = []
    Now = time.time()
    for n in range(iters):
        # print("Iteration:", n)
        F = PF.PoseFinder(Init, Legs[n], rr.TableID, rr.Base)
        computedPoses.append(F)
    print("V6_2 Average Time: ", (time.time() - Now) / iters)
    absDist = 0
    for n in range(iters):
        absDist += (Poses[n] - computedPoses[n]).size()
    print("V6_2 Average Error : ", absDist / iters)


def TestV7(iters=1000):  # Lie Algebra and Modified Newton Raphson
    import PoseFinderV7 as PF

    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(Randomizer.MakeRandomPose())
        Legs.append(Randomizer.LegLengths(Poses[n]))

    computedPoses = []
    Now = time.time()
    for n in range(iters):
        F = PF.PoseFinder(Init, Legs[n])
        computedPoses.append(F)
    print("V7 Average Time: ", (time.time() - Now) / iters)
    absDist = 0
    for n in range(iters):
        absDist += (Poses[n] - computedPoses[n]).size()
    print("V7 Average Error : ", absDist / iters)


def TestV8_3(iters=1000):  # Garbage Gradient Descent. Not working.
    import PoseFinderV8_3 as PF

    Legs = []
    Poses = []
    for n in range(iters):
        Poses.append(Randomizer.MakeRandomPose())
        Legs.append(Randomizer.LegLengths(Poses[n]))

    computedPoses = []
    Now = time.time()
    for n in range(iters):
        F = PF.PoseFinder(Init, Legs[n], rr.TableID, rr.Base)
        computedPoses.append(F)
    # print("V8 Average Time: ", (time.time() - Now) / iters)
    absDist = 0
    for n in range(iters):
        absDist += (Poses[n] - computedPoses[n]).size()
    # print("V8 Average Error : ", absDist / iters)
    print("Poses: ", Poses[0])
    print("Computed Poses: ", computedPoses[0])


def TestAll():
    TestV1()
    TestV6()
    TestV6_2()
    TestV7()


# TestV1()
# TestV5()
# TestV6()
# TestV6_2()
# TestV7()
# TestV8_3(1)

TestAll()
