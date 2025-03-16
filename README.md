# Quaternion-Kinematics
This is a collection of methods that solve the forward kinematics problem for parallel robots. (Namely the Stewart-Gough platform).
DQClass defines the quaternions and dual quaternions.
Randomizer Redacted generates random poses and leg lengths. It is a modification of the previous randomizer and it protects proprietary data.
Pose Tester is a way to test the program and see how long it takes to run a particular version of the solution
V1 Uses direct formulas with quaternions
V6_2 uses Lie algebras of dual quaternions. V6_2 uses the redacted table data, so it came second, hence the naming.
V7 is a modification of V6_2 that uses a less numerically taxing version of Newton's method. I just wanted to see what fewer matrix inversions would be like. It is far less efficient.
V9 uses neural networks. But it is not up yet.
