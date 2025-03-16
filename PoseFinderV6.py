import numpy as np
import DQClass

Base = [[952.5055, 91.0723, -1410.0000], [-398.5396, 869.5826, -1409.8621],
        [-555.4801, 779.1038, -1410.0000], [-555.0219, -779.3507, -1409.6010],
        [-398.5396, -869.9006, -1410.0000], [952.7381, -89.7865, -1409.8718]]
Base = [DQClass.ToVectorQuaternion(x) for x in Base]
TableID = [[314.4868, 327.8608, -111.0000], [126.7447, 436.2739, -111.1102],
           [-441.2953, 107.9497, -111.0000], [-441.2826, -108.6562, -111.3975],
           [126.7447, -436.2688, -111.0000], [314.5916, -328.3827, -110.9675]]
TableID = [DQClass.ToVectorQuaternion(x) for x in TableID]

# init is a dual quaternion pose, and lengths is an array of 6 floats. Output is a dual quaternion pose.
def PoseFinder(init, lengths):
    X, Y = init, DQClass.ZeroDQ()
    while (X-Y).size() > 1e-4:
        Y = X
        f, Lf = [], []
        for n in range(6):                                                 # This loop finds the function and derivative
            s = (X.A.conjugate() * Base[n] + X.B.conjugate() * 2) * X.A           # Formula (8), page 4
            f.append((TableID[n] - s).norm() - lengths[n])                        # Formula (80), page 16
            u = np.array((TableID[n] - s).normalization().ToPureVec())            # Formula (66), page 13,
            Cross = np.cross(np.array(TableID[n].ToPureVec()), u)                 # Formula (67), page 13
            Lf.append(np.concatenate([Cross, u]) * 2)                             # Formula (68), page 14
        LInv = np.linalg.inv(np.array(Lf)) * (-1)
        Theta = DQClass.ToVectorDualQuaternion(LInv.dot(np.array(f)))             # Equation (80), page 16
        Hat = (DQClass.IdentityDQ() + Theta).normalization()                      # Formula (79), page 16
        X = X * Hat                                                               # Formula (79), page 16
    return X
