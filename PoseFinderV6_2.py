# I realized that the table data may be proprietary, so I went in and manually computed the base and table coordinates
# based solely off of the data that is already on the Mikrolar website and a little guesswork.

import numpy as np
import DQClass


# init is a dual quaternion pose, and lengths is an array of 6 floats. Output is a dual quaternion pose.
def PoseFinder(init, lengths, TableID, Base):
    X, Y = init, DQClass.ZeroDQ()
    hhh = 0
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
        hhh += 1
    return [X, hhh]
