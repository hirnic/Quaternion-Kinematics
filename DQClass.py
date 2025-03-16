# This is a program that carries out computations for kinematics using dual quaternions.
import math


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2+self.z**2)

    def inverse(self):
        conjugate = self.conjugate()
        norm = self.norm()**2
        try:
            return Quaternion(conjugate.w / norm, conjugate.x / norm, conjugate.y / norm, conjugate.z / norm)
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def __truediv__(self, other):
        return self * other.inverse()

    def normalization(self):
        norm = self.norm()
        try:
            return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
        except ZeroDivisionError:
            return Quaternion(0, 0, 0, 0)

    def ImaginaryPart(self):
        return Quaternion(0, self.x, self.y, self.z)

    def RealPart(self):
        return Quaternion(self.w, 0, 0, 0)

    def DotProduct(self, other):
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def ParallelPart(self, other):  # gives the projection of self onto other
        d = other.normalization()
        return d * Quaternion(d.DotProduct(self), 0, 0, 0)

    def OrthogonalPart(self, other):  #give the projection of self onto orthogonal component of other
        return self - self.ParallelPart(other)

    def Exponential(self):
        a = self.ImaginaryPart()
        m = a.norm()
        r = self.w
        C = Quaternion(math.cos(m), 0, 0, 0)
        S = Quaternion(math.sin(m), 0, 0, 0)
        M = Quaternion(math.exp(r), 0, 0, 0)
        return M * (C + S * a)

    def Logarithm(self):
        N = self.norm()
        Q = self.normalization()
        Re = Q.w
        Im = Q.ImaginaryPart().normalization()
        return Quaternion(math.log(N), 0, 0, 0) + Quaternion(math.atan2(N, Re), 0, 0, 0) * Im

    def ToFullVec(self):
        return [self.w, self.x, self.y, self.z]

    def ToPureVec(self):
        return [self.x, self.y, self.z]

    def PoseIt(self, other):  # Other must be a pose.
        return other.Q * self * other.Q.conjugate()  + other.T

    def Rotate(self, other): # Self is the vector being rotated, other is the rotation quaternion
        return other * self * other.conjugate()


# Example usage
q1 = Quaternion(0, 2*math.pi, 0, 0)
q2 = Quaternion(-3, -9, -15, -21)
# print("q1:", q1)
# print("q2:", q2)
# print("Sum:", q1 + q2)
# print("Product:", q1 * q2)
# print("Conjugate of q1:", q1.conjugate())
# print("Norm of q1:", q1.norm())
# print("Quotient:", q1/q2)
# print("Normalization of q1:", q1.normalization())
# print("Inverse of q1:", q1.inverse())
# print("Real part of q1:", q1.RealPart())
# print("Imaginary part of q1:", q1.ImaginaryPart())
# print("Dot product of q1 and q2:", q1.DotProduct(q2))
# print("Component of q1 parallel to q2:", q1.ParallelPart(q2))
# print("Component of q1 orthogonal to q2:", q1.OrthogonalPart(q2))
# print("Exponential of q1:", q1.Exponential())
# print("Logarithm of q1:", q1.Logarithm())


class DQuaternion:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"DQuaternion({self.A}, {self.B})"

    def __add__(self, other):
        return DQuaternion(self.A + other.A, self.B + other.B)

    def __sub__(self, other):
        return DQuaternion(self.A - other.A, self.B - other.B)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DQuaternion(self.A * other, self.B * other)
        return DQuaternion(self.A * other.A, self.A * other.B + self.B * other.A)

    def conjugate(self):
        return DQuaternion(self.A.conjugate(), self.B.conjugate())

    def overbar(self):
        return DQuaternion(self.A, Quaternion(-1, 0, 0, 0) * self.B)

    def inverse(self):
        try:
            return DQuaternion(self.A.inverse(), self.A.inverse() * self.B * self.A.inverse()).overbar()
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def __truediv__(self, other):
        return self * other.inverse()

    def norm(self):
        try:
            R = self.A.norm()
            D = self.A.DotProduct(self.B)/self.A.norm()
            if self.A == Quaternion(0, 0, 0, 0):
                return DQuaternion(Quaternion(0, 0, 0, 0,), Quaternion(0, 0, 0, 0))
            else:
                return DQuaternion(Quaternion(R, 0, 0, 0), Quaternion(D, 0, 0, 0))
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def normalization(self):
        try:
            R = self.A.norm()
            D1 = self.B / Quaternion(R, 0, 0, 0)
            D2 = Quaternion(self.A.DotProduct(self.B), 0, 0, 0) * self.A / Quaternion(R ** 3, 0, 0, 0)
            return DQuaternion(self.A.normalization(), D1 - D2)
        except ZeroDivisionError:
            return DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0))

    def Exponential(self):
        c = self.A.w
        d = self.A.ImaginaryPart()
        x = self.B.w
        y = self.B.ImaginaryPart()
        yPar = y.ParallelPart(d)
        yPerp = y.OrthogonalPart(d)
        a = yPar.normalization()
        b = yPerp.normalization()
        w = 2 * d.norm()
        v1 = 2 * yPar.norm()
        v2 = 2 * yPerp.norm()
        scale = DQuaternion(Quaternion(math.exp(c), 0, 0, 0), Quaternion(x * math.exp(c), 0, 0, 0))
        if w == 0:
            return scale * DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(1 / 2 * v1, 0, 0, 0) * a)
        else:
            vectorND = Quaternion(math.cos(1 / 2 * w), 0, 0, 0) + Quaternion(math.sin(1 / 2 * w), 0, 0, 0) * a
            vectorD = vectorND * Quaternion(1 / 2 * v1, 0, 0, 0) * a + Quaternion(v2 / w * math.sin(1 / 2 * w), 0, 0, 0) * b
            return scale * DQuaternion(vectorND, vectorD)

    def Logarithm(self):
        NDPart = self.A
        norm = math.sqrt(NDPart.w**2 + NDPart.ImaginaryPart().norm()**2)
        hat = self.normalization()
        a = hat.A.ImaginaryPart()
        c = hat.A.w
        s = a.norm()
        t = math.atan2(s, c)
        x = hat.B.w
        yPar = hat.B.ImaginaryPart().ParallelPart(a)
        yPerp = hat.B.ImaginaryPart().OrthogonalPart(a)
        b = yPerp.normalization()
        y1 = yPar.norm()
        y2 = 0
        tysb = 0
        if s == 0:
            tysb = Quaternion(0, 0, 0, 0)
        else:
            y2 = yPerp.norm()
            tysb = Quaternion(t * y2 / s, 0, 0, 0) * b
        scale = DQuaternion(Quaternion(math.log(norm), 0, 0, 0), Quaternion(0, 0, 0, 0))
        vectorND = Quaternion(t, 0, 0, 0) * a
        vectorD = Quaternion(c * y1 - s * x, 0, 0, 0) * a + tysb
        return scale + DQuaternion(vectorND, vectorD)

    def slerp(self, other, t):
        T = DQuaternion(Quaternion(t, 0, 0, 0), Quaternion(0, 0, 0, 0))
        try:
            A = self.conjugate() * other
            L = T * A.Logarithm()
            return self * L.Exponential()
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def ToFullVec(self):
        return [self.A.w, self.A.x, self.A.y, self.A.z, self.B.w, self.B.x, self.B.y, self.B.z]

    def To6Vec(self):
        return [self.A.x, self.A.y, self.A.z, self.B.x, self.B.y, self.B.z]

    def size(self):
        return math.sqrt(self.A.norm()**2 + self.B.norm()**2)


def IdentityQ():
    return Quaternion(1, 0, 0, 0)


def IdentityDQ():
    return DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0))


def BasisDQ():
    X = [DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 1, 0, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 1, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(1, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 1, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 1, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 1))]
    return X


def ZeroDQ():
    return DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0))


def ToQuaternion(x):  # x must be a 3-vector
    if len(x) != 4:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 4.")
    else:
        return Quaternion(x[0], x[1], x[2], x[3])


def ToVectorQuaternion(x):  # x must be a 3-vector
    if len(x) != 3:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 3.")
    else:
        return Quaternion(0, x[0], x[1], x[2])

def ToQuaternionRotation(x):  # x must be a 3-vector of norm < 1
    if len(x) != 3:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 3.")
        return
    if x[0]**2 + x[1]**2 + x[2]**2 > 1:
        print("Rotation vectors must have norm less than or equal to 1!")
        return
    else:
        S = Quaternion(0, x[0], x[1], x[2])
        try:
            C = math.sqrt(1 - S.x**2 - S.y**2 - S.z**2)
        except:
            C = 0
        return Quaternion(C, 0, 0, 0) + S

def ToQuaternionTranslation(x):  # x must be a 3-vector
    if len(x) != 3:
        print("You cannot turn this vector into a translation quaternion. The input must be a 1D array of length 3.")
    else:
        return Quaternion(0, x[0], x[1], x[2])


def ToDualQuaternion(x):
    if len(x) != 8:
        print("You cannot turn this vector into a dual quaternion. Input must be a 1D array of length 8.")
    else:
        Q = ToQuaternion(x[:4])
        t = ToQuaternion(x[4:])
        return DQuaternion(Q, t)


def ToVectorDualQuaternion(x):
    if len(x) != 6:
        print("You cannot turn this vector into a unit dual quaternion. Input must be a 1D array of length 6.")
    else:
        Q = ToVectorQuaternion(x[:3])
        t = ToQuaternionTranslation(x[3:])
        return DQuaternion(Q, t)


def ToPoseDualQuaternion(x):
    if len(x) != 6:
        print("You cannot turn this vector into a unit dual quaternion. Input must be a 1D array of length 6.")
    else:
        Q = ToQuaternionRotation(x[:3])
        t = ToQuaternionTranslation(x[3:])
        B = t * Q * (1 / 2)
        return DQuaternion(Q, B)


d1 = DQuaternion(Quaternion(0, 0, 0, 1), Quaternion(2, 0, 0, 0))
d2 = DQuaternion(Quaternion(7, 3.14, 9, 10), Quaternion(2, 2, 2.1, 1.3))


# print("d1:", d1)
# print("d2:", d2)
# print("d1 + d2:", d1 + d2)
# print("d1*d2:", d1 * d2)
# print("d1-d2:", d1 - d2)
# print("Conjugate of d1:", d1.conjugate())
# print("Alternative Conjugate of d1:", d1.overbar())
# print("Inverse of d1:", d1.inverse())
# print("d1/d2:", d1 / d2)
# print("Norm of d1:", d1.norm())
# print("Normalization of d1:", d1.normalization())
# print("Exponential of d1:", d1.Exponential())
# print("Logarithm of d1:", d1.Logarithm())
# print("SLERP(d1, d2, .5):", d1.slerp(d2, 0.5))


class Pose:
    def __init__(self, Q, T):
        self.Q = Q
        self.T = T

    def __repr__(self):
        return f"Pose({self.Q}, {self.T})"

def ToPose(x):
    Q = ToQuaternionRotation(x[:3])
    T = ToQuaternionTranslation(x[3:])
    return Pose(Q,T)