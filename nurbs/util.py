import numpy as np


def norm(V):
    ''' Find the norm of the vector V (faster than np.linalg.norm).  '''
    return np.sqrt(np.dot(V, V))

def normalize(V):
    ''' Normalize the vector V. '''
    return V / norm(V)

def distance(P1, P2):
    ''' Calculate the distance between two points. '''
    return norm(np.asfarray(P2) - P1)

def distance_v(P1, P2):
    ''' Idem distance, vectorized in P1. '''
    P12 = np.asfarray(P2).reshape((3, 1)) - P1
    return np.sqrt(np.sum(P12**2, axis=0))

def intersect_3D_lines(P0, T0, P2, T2):
    ''' Find the intersection point of two lines in three-dimensional
    space. '''
    V0, V1 = [normalize(T) for T in T0, T2]
    c = np.cross(V0, V1)
    den = norm(c)**2
    if np.allclose(den, 0.0):
        raise ParallelLines(V0, V1)
    P20 = P2 - P0
    t = np.linalg.det((P20, V1, c)) / den
    s = np.linalg.det((P20, V0, c)) / den
    P11, P12 = P0 + V0 * t, P2 + V1 * s
    if not np.allclose(P11, P12):
        raise NonIntersectingLines(P11, P12)
    return P11

def intersect_line_plane(L0, L, P0, N):
    ''' Find the intersection point of a line and a plane. '''
    L, N = [normalize(T) for T in L, N]
    num = np.dot(P0 - L0, N)
    den = np.dot(L, N)
    if np.allclose(den, 0.0):
        if np.allclose(num, 0.0):
            raise LineIntersectPlaneEveryWhere()
        else:
            raise NonIntersectingLinePlane()
    d = num / den
    return L0 + d * L

def intersect_three_planes(P0, N0, P1, N1, P2, N2):
    ''' Find the intersection point between three planes. '''
    N0, N1, N2 = [normalize(N) for N in N0, N1, N2]
    N1N2 = np.cross(N1, N2)
    den = np.dot(N0, N1N2)
    if np.allclose(den, 0.0):
        raise ParallelPlanes()
    num = np.dot(P0, N0) * N1N2 + \
          np.dot(P1, N1) * np.cross(N2, N0) + \
          np.dot(P2, N2) * np.cross(N0, N1)
    return num / den

def angle(P, R, Q=[0,0,0]):
    ''' Return the angle (in deg) between the vectors QP and QR. '''
    P, R, Q = [np.asfarray(V) for V in P, R, Q]
    QP, QR = [normalize(V - Q) for V in P, R]
    dot = np.dot(QP, QR)
    if dot < - 1.0 and np.allclose(dot, - 1.0):
        dot = - 1.0
    elif dot > 1.0 and np.allclose(dot, 1.0):
        dot = 1.0
    ang = np.arccos(dot)
    return np.rad2deg(ang)

def signed_angle(A, B):
    ''' Find the signed, shortest angle (in degrees) between the 2D
    vectors A and B. '''
    ang = np.arctan2(np.cross(A, B), np.dot(A, B))
    return np.rad2deg(ang)

def point_to_line(S, T, P):
    ''' Project a point P onto a line defined by the point S and
    direction vector T. '''
    S, T, P = [np.asfarray(V) for V in S, T, P]
    T = normalize(T)
    return S + np.dot(T, P - S) * T

def point_to_plane(S, T, P):
    ''' Project a point P onto a plane defined by the point S and normal
    vector T. '''
    S, T, P = [np.asfarray(V) for V in S, T, P]
    T = normalize(T)
    return P - np.dot(T, P - S) * T

def eval_ders_trigo(u, k):
    ''' Evaluate all d derivatives up to and including the kth (0 <= d
    <= k) of the unit circle (cos(u), sin(u), 0). '''
    ders = np.zeros((k + 1, 3))
    deri = {0: (' np.cos', ' np.sin'), 1: ('-np.sin', ' np.cos'),
            2: ('-np.cos', '-np.sin'), 3: (' np.sin', '-np.cos')}
    for i in xrange(min(k + 1, 4)):
        fx, fy = deri[i]
        ders[i,:2] = (eval('{0}({1})'.format(fx, u)),
                      eval('{0}({1})'.format(fy, u)))
    if k > 3:
        for i in xrange(4, k + 1):
            ders[i] = ders[i-4]
    return ders

def bounds(*os):
    ''' Return the global min/max bounds of the given Points, Curves,
    Surfaces and/or Volumes. '''
    bs = os[0].bounds if os else [(0,0), (0,0), (0,0)]
    if len(os) > 1:
        for o in os[1:]:
            b = o.bounds
            for i in xrange(3):
                bs[i] = (min(bs[i][0], b[i][0]),
                         max(bs[i][1], b[i][1]))
    return bs

def construct_flat_grid(Us, nums=None):
    ''' Construct a flattened, nonuniform (or optionally uniform)
    n-dimensional parametric grid.  Tried to use mgrid, but, unlike with
    linspace, I encountered floating point inaccuracies. '''
    Us = [np.asfarray(U) for U in Us]
    if nums is not None:
        Us = [np.linspace(U[0], U[-1], num)
              for U, num in zip(Us, nums)]
    if len(Us) == 1:
        return Us
    if len(Us) == 2:
        U, V = Us
        numu, numv = len(U), len(V)
        us = U[:,np.newaxis]
        vs = V[np.newaxis,:]
        us = us.repeat(numv, axis=1)
        vs = vs.repeat(numu, axis=0)
        return [u.flatten() for u in (us, vs)]
    if len(Us) == 3:
        U, V, W = Us
        numu, numv, numw = len(U), len(V), len(W)
        us = U[:,np.newaxis,np.newaxis]
        vs = V[np.newaxis,:,np.newaxis]
        ws = W[np.newaxis,np.newaxis,:]
        us = us.repeat(numv, axis=1).repeat(numw, axis=2)
        vs = vs.repeat(numu, axis=0).repeat(numw, axis=2)
        ws = ws.repeat(numu, axis=0).repeat(numv, axis=1)
        return [u.flatten() for u in (us, vs, ws)]


# EXCEPTIONS


class UtilException(Exception):
    pass

class ParallelLines(UtilException):
    pass

class ParallelPlanes(UtilException):
    pass

class NonIntersectingLines(UtilException):
    pass

class LineIntersectPlaneEveryWhere(UtilException):
    pass

class NonIntersectingLinePlane(UtilException):
    pass
