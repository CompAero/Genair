import functools

import numpy as np

import nurbs
import point
import util

import plot.pobject


def custom(Pw, R=[[1,0,0],
                  [0,1,0],
                  [0,0,1]], T=[0,0,0], P=[0,0,0], c33=1):

    ''' Transform Pw (IN-PLACE).

    '''

    m = Pw.reshape((-1, 4)).T
    C = np.identity(4)
    C[:3,:3] = R
    C[:3,3] = T
    C[3,:3] = P
    C[3,3] = c33
    m = np.dot(C, m).T
    m.shape = Pw.shape; Pw[:] = m

def translate(Pw, w):

    ''' Translate Pw (IN-PLACE).

        w = the translation vector

    Source: Goldman, Matrices and transformations, Graphics Gems I,
            1990.

    '''

    m = Pw.reshape((-1, 4)).T
    T = np.identity(4)
    T[:3,3] = w
    m = np.dot(T, m).T
    m.shape = Pw.shape; Pw[:] = m

def rotate(Pw, theta, L=(0,1,0), Q=(0,0,0)):

    ''' Rotate Pw, clockwise (IN-PLACE).

        theta = the angle of rotation (in degrees)
        L = the axis line (default: (0,1,0))
        Q = a point on L (default: (0,0,0))

    Source: Goldman, Matrices and transformations, Graphics Gems I,
            1990.

    '''

    m = Pw.reshape((-1, 4)).T
    w = util.normalize(L)
    theta = np.deg2rad(theta)
    R = np.identity(4)
    S = np.array([[   0 , -w[2],  w[1]],
                  [ w[2],    0 , -w[0]],
                  [-w[1],  w[0],    0 ]])
    M = np.cos(theta) * np.identity(3)
    M += (1.0 - np.cos(theta)) * np.outer(w, w)
    M += np.sin(theta) * S
    R[:3,:3] = M
    R[:3,3] = Q - np.dot(M, Q)
    m = np.dot(R, m).T
    m.shape = Pw.shape; Pw[:] = m

def mirror(Pw, N=(0,1,0), Q=(0,0,0)):

    ''' Mirror Pw (IN-PLACE).

        N = a unit vector perpendicular to S (default: (0,1,0))
        Q = a point on S (default: (0,0,0))

    Source: Goldman, Matrices and transformations, Graphics Gems I,
            1990.

    '''

    m = Pw.reshape((-1, 4)).T
    N = util.normalize(N)
    M = np.identity(4)
    M[:3,:3] -= 2 * np.outer(N, N)
    M[:3,3] = 2 * np.dot(Q, N) * N
    m = np.dot(M, m).T
    m.shape = Pw.shape; Pw[:] = m

def scale(Pw, c, Q=(0,0,0), L=None):

    ''' Scale Pw (IN-PLACE).

        c = the scaling factor
        Q = the scaling origin (default: (0,0,0))
        L = a unit vector parallel to scaling direction (default: None)

    Source: Goldman, Matrices and transformations, Graphics Gems I,
            1990.

    '''

    m = Pw.reshape((-1, 4)).T
    S = np.identity(4)
    if L is None: # uniform
        S[:3,:3] *= c
        S[:3,3] = (1.0 - c) * np.asarray(Q)
    else: # nonuniform
        w = util.normalize(L)
        S[:3,:3] -= (1.0 - c) * np.outer(w, w)
        S[:3,3] = (1.0 - c) * np.dot(Q, w) * w
    m = np.dot(S, m).T
    m.shape = Pw.shape; Pw[:] = m

def shear(Pw, phi, v=(0,1,0), w=(1,0,0), Q=(0,0,0)):

    ''' Shear Pw (IN-PLACE).

    A shear is defined in terms of a shearing plane S, a vector w in S,
    and an angle phi.  Given any point P, project P orthogonally onto a
    point P' in the shearing plane S.  Now, slide P parallel to w to a
    point P'', so that angle(P'',P',P) = phi.  The point P'' is the
    result of applying the shearing transformation to the point P.

                                        _____
                                      P|_|   / P''
                   ____v|______________|____/_______
                  |     |_             |phi/        |
                  |     |_|___         |  /         |
                  |    Q     w        _| /          |
                  |                  | |/ P'        |
                  |                              S  |
                  |_________________________________|

        phi = the shear angle
        v = a unit vector perpendicular to S (default: (0,1,0))
        w = a unit vector in S (i.e., unit vector perpendicular to v)
            (default: (1,0,0))
        Q = a point on S (default: (0,0,0))

    Source: Goldman, More matrices and transformations: shear and
            pseudo-perspective, Graphics Gems II, 1990.

    '''

    m = Pw.reshape((-1, 4)).T
    v, w = [util.normalize(v) for v in v, w]
    phi = np.deg2rad(phi)
    S = np.identity(4)
    S[:3,:3] += np.tan(phi) * np.outer(w, v)
    S[:3,3] = - np.tan(phi) * np.dot(Q, v) * w
    m = np.dot(S, m).T
    Pw[:] = m.reshape(Pw.shape)


def transform(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'glued') and self.glued:
            os = set(self.glued)
        else:
            os = {self}
        for o in os:
            if isinstance(o, point.Point):
                co = o._xyzw
            elif isinstance(o, nurbs.NURBSObject):
                co = o.cobj.Pw
            else:
                print('nurbs.transform.transform :: '
                      'could not transform ({}); glue it first?'
                      .format(o))
                continue
            func(co, *args, **kwargs)
        plot.pobject.update_figures(os)
    return wrapper

point.Point.translate = transform(translate)
point.Point.rotate    = transform(rotate)
point.Point.mirror    = transform(mirror)
point.Point.scale     = transform(scale)
point.Point.shear     = transform(shear)

nurbs.NURBSObject.translate = transform(translate)
nurbs.NURBSObject.rotate    = transform(rotate)
nurbs.NURBSObject.mirror    = transform(mirror)
nurbs.NURBSObject.scale     = transform(scale)
nurbs.NURBSObject.shear     = transform(shear)
