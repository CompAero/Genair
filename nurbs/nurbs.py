import numpy as np

import knot
import point


class ControlObject(object):

    ''' A ControlObject represents the (ordered) set of control Points
    of a NURBSObject.

    '''

    def __init__(self, cpts, Pw):

        ''' Initialize the ControlObject with either (not both) a list
        of Point objects or an object matrix.

        An object matrix, denoted by Pw, is a matrix that contains the
        4D homogenous coordinates of all control points constituting the
        ControlObject.  Hence, for curves, Pw[i,:] contains Pi = (wi*xi,
        wi*yi, wi*zi, wi), and likewise for surfaces, Pw[i,j,:], and
        volumes, Pw[i,j,k,:].  Its dimension is ((n + 1) x 4) for a
        ControlPolygon (Curve), ((n + 1) x (m + 1) x 4) for a ControlNet
        (Surface) and ((n + 1) x (m + 1) x (l + 1) x 4) for a
        ControlVolume (Volume).

        NOTE: the xyzw coordinates of a control Point are only shallow
        copy (views) on the relevant data of the object matrix.  Thus,
        as long as modifications are performed in-place, manipulating a
        control Point on a ControlObject will automatically update its
        object matrix (and vice versa).

        '''

        if cpts is not None:
            cpts = np.asarray(cpts)
            Pw = point.points_to_obj_mat(cpts)
        elif Pw is not None:
            Pw = np.asfarray(Pw)
            cpts = point.obj_mat_to_points(Pw)
        n = np.array(cpts.shape) - 1
        self._n = tuple(n)
        self.cpts = cpts; self.Pw = Pw

        # Reset all weights deviating by less than 1.0 +/- TOL to 1.0
        TOL = 1e-8; w = Pw[...,-1]
        m = ((1.0 - TOL) <= w) & (w <= (1.0 + TOL)); w[m] = 1.0

    def __getstate__(self):
        ''' Pickling. '''
        d = self.__dict__.copy()
        ds = d.viewkeys() - {'Pw', '_n'}
        for k in ds:
            del d[k]
        return d

    def __setstate__(self, d):
        ''' Unpickling. '''
        self.__dict__.update(d)
        self.cpts = point.obj_mat_to_points(d['Pw'])

    @property
    def n(self):
        ''' There are (n + 1) control Points. '''
        return self._n

    @property
    def bounds(self):
        ''' Return the xyz min/max bounds. '''
        m = self.Pw.reshape((-1, 4))
        m = obj_mat_to_3D(m)
        return zip(np.min(m, axis=0), np.max(m, axis=0))

    def copy(self):
        ''' Self copy. '''
        Pw = self.Pw.copy()
        return self.__class__(Pw=Pw)


class NURBSObject(object):

    ''' A NURBSObject is meant to be subclassed into either a NURBS
    Curve, Surface or Volume.  It is fully defined by a ControlObject,
    degree(s) and accompanying knot vector(s).  If no knot vector(s) is
    specified, a uniform knot vector(s) is used.

    '''

    def __init__(self):
        ''' Initialize the NURBSObject. '''
        super(NURBSObject, self).__init__()
        self._set_cpoint_association()

    def __getstate__(self):
        ''' Pickling. '''
        d = self.__dict__.copy()
        ds = d.viewkeys() - {'_cobj', '_p', '_U'}
        for k in ds:
            del d[k]
        return d

    def _set_cpoint_association(self):
        ''' Give every control Point a pointer to self. '''
        for cpt in self.cobj.cpts.flat:
            cpt.nurbs = self

    @property
    def cobj(self):
        ''' Get the ControlObject. '''
        return self._cobj

    @property
    def p(self):
        ''' Get the degree(s). '''
        return self._p

    @property
    def U(self):
        ''' Get the knot vector(s). '''
        return self._U

    @U.setter
    def U(self, new_U):
        ''' Set the knot vector(s). '''
        new_U = [np.asfarray(U).copy() for U in new_U]
        for n, p, U in zip(self.cobj.n, self.p, new_U):
            knot.clean_knot_vec(U)
            knot.check_knot_vec(n, p, U)
        self._U = tuple(new_U)

    @property
    def bounds(self):
        ''' Get the min/max bounds of the ControlObject. '''
        return self.cobj.bounds

    @property
    def isrational(self):
        ''' Is the NURBSObject rational? '''
        return (self.cobj.Pw[...,-1] != 1.0).any()

    def isequivalent(self, n, TOL=1e-8):
        ''' Is self equivalent to another NURBSObject? '''
        P = [obj_mat_to_3D(Pw) for Pw in self.cobj.Pw, n.cobj.Pw]
        if P[0].shape != P[1].shape:
            return False
        m = ()
        for v in [P] + zip(self.U, n.U):
            m += np.allclose(*v, atol=TOL),
        return all(m)

    def var(self):
        ''' Return copies of internal variables. '''
        v = ()
        for n, p, U in zip(self.cobj.n, self.p, self.U):
            v += n, p, U.copy()
        v += self.cobj.Pw.copy(),
        return v

    def copy(self):
        ''' Self copy. '''
        return self.__class__(self.cobj, self.p, self.U)


def obj_mat_to_3D(Pw):
    ''' Convert a 4D (homogeneous) object matrix to a 3D object matrix,
    i.e. a (... x 4) to a (... x 3) matrix. '''
    Pw = np.asfarray(Pw)
    if Pw.shape[-1] == 3:
        return Pw
    w = Pw[...,-1]
    return Pw[...,:-1] / w[...,np.newaxis]

def obj_mat_to_4D(P, w=None):
    ''' Idem obj_mat_to_3D, vice versa.  If w is None, all weights are
    set to unity, otherwise it is assumed that w has one less dimension
    than P. '''
    P = np.asfarray(P); s = P.shape
    if s[-1] == 4:
        return P
    Pw = np.ones(list(s[:-1]) + [4])
    Pw[...,:-1] = P
    if w is not None:
        Pw *= w[...,np.newaxis]
    return Pw


# EXCEPTIONS


class NURBSException(Exception):
    pass

class TooFewControlPoints(NURBSException):
    pass

class NewtonLikelyDiverged(NURBSException):
    pass

class RationalNURBSObjectDetected(NURBSException):
    pass
