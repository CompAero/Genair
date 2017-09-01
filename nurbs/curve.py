''' A pth-degree NURBS curve is defined by

            sum_(i=0)^(n) (Nip(u) * wi * Pi)
    C(u) =  --------------------------------
              sum_(i=0)^(n) (Nip(u) * wi)

(a <= u <= b).  The {Pi} are the control points (forming a control
polygon), the {wi} are the weights, and the {Nip(u)} are the pth-degree
B-spline basis functions defined on the nonperiodic and nonuniform knot
vector

    U = {a,...,a, u_(p+1),...,u_(m-p-1), b,...,b}.

where (m = n + p + 1).

Homogeneous coordinates offer an efficient method of representing NURBS
curves.  For a given set of control points, {Pi}, and weights, {wi},
construct the weighted control points, Pwi = (wi*xi, wi*yi, wi*zi, wi).
Then define the nonrational (piecewise polynomial) B-spline curve in
four-dimensional space as

    Cw(u) = sum_(i=0)^(n) (Nip(u) * Pwi)

Applying a perspective map on the (w = 1) plane, H, to Cw(u) yields the
corresponding rational B-spline curve (piecewise rational in
three-dimensional space) i.e. C(u) = H{Cw(u)}.

'''

import numpy as np
from scipy.integrate import quad
from scipy.misc import comb

import basis
import knot
import nurbs
import point
import util

import plot.pobject


__all__ = ['Curve', 'ControlPolygon',
           'make_composite_curve',
           'make_linear_curve',
           'param_to_arc_length',
           'arc_length_to_param',
           'reparam_arc_length_curve']


class ControlPolygon(nurbs.ControlObject):

    def __init__(self, cpts=None, Pw=None):

        ''' See nurbs.nurbs.ControlObject.

        Parameters
        ----------
        cpts = the list of (control) Points
        Pw = the object matrix

        Examples
        --------
        >>> P0 = Point(-4, 2, 0)
        >>> P1 = Point(-2, 3, 1)
        >>> P2 = Point( 0, 4, 2)
        >>> cpol = ControlPolygon([P0, P1, P2])

        or, equivalently,

        >>> Pw = [(-4, 2, 0, 1), (-2, 3, 1, 1), (0, 4, 2, 1)]
        >>> cpol = ControlPolygon(Pw=Pw)

        '''

        super(ControlPolygon, self).__init__(cpts, Pw)


class Curve(nurbs.NURBSObject, plot.pobject.PlotCurve):

    def __init__(self, cpol, p, U=None):

        ''' See nurbs.nurbs.NURBSObject.

        Parameters
        ----------
        cpol = the ControlPolygon
        p = the degree (order p + 1) of the Curve
        U = the knot vector

        Examples
        --------
        >>> P0 = Point(1, 0, w=1)
        >>> P1 = Point(1, 1, w=1)
        >>> P2 = Point(0, 1, w=2)
        >>> cpol = ControlPolygon([P0, P1, P2])
        >>> c = Curve(cpol, (2,))

        '''

        n, = cpol.n
        p, = p
        if n < p:
            raise nurbs.TooFewControlPoints((n, p),)
        self._cobj = cpol.copy()
        self._p = p,
        if not U:
            U = knot.uni_knot_vec(n, p),
        self.U = U
        super(Curve, self).__init__()

    def __setstate__(self, d):
        ''' Unpickling. '''
        self.__dict__.update(d)
        super(Curve, self).__init__()

# EVALUATION OF POINTS AND DERIVATIVES

    def eval_point(self, u):

        ''' Evaluate a point.

        Parameters
        ----------
        u = the parameter value of the point

        Returns
        -------
        C = the xyz coordinates of the point

        '''

        n, p, U, Pw = self.var()
        return rat_curve_point(n, p, U, Pw, u)

    def eval_points(self, us):

        ''' Evaluate multiple points.

        Parameters
        ----------
        us = the parameter values of each point

        Returns
        -------
        C = the xyz coordinates of all points

        '''

        n, p, U, Pw = self.var()
        return rat_curve_point_v(n, p, U, Pw, us, len(us))

    def eval_derivatives(self, u, d):

        ''' Evaluate derivatives at a point.

        Parameters
        ----------
        u = the parameter value of the point
        d = the number of derivatives to evaluate

        Returns
        -------
        CK = all derivatives, where CK[k,:] is the derivative of C(u)
             with respect to u k times (0 <= k <= d)

        '''

        n, p, U, Pw = self.var()
        Aders = curve_derivs_alg1(n, p, U, Pw[:,:-1], u, d)
        if self.isrational:
            wders = curve_derivs_alg1(n, p, U, Pw[:,-1], u, d)
            return rat_curve_derivs(Aders, wders, d)
        return Aders

    def eval_curvature(self, u):

        ''' Evaluate the curvature at a point.

        Parameters
        ----------
        u = the parameter value of the point

        Returns
        -------
        M = the curvature

        '''

        CK = self.eval_derivatives(u, 2)
        CU, CUU = CK[1:]
        num = util.norm(np.cross(CU, CUU))
        den = util.norm(CU)**3
        return num / den

# KNOT INSERTION

    def insert(self, u, e):

        ''' Insert a knot multiples times.

        Parameters
        ----------
        u = the knot value to insert
        e = the number of times to insert u

        Returns
        -------
        Curve = the new Curve with u inserted e times

        '''

        n, p, U, Pw = self.var()
        if e > 0:
            u = knot.clean_knot(u)
            k, s = basis.find_span_mult(n, p, U, u)
            U, Pw = curve_knot_ins(n, p, U, Pw, u, k, s, e)
        return Curve(ControlPolygon(Pw=Pw), (p,), (U,))

    def split(self, u):

        ''' Split the Curve.

        Parameters
        ----------
        u = the parameter value at which to split the Curve

        Returns
        -------
        [Curve, Curve] = the two split Curves

        '''

        n, p, U, Pw = self.var()
        u = knot.clean_knot(u)
        if u == U[0] or u == U[-1]:
            return [self.copy()]
        k, s = basis.find_span_mult(n, p, U, u)
        r = p - s
        if r > 0:
            U, Pw = curve_knot_ins(n, p, U, Pw, u, k, s, r)
        Ulr = (np.append(U[:k+r+1], u), np.insert(U[k-s+1:], 0, u))
        Pwlr = Pw[:k-s+1], Pw[k-s:]
        return [Curve(ControlPolygon(Pw=Pw), (p,), (U,))
                for Pw, U in zip(Pwlr, Ulr)]

    def extend(self, l, end=False):

        ''' Extend the Curve.

        Parameters
        ----------
        l = the (estimated) length of the extension
        end = whether to extend the start or the end part of the Curve

        Returns
        -------
        Curve = the Curve extension

        Source
        ------
        Shetty and White, Curvature-continuous extensions for rational
        B-spline curves and surfaces, Computer-aided design, 1991.

        '''

        if end:
            Cext = self.reverse().extend(l)
            return Cext.reverse() if Cext else None
        n, p, U, Pw = self.var()
        u = arc_length_to_param(self, l)
        Cs = self.split(u)
        if len(Cs) == 1:
            return None
        Cext, dummy = Cs
        Q, N = self.eval_derivatives(U[0], 1)
        Cext.mirror(N, Q)
        U = Cext.U[0]; U -= U[0]
        return Cext.reverse()

# KNOT REFINEMENT

    def refine(self, X):

        ''' Refine the knot vector.

        Parameters
        ----------
        X = a list of the knots to insert

        Returns
        -------
        Curve = the refined Curve

        '''

        n, p, U, Pw = self.var()
        if len(X) != 0:
            if X == 'mid':
                X = knot.midpoints_knot_vec(U)
            X = knot.clean_knot(X)
            U, Pw = refine_knot_vect_curve(n, p, U, Pw, X)
        return Curve(ControlPolygon(Pw=Pw), (p,), (U,))

    def decompose(self):

        ''' Decompose the Curve into Bezier segments.

        Returns
        -------
        Cs = a list of Bezier segments

        '''

        n, p, U, Pw = self.var()
        nb, U, Pw = decompose_curve(n, p, U, Pw)
        Cs = []
        for b in xrange(nb):
            c = Curve(ControlPolygon(Pw=Pw[b]), (p,), (U[b],))
            Cs.append(c)
        return Cs

    def segment(self, u1, u2):

        ''' Segment the Curve.

        Let U be the new segmented knot vector, then moving any of the
        control Points i satisfying (u1 <= U[i]) and (U[i+p+1] <= u2)
        would only modify the part of the new Curve whose parametric
        region lies between u1 and u2.

        Parameters
        ----------
        u1, u2 = the parametric bounds

        Returns
        -------
        Curve = the segmented Curve

        '''

        n, p, U, dummy = self.var()
        u1, u2 = knot.clean_knot([u1, u2])
        ks = knot.segment_knot_vec(n, p, U, u1, u2)
        return self.refine(ks)

# KNOT REMOVAL

    def remove(self, u, num, d=1e-3):

        ''' Try to remove an interior knot multiple times.

        Parameters
        ----------
        u = the knot to remove
        num = the number of times to remove u (1 <= num <= s), where s
              is the multiplicity of u
        d = the maximum deviation allowed

        Returns
        -------
        e = the number of times u has been successfully removed
        Curve = the new Curve with u removed at most num times

        '''

        n, p, U, Pw = self.var()
        e = 0
        if num > 0:
            u = knot.clean_knot(u)
            k, s = basis.find_span_mult(n, p, U, u)
            e, U, Pw = remove_curve_knot(n, p, U, Pw, u, k, s, num, d)
        return e, Curve(ControlPolygon(Pw=Pw), (p,), (U,))

    def removes(self, d=1e-3):

        ''' Remove all removable interior knots.

        Parameters
        ----------
        d = the maximum deviation allowed

        Returns
        -------
        e = the number of knots that has been successfully removed
        Curve = the new Curve with all removable knots removed

        '''

        n, p, U, Pw = self.var()
        e, U, Pw = remove_curve_knots(n, p, U, Pw, d)
        return e, Curve(ControlPolygon(Pw=Pw), (p,), (U,))

# DEGREE ELEVATION

    def elevate(self, t):

        ''' Elevate the Curve's degree.

        Parameters
        ----------
        t = the number of degrees to elevate the Curve with

        Returns
        -------
        Curve = the degree elevated Curve

        '''

        n, p, U, Pw = self.var()
        if t > 0:
            U, Pw = degree_elevate_curve(n, p, U, Pw, t)
            p += t
        return Curve(ControlPolygon(Pw=Pw), (p,), (U,))

# DEGREE REDUCTION

    def reduce(self, d=1e-3):

        ''' Try to reduce the Curve's degree by one.

        Parameters
        ----------
        d = the maximum deviation allowed

        Returns
        -------
        success = whether or not degree reduction was successful
        Curve = the degree reduced Curve

        '''

        n, p, U, Pw = self.var()
        try:
            U, Pw = degree_reduce_curve(n, p, U, Pw, d)
            success = True; p -= 1
        except MaximumToleranceReached:
            success = False
        return success, Curve(ControlPolygon(Pw=Pw), (p,), (U,))

# MISCELLANEA

    def project(self, xyz, ui=None):

        ''' Project a point.

        Parameters
        ----------
        xyz = the xyz coordinates of a point to project
        ui = the initial guess for Newton's method

        Returns
        -------
        u = the parameter value of the projected point

        '''

        n, p, U, Pw = self.var()
        return curve_point_projection(n, p, U, Pw, xyz, ui)

    def reverse(self):

        ''' Reverse the Curve's direction.

        Returns
        -------
        Curve = the reversed Curve

        '''

        n, p, U, Pw = self.var()
        U, Pw = reverse_curve_direction(n, p, U, Pw)
        return Curve(ControlPolygon(Pw=Pw), (p,), (U,))


# HEAVY LIFTING FUNCTIONS

# From here on out most functions are the direct equivalent of the
# pseudo-algorithms found in 'The NURBS Book (2nd Ed.)', hence their
# not-so pythonic nature.  Ideally they should either be wrapped in
# Fortran or C, or, as some have already been, completely vectorized.


def curve_deriv_cpts(n, p, U, P, d, r1, r2):

    ''' Compute the control points of curve derivatives up to and
    including the dth derivative (d <= p).  On output, PK[k,i] is the
    ith control point of the kth derivative curve, where (0 <= k <= d)
    and (r1 <= i <= r2 - k).  If (r1 = 0) and (r2 = n), all control
    points are computed.

    Source: The NURBS Book (2nd Ed.), Pg. 98.

    '''

    r = r2 - r1
    PK = np.zeros((d + 1, r + 1, 3))
    PK[0,:r+1] = P[r1:r1+r+1]
    for k in xrange(1, d + 1):
        tmp = p - k + 1
        for i in xrange(r - k + 1):
            PK[k,i] = tmp * ((PK[k-1,i+1] - PK[k-1,i]) /
                             (U[r1+i+p+1] - U[r1+i+k]))
    return PK


def curve_derivs_alg1(n, p, U, P, u, d):

    ''' Compute curve derivatives up to and including the dth.  (d > p)
    is allowed, although the derivatives are 0 in this case (for
    nonrational curves); these derivatives are necessary for rational
    curves.  Output is the array CK, where CK[k,:] are the xyz component
    of the kth derivative (0 <= k <= d).

    Source: The NURBS Book (2nd Ed.), Pg. 93.

    '''

    CK = np.zeros((d + 1, 3))
    du = min(d, p)
    span = basis.find_span(n, p, U, u)
    nders = basis.ders_basis_funs(span, u, p, du, U)
    for k in xrange(du + 1):
        for j in xrange(p + 1):
            CK[k] += nders[k,j] * P[span-p+j]
    return CK


def curve_derivs_alg2(n, p, U, P, u, d):

    ''' Idem curve_derivs_alg1, but by using the control points of
    derivative curves yielded by curve_deriv_cpts.

    Source: The NURBS Book (2nd Ed.), Pg. 99.

    '''

    CK = np.zeros((d + 1, 3))
    du = min(d, p)
    span = basis.find_span(n, p, U, u)
    N = basis.all_basis_funs(span, u, p, U)
    PK = curve_deriv_cpts(n, p, U, P, du, span - p, span)
    for k in xrange(du + 1):
        for j in xrange(p - k + 1):
            CK[k] += N[j,p-k] * PK[k,j]
    return CK


def rat_curve_point(n, p, U, Pw, u):

    ''' Compute a point on a rational B-spline curve at a fixed u
    parameter value.

    Source: The NURBS Book (2nd Ed.), Pg. 124.

    '''

    Cw = np.zeros(4)
    span = basis.find_span(n, p, U, u)
    N = basis.basis_funs(span, u, p, U)
    for j in xrange(p + 1):
        Cw += N[j] * Pw[span-p+j]
    return Cw[:3] / Cw[-1]


def rat_curve_point_v(n, p, U, Pw, u, num):

    ''' Idem rat_curve_point, vectorized in u.  Makes use of array
    broadcasting (see numpy manual).

    '''

    u = np.asfarray(u)
    Cw = np.zeros((4, num))
    span = basis.find_span_v(n, p, U, u, num)
    N = basis.basis_funs_v(span, u, p, U, num)
    for j in xrange(p + 1):
        Cw += N[j] * Pw[span-p+j].T
    return Cw[:3] / Cw[-1]


def rat_curve_derivs(Aders, wders, d):

    ''' Given that Cw(u) has already been differentiated and its
    coordinates separated off into Aders and wders, this algorithm
    computes the point, C(u), and the derivatives, C^k(u), (1 <= k <=
    d).  The curve point is returned in CK[0,:] and the kth derivative
    is returned in CK[k,:].

    Source: The NURBS Book (2nd Ed.), Pg. 127.

    '''

    CK = np.zeros((d + 1, 3))
    for k in xrange(d + 1):
        v = Aders[k]
        for i in xrange(1, k + 1):
            v -= comb(k, i) * wders[i] * CK[k - i]
        CK[k] = v / wders[0]
    return CK


# FUNDAMENTAL GEOMETRIC ALGORITHMS


# Knot insertion.
#
#   Let Cw(u) = sum_(i=0)^(n) (Nip(u) * Pwi) be a NURBS curve defined on
#   U = {u0,...,um}.  Let ub in [ u_k, u_(k+1) ), and insert ub into U
#   to form the new knot vector Ub = {ub_0 = u_0,...,ub_k = u_k,ub_(k+1)
#   = ub,ub_(k+2) = u_(k+1),...,ub_(m+1) = u_m}.  Then Cw(u) also has a
#   representation of the form Cw(u) = sum_(i=0)^(n+1) (Nbip(u) * Qwi)
#   where the {Nbip(u)} are the pth-degree basis functions on Ub.  It is
#   important to note that knot insertion is really just a change of vector
#   space basis; the curve is not changed, either geometrically or
#   parametrically.
#
#   Some of its uses are:
#
#   - evaluating points and derivatives on curves and surfaces
#
#   - subdividing curves and surfaces
#
#   - adding control points in order to increase flexibility in shape
#     control (interactive design)
#
#   - extracting an isoparametric curve from a surface


def curve_knot_ins(n, p, UP, Pw, u, k, s, r):

    ''' Compute the new curve corresponding to the insertion ubar into
    [u_k, u_(k+1) ) r times, where it is assumed that (r + s <= p), s
    being the initial multiplicity of the knot.  Note that generally it
    makes no practical sense to have interior knot multiplicities
    greater than p.

    Source: The NURBS Book (2nd Ed.), Pg. 151.

    '''

    m = n + p + 1; nq = n + r
    UQ = np.zeros(m + r + 1)
    Qw = np.zeros((nq + 1, 4))
    Rw = np.zeros((p + 1, 4))
    UQ[:k+1] = UP[:k+1]
    UQ[k+1:k+r+1] = u
    UQ[k+r+1:] = UP[k+1:]
    Qw[:k-p+1] = Pw[:k-p+1]
    Qw[k-s+r:n+r+1] = Pw[k-s:n+1]
    Rw[:p-s+1] = Pw[k-p:k-s+1]
    for j in xrange(1, r + 1):
        L = k - p + j
        for i in xrange(p - j - s + 1):
            alpha = (u - UP[L+i]) / (UP[i+k+1] - UP[L+i])
            Rw[i] = alpha * Rw[i+1] + (1.0 - alpha) * Rw[i]
        Qw[L] = Rw[0]
        Qw[k+r-j-s] = Rw[p-j-s]
    Qw[L+1:k-s] = Rw[1:k-s-L]
    return UQ, Qw


def inv_curve_knot_ins(n, p, U, Pw, i, Q):

    ''' Suppose a point Q on the three-dimensional polygon leg,
    P_(i-1)P_i.  Denote the corresponding four-dimensional point on the
    leg, Li = Pw_(i_1)Pw_i, by Qw.  Then there exists a ub in [ u_i,
    u_(i+p) ) such that a single insertion of ub cause Qw to become a
    new control point and Q its projection.  The process of determining
    ub is called inverse knot insertion.

    Source: The NURBS Book (2nd Ed.), Pg. 154.

    '''

    p0 = Pw[i-1]; p1 = Pw[i]
    d0, d1 = [util.distance(Q, pt[:3] / pt[-1])
              for pt in p0, p1]
    s = (p0[-1] * d0) / (p0[-1] * d0 + p1[-1] * d1)
    return U[i] + s * (U[i+p] - U[i])


# Knot refinement.
#
#   Knot insertion concerns itself with inserting a single knot,
#   possibly multiple times.  It is often necessary to insert many knots
#   at once; this is called knot refinement.  Let Cw(u) = sum_(i=0)^(n)
#   (Nip(u) * Pwi) be defined on the knot vector U = {u0,...,um}., and
#   let X = {x0,...,xr} satisfy (x_i <= x_(i+1)) and (u0 < xi < um) for
#   all i.  The elements of X are to be inserted into U, and the
#   corresponding new set of control points, {Qwi}, i=0,...,n+r+1, is to
#   be computed.
#
#   The applications of knot refinement include:
#
#   - decomposition of B-splines curves and surfaces into their
#     constituent (Bezier) polynomial pieces
#
#   - merging of two or more knot vectors in order to obtain a set of
#     curves which are defined on one common knot vector
#
#   - obtaining polygonal (polyhedral) approximations to curves
#     (surfaces); refining knot vectors brings the control polygon (net)
#     closer to the curve (surface), and in the limit the polygon (net)
#     converges to the curve (surface).


def refine_knot_vect_curve(n, p, U, Pw, X):

    ''' Let Cw(u) be defined on the knot vector U = {u0,...,um}, and let
    X = {x0,...,xr} satisfy (x_i <= x_(i+1)) and (u0 < xi < um) for all
    i.  The elements of X are to be inserted into U, and the
    corresponding new set of control points, {Qwi}, i = 0,...,n+r+1, is
    to be computed.  New knots should be repeated in X with their
    multiplicities; e.g. if x and y (x < y) are to be inserted with
    multiplicities 2 and 3, respectively, then X = [x, x, y, y, y].

    Source: The NURBS Book (2nd Ed.), Pg. 164.

    '''

    X = np.asfarray(X)
    r = X.size - 1
    if r < 0: return U, Pw
    a = basis.find_span(n, p, U, X[0])
    b = basis.find_span(n, p, U, X[r]); b += 1
    ns = n + r + 1; m = n + p + 1
    Ubar = np.zeros(m + r + 2)
    Qw = np.zeros((ns + 1, 4))
    Qw[:a-p+1] = Pw[:a-p+1]
    Qw[r+b:n+r+2] = Pw[b-1:n+1]
    Ubar[:a+1] = U[:a+1]
    Ubar[b+p+r+1:m+r+2] = U[b+p:m+1]
    i, k = b + p - 1, b + p + r
    for j in xrange(r, -1, -1):
        while X[j] <= U[i] and i > a:
            Qw[k-p-1] = Pw[i-p-1]
            Ubar[k] = U[i]
            k, i  = k - 1, i - 1
        Qw[k-p-1] = Qw[k-p]
        for l in xrange(1, p + 1):
            ind = k - p + l
            alfa = Ubar[k+l] - X[j]
            if abs(alfa) == 0.0:
                Qw[ind-1] = Qw[ind]
            else:
                alfa /= Ubar[k+l] - U[i-p+l]
                Qw[ind-1] = alfa * Qw[ind-1] + (1.0 - alfa) * Qw[ind]
        Ubar[k] = X[j]
        k -= 1
    return Ubar, Qw


def decompose_curve(n, p, U, Pw):

    ''' Decompose a NURBS curve and returns nb Bezier segments.  Qw[j,k]
    is the kth control point of the jth segment.

    Source: The NURBS Book (2nd Ed.), Pg. 173.

    '''

    ni = np.unique(U[p+1:n+1]).size
    Qw = np.zeros((ni + 1, p + 1, 4))
    Ub = np.zeros((ni + 1, 2 * p + 2))
    alphas = np.zeros(p - 1)
    m = n + p + 1; a = p; b = p + 1; nb = 0
    Qw[nb,:p+1] = Pw[:p+1]
    while b < m:
        i = b
        while b < m and U[b+1] == U[b]:
            b += 1
        mult = b - i + 1
        Ub[nb] = np.hstack(((p + 1) * [U[a]], (p + 1) * [U[b]]))
        if mult < p:
            numer = U[b] - U[a]
            for j in xrange(p, mult, -1):
                alphas[j-mult-1] = numer / (U[a+j] - U[a])
            r = p - mult
            for j in xrange(1, r + 1):
                save = r - j
                s = mult + j
                for k in xrange(p, s - 1, -1):
                    alpha = alphas[k-s]
                    Qw[nb,k] = alpha * Qw[nb,k] + (1.0 - alpha) * Qw[nb,k-1]
                if b < m:
                    Qw[nb+1,save] = Qw[nb,p]
        nb += 1
        if b < m:
            for i in xrange(p - mult, p + 1):
                Qw[nb,i] = Pw[b-p+i]
            a = b; b += 1
    return nb, Ub, Qw


# Knot removal.
#
#   is the reverse of knot insertion.  Let Cw(u) = sum_(i=0)^(n) (Nip(u)
#   * Pwi) be defined on U, and let ur be an interior knot of
#   multiplicity s in U; end knots are not removed.  Let Ut denote the
#   knot vector obtained by removing ur t times from U (1 <= t <= s).
#   ur is said to be t times removable if Cw(u) has a precise
#   representation of the form Cw(u) = sum_(i=0)^(n-t) (Nbip(u) * Qwi)
#   where Nbip(u) are the basis functions on the Ut.  Note that both
#   representation are geometrically and parametrically identical.
#
#   Knot removal is an important utility in several applications:
#
#   - When interactively shaping B-spline curves and surfaces, knots are
#     sometimes added to increase the number of control points which can
#     be modified.  When control points are moved, the level of
#     continuity at the knots can change (increase or decrease); hence,
#     after modification is completed knot removal can be invoked in
#     order to obtain the most compact representation of the curve or
#     surface.
#
#   - It is sometimes useful to link B-spline curves together to form
#     composite curves.  The first step is to make the curves
#     compatible, i.e., of common degree, and the end parameter value of
#     the ith curve is equal to the start parameter of the (i + 1)th
#     curve.  Once this is done, the composition is accomplished by
#     using interior knots of multiplicity equal to the common degree of
#     the curves.  Knot removal can then be invoked in order to remove
#     unnecessary knots.


def remove_curve_knot(n, p, U, Pw, u, r, s, num, d=1e-3):

    ''' Try to remove the knot (u = ur != u_(r+1)) num times, where (1
    <= num <= s).  It returns nr, the actual number of times the knot is
    removed.

    Source: The NURBS Book (2nd Ed.), Pg. 185.

    '''

    TOL = calc_tol_removal(Pw, d)
    tmp = np.zeros((2 * p + 1, 4))
    U = U.copy(); Pw = Pw.copy()
    m = n + p + 1; o = p + 1; fout = (2 * r - s - p) // 2
    first = r - p; last = r - s; nr = 0
    for t in xrange(num):
        off = first - 1
        tmp[0] = Pw[off]; tmp[last+1-off] = Pw[last+1]
        i = first; j = last; ii = 1; jj = last - off
        remflag = 0
        while j - i > t:
            alfi = (u - U[i]) / (U[i+o+t] - U[i])
            alfj = (u - U[j-t]) / (U[j+o] - U[j-t])
            tmp[ii] = (Pw[i] - (1.0 - alfi) * tmp[ii-1]) / alfi
            tmp[jj] = (Pw[j] - alfj * tmp[jj+1]) / (1.0 - alfj)
            i += 1; ii += 1; j -= 1; jj -= 1
        if j - i < t:
            if util.distance(tmp[ii-1], tmp[jj+1]) <= TOL:
                remflag = 1; nr += 1
        else:
            alfi = (u - U[i]) / (U[i+o+t] - U[i])
            if (util.distance(Pw[i], alfi * tmp[ii+t+1] +
                              (1.0 - alfi) * tmp[ii-1])) <= TOL:
                remflag = 1; nr += 1
        if remflag == 0:
            break
        else:
            i = first; j = last
            while j - i > t:
                Pw[i], Pw[j] = tmp[i-off], tmp[j-off]
                i += 1; j -= 1
        first -= 1; last += 1
    if nr == 0:
        return nr, U, Pw
    U[r+1-nr:m+1-nr] = U[r+1:m+1]
    j = fout; i = j
    for k in xrange(1, nr):
        if (np.mod(k, 2) == 1):
            i += 1
        else:
            j -= 1
    for k in xrange(i + 1, n + 1):
        Pw[j] = Pw[k]
        j += 1
    U = U[:-nr]; Qw = Pw[:-nr]
    return nr, U, Qw


def remove_curve_knots(n, p, U, Pw, d=1e-3):

    ''' Remove as many knots as possible from a Curve.  This is
    essentially a loop of remove_curve_knot over each distinct knot.

    Source: Tiller, Knot-removal algorithms for NURBS curves and
            surfaces, CAD, 1992.

    '''

    if n < p + 1:
        return 0, U, Pw
    TOL = calc_tol_removal(Pw, d)
    tmp = np.zeros((2 * p + 1, 4))
    U = U.copy(); Pw = Pw.copy()
    o = p + 1; hispan = n; hiu = U[hispan]
    gap = 0; u = U[o]; r = o
    while u == U[r+1]:
        r += 1
    s = r - p; fout = (2 * r - s - p) // 2
    first = s; last = r - s
    bgap = r; agap = bgap + 1
    while True:
        nr = 0
        for t in xrange(s):
            off = first - 1
            tmp[0] = Pw[off]; tmp[last+1-off] = Pw[last+1]
            i = first; j = last
            ii = first - off; jj = last - off
            remflag = 0
            while j - i > t:
                alfi = (u - U[i]) / (U[i+o+gap+t] - U[i])
                alfj = (u - U[j-t]) / (U[j+o+gap] - U[j-t])
                tmp[ii] = (Pw[i] - (1.0 - alfi) * tmp[ii-1]) / alfi
                tmp[jj] = (Pw[j] - alfj * tmp[jj+1]) / (1.0 - alfj)
                i += 1; ii += 1; j -= 1; jj -= 1
            if j - i < t:
                if util.distance(tmp[ii-1], tmp[jj+1]) <= TOL:
                    remflag = 1; nr += 1
            else:
                alfi = (u - U[i]) / (U[i+o+gap+t] - U[i])
                if (util.distance(Pw[i], alfi * tmp[ii+t+1] +
                                  (1.0 - alfi) * tmp[ii-1])) <= TOL:
                    remflag = 1; nr += 1
            if remflag == 0:
                break
            else:
                i = first; j = last
                while j - i > t:
                    Pw[i], Pw[j] = tmp[i-off], tmp[j-off]
                    i += 1; j -= 1
            first -= 1; last += 1
        if nr > 0:
            j = fout; i = j
            for k in xrange(1, nr):
                if (np.mod(k, 2) == 1):
                    i += 1
                else:
                    j -= 1
            for k in xrange(i + 1, bgap + 1):
                Pw[j] = Pw[k]
                j += 1
        else:
            j = bgap + 1
        if u == hiu:
            gap += nr
            break
        else:
            k1 = i = r - nr + 1; k = r + gap + 1
            u = U[k]
            while u == U[k]:
                U[i] = U[k]
                i += 1; k += 1
            s = i - k1; r = i - 1
            gap += nr
            for k in xrange(s):
                Pw[j] = Pw[agap]
                j, agap = j + 1, agap + 1
            bgap = j - 1
            fout = (2 * r - p - s) // 2
            first = r - p; last = r - s
    if gap == 0:
        return gap, U, Pw
    i = hispan + 1; k = i - gap
    for j in xrange(1, o + 1):
        U[k] = U[i]
        k += 1; i += 1
    U = U[:-gap]; Qw = Pw[:-gap]
    return gap, U, Qw


# Degree elevation.
#
#   Let Cw_p(u) be a pth-degree NURBS curve on the knot vector U.  Since
#   Cw_p(u) is a piecewise polynomial curve, it should be possible to
#   elevate its degree to (p + 1), i.e. there must exist control points
#   Qw and a knot vector Uh such that
#
#       Cw_p(u) = Cw_(p+1)(u) = sum_(i=0)^(nh) (N_(i,p+1)(u) * Qwi)
#
#   Cw_p(u) and Cw_(p+1)(u) are the same curve, both geometrically and
#   parametrically.
#
#   The applications of degree elevation include:
#
#   - Using tensor product when constructing certain types of surfaces
#     from a set curves C1,...,Cn (n >= 2) requires that these curves
#     have a common degree, hence the degrees of some curves may require
#     elevation.
#
#   - Let C1,...,Cn (n >= 2) be a sequence of NURBS curves with the
#     property that the endpoint of C_i is coincident with the starting
#     point of C_(i+1); then the curves can be combined into a single
#     NURBS curve.  One step in this process is to elevate the curves to
#     a common degree.


def degree_elevate_curve(n, p, U, Pw, t):

    ''' Raise the degree from p to (p + t), (t >= 1), by computing nh,
    Uh and Qw.

    Source: The NURBS Book (2nd Ed.), Pg. 206.

    '''

    bezalfs = product_matrix(p, t)
    bpts = np.zeros((p + 1, 4))
    ebpts = np.zeros((p + t + 1, 4))
    Nextbpts = np.zeros((p - 1, 4))
    alfs = np.zeros(p - 1)
    nh, dummy = mult_degree_elevate(n, p, U, t)
    Qw = np.zeros((nh + 1, 4))
    Uh = np.zeros(nh + p + t + 2)
    m = n + p + 1
    ph = p + t
    kind = ph + 1
    r = -1
    a = p; b = p + 1
    cind = 1
    ua = U[0]
    Qw[0] = Pw[0]
    Uh[:ph+1] = ua
    bpts[:p+1] = Pw[:p+1]
    while b < m:
        i = b
        while b < m and U[b] == U[b+1]:
            b += 1
        mul = b - i + 1
        ub = U[b]
        oldr = r
        r = p - mul
        lbz = (oldr + 2) // 2 if oldr > 0 else 1
        rbz = ph - (r + 1) // 2 if r > 0 else ph
        if r > 0:
            numer = ub - ua
            for k in xrange(p, mul, -1):
                alfs[k-mul-1] = numer / (U[a+k] - ua)
            for j in xrange(1, r + 1):
                save = r - j
                s = mul + j
                for k in xrange(p, s - 1, -1):
                    bpts[k] = (alfs[k-s] * bpts[k] +
                               (1.0 - alfs[k-s]) * bpts[k-1])
                Nextbpts[save] = bpts[p]
        for i in xrange(lbz, ph + 1):
            ebpts[i] = 0.0
            mpi = min(p, i)
            for j in xrange(max(0, i - t), mpi + 1):
                ebpts[i] += bezalfs[i,j] * bpts[j]
        if oldr > 1:
            first = kind - 2; last = kind
            den = ub - ua
            bet = (ub - Uh[kind-1]) / den
            for tr in xrange(1, oldr):
                i = first; j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - Uh[i]) / (ua - Uh[i])
                        Qw[i] = (alf * Qw[i] + (1.0 - alf) * Qw[i-1])
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - Uh[j-tr]) / den
                            ebpts[kj] = (gam * ebpts[kj] +
                                         (1.0 - gam) * ebpts[kj+1])
                        else:
                            ebpts[kj] = (bet * ebpts[kj] +
                                         (1.0 - bet) * ebpts[kj+1])
                    i += 1; j -= 1; kj -= 1
                first -= 1; last += 1
        if a != p:
            for i in xrange(ph - oldr):
                Uh[kind] = ua
                kind += 1
        for j in xrange(lbz, rbz + 1):
            Qw[cind] = ebpts[j]
            cind += 1
        if b < m:
            bpts[:r] = Nextbpts[:r]
            bpts[r:p+1] = Pw[b-p+r:b+1]
            a = b; b += 1
            ua = ub
        else:
            Uh[kind:kind+ph+1] = ub
    return Uh, Qw


#   Degree reduction.
#
#   Let Cw(u) = sum_(i=0)^(n) (Nip(u) * Qwi) be a pth-degree NURBS curve
#   on the knot vector U.  Cw(u) is degree reducible if it has a precise
#   representation of the form Cw(u) = Chw(u) = sum_(i=0)^(nh)
#   (N_(i,p-1)(u) * Pwi) on the knot vector Uh.  Due to floating point
#   round-off error, one can never expect Ch(u) to coincide precisely
#   with C(u), and therefore C(u) should only be declared degree
#   reducible if max(E(u)) <= TOL, where E(u) = |C(u) - Ch(u)|.  Here
#   only precise degree reduction is considered, i.e. TOL is assumed
#   small.
#
#   The main application is to reverse the degree elevation process.
#   For example, degree reduction should be applied to each constituent
#   piece when decomposing a composite curve, because degree elevation
#   may have been required in the composition process.


def bez_degree_reduce(bpts, rbpts, MaxErr):

    ''' Supporting function which implements Bezier degree reduction and
    computation of the maximum error.  bpts are the Bezier control
    points of the current segment and rbpts the degree reduced control
    points.

    Source: The NURBS Book (2nd Ed.), Pg. 220-221.

    '''

    p = rbpts.shape[0]
    r = (p - 1) // 2
    odd = np.mod(p, 2)
    rbpts[0] = bpts[0]
    if not odd:
        for i in xrange(1, r + 1):
            alf = float(i) / p
            rbpts[i] = (bpts[i] - alf * rbpts[i-1]) / (1.0 - alf)
        rbpts[p-1] = bpts[p]
        for i in xrange(p - 2, r, -1):
            alf = float(i + 1) / p
            rbpts[i] = (bpts[i+1] - (1.0 - alf) * rbpts[i+1]) / alf
        MaxErr[:] = util.distance(bpts[r+1], (rbpts[r] + rbpts[r+1]) / 2)
    else:
        for i in xrange(1, r):
            alf = float(i) / p
            rbpts[i] = (bpts[i] - alf * rbpts[i-1]) / (1.0 - alf)
        rbpts[p-1] = bpts[p]
        for i in xrange(p - 2, r, -1):
            alf = float(i + 1) / p
            rbpts[i] = (bpts[i+1] - (1.0 - alf) * rbpts[i+1]) / alf
        alfl, alfr = float(r) / p, float(r + 1) / p
        prl = (bpts[r] - alfl * rbpts[r-1]) / (1.0 - alfl)
        prr = (bpts[r+1] - (1.0 - alfr) * rbpts[r+1]) / alfr
        rbpts[r] = (prl + prr) / 2
        MaxErr[:] = util.distance(prl, prr)


def degree_reduce_curve(n, p, U, Pw, d=1e-3):

    ''' Try to degree reduce a NURBS curve by 1, subject to a maximum
    deviation d.

    Source: The NURBS Book (2nd Ed.), Pg. 223.

    '''

    TOL = calc_tol_removal(Pw, d)
    bpts = np.zeros((p + 1, 4))
    Nextbpts = np.zeros((p - 1, 4))
    rbpts = np.zeros((p, 4))
    alphas = np.zeros(p - 1)
    MaxErr = np.zeros(1)
    e = np.zeros(n + p + 1)
    nh = mult_degree_reduce(n, p, U)
    Qw = np.zeros((nh + 1, 4))
    Uh = np.zeros(nh + p + 1)
    ph = p - 1; mh = ph
    kind = ph + 1; r = -1; a = p
    b = p + 1; cind = 1; mult = p
    m = n + p + 1
    Qw[0] = Pw[0]
    Uh[:ph+1] = U[0]
    bpts[:p+1] = Pw[:p+1]
    while b < m:
        i = b
        while b < m and U[b] == U[b+1]:
            b += 1
        mult = b - i + 1; mh += mult - 1
        oldr = r; r = p - mult
        lbz = (oldr + 2) // 2 if oldr > 0 else 1
        if r > 0:
            numer = U[b] - U[a]
            for k in xrange(p, mult - 1, -1):
                alphas[k-mult-1] = numer / (U[a+k] - U[a])
            for j in xrange(1, r + 1):
                save = r - j; s = mult + j
                for k in xrange(p, s - 1, -1):
                    bpts[k] = (alphas[k-s] * bpts[k] +
                               (1.0 - alphas[k-s]) * bpts[k-1])
                Nextbpts[save] = bpts[p]
        bez_degree_reduce(bpts, rbpts, MaxErr)
        e[a] += MaxErr[0]
        if e[a] > TOL:
            raise MaximumToleranceReached(e, TOL)
        if oldr > 0:
            first, last = kind, kind
            for k in xrange(oldr):
                i, j = first, last; kj = j - kind
                while j - i > k:
                    alfa = (U[a] - Uh[i-1]) / (U[b] - Uh[i-1])
                    beta = (U[a] - Uh[j-k-1]) / (U[b] - Uh[j-k-1])
                    Qw[i-1] = (Qw[i-1] - (1.0 - alfa) * Qw[i-2]) / alfa
                    rbpts[kj] = (rbpts[kj] - beta * rbpts[kj+1]) / (1.0 - beta)
                    i, j, kj = i + 1, j - 1, kj - 1
                if j - i < k:
                    Br = util.distance(Qw[i-2], rbpts[kj+1])
                else:
                    delta = (U[a] - Uh[i-1]) / (U[b] - Uh[i-1])
                    A = delta * rbpts[kj+1] + (1.0 - delta) * Qw[i-2]
                    Br = util.distance(Qw[i-1], A)
                K = a + oldr - k; q = (2 * p - k + 1) // 2
                L = k - q
                for ii in xrange(L, a + 1):
                    e[ii] += Br
                    if e[ii] > TOL:
                        raise MaximumToleranceReached(e, TOL)
                first, last = first - 1, last + 1
            cind = i - 1
        if a != p:
            for i in xrange(ph - oldr):
                Uh[kind] = U[a]
                kind += 1
        for i in xrange(lbz, ph + 1):
            Qw[cind] = rbpts[i]
            cind += 1
        if b < m:
            bpts[:r] = Nextbpts[:r]
            bpts[r:p+1] = Pw[b-p+r:b+1]
            a = b; b += 1
        else:
            Uh[kind:kind+ph+1] = U[b]
    return Uh, Qw


# ADVANCED GEOMETRIC ALGORITHMS


def curve_point_projection(n, p, U, Pw, Pi, ui=None, eps1=1e-15, eps2=1e-12):

    ''' Find the parameter value ui for which C(ui) is closest to Pi.
    This is achieved by minimizing, using Newton iteration, the function
    f(u) = |C'(u) * (C(u) - Pi)|.  Two zero tolerances are used to
    indicate convergence: (1) eps1, a measure of Euclidean distance and
    (2) eps2, a zero cosine measure.  If not provided, the initial
    iterate ui is obtained by brute force.

    Source: The NURBS Book (2nd Ed.), Pg. 229.

    '''

    if ui is None:
        num = 500 # knob
        us, = util.construct_flat_grid((U,), (num,))
        C = rat_curve_point_v(n, p, U, Pw, us, num)
        i = np.argmin(util.distance_v(C, Pi))
        ui = us[i]
    P, w = Pw[:,:-1], Pw[:,-1]
    rat = (w != 1.0).any()
    for ni in xrange(20): # knob
        CK = curve_derivs_alg1(n, p, U, P, ui, 2)
        if rat:
            wders = curve_derivs_alg1(n, p, U, w, ui, 2)
            CK = rat_curve_derivs(CK, wders, 2)
        C, CP, CPP = CK
        R = C - Pi; RN = util.norm(R)
        CPN = util.norm(CP)
        if RN <= eps1 or CPN == 0.0:
            return ui,
        CPR = np.dot(CP, R)
        zero_cosine = abs(CPR) / CPN / RN
        if zero_cosine <= eps2:
            return ui,
        uii = ui - CPR / (np.dot(CPP, R) + CPN**2)
        if uii < U[0]:
            uii = U[0]
        elif uii > U[-1]:
            uii = U[-1]
        if util.norm((uii - ui) * CP) <= eps1:
            return uii,
        ui = uii
    raise nurbs.NewtonLikelyDiverged(ui)


def reverse_curve_direction(n, p, U, Pw):

    ''' Reverse the direction of a curve while maintaining its
    parameterization.

    Source: The NURBS Book (2nd Ed.), Pg. 263.

    '''

    m = n + p + 1
    a = U[0]; b = U[-1]
    S = U.copy()
    for i in xrange(1, m - 2 * p):
        S[m-p-i] = - U[p+i] + a + b
    Qw = Pw[::-1]
    return S, Qw


def param_to_arc_length(C, u=None):

    ''' Return the arc length of C(u).

    Source: Peterson, Arc Length Parameterization of Spline Curves,
            Computed-Aided Design, 2006.

    '''

    def norm_der1(u):
        d1 = C.eval_derivatives(u, 1)[1]
        return util.norm(d1)

    U, = C.U; u0 = U[0]
    if u is None:
        u = U[-1]
    return quad(norm_der1, u0, u, full_output=1, limit=500)[0]


def arc_length_to_param(C, a):

    ''' Return the parameter u such that C(u) = arc length a.

    Source: Peterson, Arc Length Parameterization of Spline Curves,
            Computed-Aided Design, 2006.

    '''

    def La(u):
        return param_to_arc_length(C, u) - a

    def Lap(u):
        d1 = C.eval_derivatives(u, 1)[1]
        return util.norm(d1)

    U, = C.U; ui = U[0]
    for ni in xrange(20):
        Laui = La(ui)
        if abs(Laui) <= 1e-8:
            return ui
        uii = ui - Laui / Lap(ui)
        if uii < U[0]:
            uii = U[0]
        elif uii > U[-1]:
            uii = U[-1]
        ui = uii
    raise nurbs.NewtonLikelyDiverged(ui)


def reparam_func_curve(C, f):

    ''' Let C(u) be an arbitrary pth-degree Curve on u in [a, b], and
    let u = f(s) be a qth-degree reparameterization B-spline function (a
    Curve where only the first coordinate is considered) on s in [c, d].
    This function returns C(f(s)), a pqth-degree Curve on s in [c, d].
    Note: rational Curves are currently not supported.

    Source: The NURBS Book (2nd Ed.), Pg. 251.

    '''

    def C1(Cd, fd):
        return Cd[0] * fd[0][0]

    def C2(Cd, fd):
        return Cd[0] * fd[1][0] + \
               Cd[1] * fd[0][0]**2

    def C3(Cd, fd):
        return Cd[0] * fd[2][0] + \
               Cd[1] * fd[0][0] * fd[1][0] * 3 + \
               Cd[2] * fd[0][0]**3

    def C4(Cd, fd):
        return Cd[0] * fd[3][0] + \
               Cd[1] * (fd[0][0] * fd[2][0] * 4 + fd[1][0]**2 * 3) + \
               Cd[2] * fd[0][0]**2 * fd[1][0] * 6 + \
               Cd[3] * fd[0][0]**4


    if any([c.isrational for c in C, f]):
        raise nurbs.RationalNURBSObjectDetected()

    dummy, p, U, dummy = C.var()
    dummy, q, S, dummy = f.var()

    pq = p * q
    if q > 1 and pq > 8:
        raise ImproperInput()

    Si = np.unique(S[q+1:-q-1])
    Ui = np.array([f.eval_point(s)[0] for s in Si])
    Cr = C.refine(knot.missing_knot_vec(Ui, U))

    n, p, Ur, Pw = Cr.var()
    nb, Us, Pws = decompose_curve(n, p, Ur, Pw)
    Cs = []
    for b in xrange(nb):
        c = Curve(ControlPolygon(Pw=Pws[b]), (p,), (Us[b],))
        Cs.append(c)

    Up = np.unique(np.hstack(Us))[1:-1]
    Sp = [f.project((u, 0, 0))[0] for u in Up]
    Sp = np.repeat(Sp, pq)
    Sp = np.hstack(((pq + 1) * [S[0]], Sp, (pq + 1) * [S[-1]]))

    ml = (pq + 1) // 2
    mr = pq - pq // 2 - 1

    Spu = np.unique(Sp)
    if q == 1:
        Cr = make_composite_curve(Cs, remove=False)
        Cr.U = Sp,
    else:
        for b in xrange(nb):
            c, u, Pw = Cs[b], Us[b], Pws[b]
            Cdl = [c.eval_derivatives(u[0], i)[i]
                   for i in xrange(1, ml + 1)]
            Cdr = [c.eval_derivatives(u[-1], i)[i]
                   for i in xrange(1, mr + 1)]
            fdl = [f.eval_derivatives(Spu[b], i)[i]
                   for i in xrange(1, ml + 1)]
            fdr = [f.eval_derivatives(Spu[b+1], i)[i]
                   for i in xrange(1, mr + 1)]
            ds = Spu[b+1] - Spu[b]
            for i in xrange(ml + 1):
                if i == 0:
                    Ql = Pw[0,:-1]
                    continue
                elif i == 1:
                    Qn = ds / pq * C1(Cdl, fdl) \
                         + Ql
                elif i == 2:
                    Qn = ds**2 / pq / (pq - 1) * C2(Cdl, fdl) \
                         - Ql[0] + 2 * Ql[1]
                elif i == 3:
                    Qn = ds**3 / pq / (pq - 1) / (pq - 2) * C3(Cdl, fdl) \
                         + Ql[0] - 3 * Ql[1] + 3 * Ql[2]
                elif i == 4:
                    Qn = ds**4 / pq / (pq - 1) / (pq - 2) / (pq - 3) * C4(Cdl, fdl) \
                         - Ql[0] + 4 * Ql[1] - 6 * Ql[2] + 4 * Ql[3]
                Ql = np.vstack((Ql, Qn))
            for i in xrange(mr + 1):
                if i == 0:
                    Qr = Pw[-1,:-1]
                    continue
                elif i == 1:
                    Qn = - ds / pq * C1(Cdr, fdr) \
                         + Qr
                elif i == 2:
                    Qn = ds**2 / pq / (pq - 1) * C2(Cdr, fdr) \
                         - Qr[0] + 2 * Qr[1]
                elif i == 3:
                    Qn = - ds**3 / pq / (pq - 1) / (pq - 2) * C3(Cdr, fdr) \
                         + Ql[0] - 3 * Ql[1] + 3 * Ql[2]
                Qr = np.vstack((Qr, Qn))
            Q = np.vstack((Ql, Qr[-1::-1]))
            if b == 0:
                P = Q
            else:
                P = np.vstack((P, Q[1:]))

        Pw = nurbs.obj_mat_to_4D(P)
        Cr = Curve(ControlPolygon(Pw=Pw), (pq,), (Sp,))

    mult = knot.find_int_mult_knot_vec(pq, Sp)
    for si in Spu[1:-1]:
        ui = f.eval_point(si)[0]
        mui = U.tolist().count(ui)
        msi = S.tolist().count(si)
        if msi == 0:
            mi = pq - p + mui
        elif mui == 0:
            mi = pq - q + msi
        else:
            mi = max(pq - p + mui, pq - q + msi)
        Cr = Cr.remove(si, mult[si] - mi)[1]
    return Cr


def mult_func_curve(G, f):

    ''' Compute the product of a B-spline function f(u) and a NURBS
    Curve G(u).  Their product

                      [f(u)x(u)w(u), f(u)y(u)w(u), f(u)z(u)w(u)]
        f(u) * G(u) = ------------------------------------------
                                        w(u)

    is a (p + q)th degree NURBS Curve.  Note that, though f(u) is given
    in the form of a NURBS Curve, only the first coordinate is
    considered i.e. f(u) can be think of a univariate spline.

    Sources
    -------
    - Piegl & Tiller, Symbolic operators for NURBS, Computer-Aided
      Design, 1997.

    - Piegl & Tiller, Algorithm for Computing the Product of two
      B-splines, Curves and surfaces with applications in CAGD, 1997.

    '''

    # refine functions
    n, p, R, fw = f.var()
    m, q, S, gw = G.var()
    knot.normalize_knot_vec(R)
    knot.normalize_knot_vec(S)
    sk = np.unique(knot.missing_knot_vec(S[q+1:-q-1], R[p+1:-p-1]))
    rk = np.unique(knot.missing_knot_vec(R[p+1:-p-1], S[q+1:-q-1]))
    R, fw = refine_knot_vect_curve(n, p, R, fw, sk)
    S, gw = refine_knot_vect_curve(m, q, S, gw, rk)
    n += sk.size
    m += rk.size

    # initialize local arrays
    fb = np.zeros(p + 1); nextfb = np.zeros(p + 1)
    gb = np.zeros((q + 1, 4)); nextgb = np.zeros((q + 1, 4))
    alf = np.zeros(p + 1); bet = np.zeros(q + 1)
    T = np.zeros(p + q + 1); h = np.zeros((1, 4))
    f = fw[:,0]

    # initialize some variables
    ar = p; br = p + 1; as2 = q; bs = q + 1; rem = -1; nh = 1
    mt, mr, ms = p + q + 1, n + p + 1, m + q + 1
    h[0,:3], h[0,-1] = f[0] * gw[0,:3], gw[0,-1]

    # initialize first Bezier segments
    for i in xrange(p + 1):
        fb[i] = f[i]
    for j in xrange(q + 1):
        gb[j] = gw[j]

    # precompute Bezier product matrix
    M = product_matrix(p, q)

    # loop through the knot vectors
    while br < mr and bs < ms:

        # compute knot multiplicities
        i = br
        while br < mr and R[br] == R[br+1]:
            br += 1
        mlr = mir = br - i + 1
        j = bs
        while bs < ms and S[bs] == S[bs+1]:
            bs += 1
        mls = mis = bs - j + 1

        # adjust multiplicities
        if R[br] in sk: mir = 0
        if S[bs] in rk: mis = 0

        # compute multiplicity of output knot
        if mis == 0:
            mi = q + mir
        elif mir == 0:
            mi = p + mis
        else:
            mi = max(q + mir, p + mis)

        # insert knots R[br] and S[bs] to get Bezier functions
        rr, rs = p - mlr, q - mls
        t = p + q - mi
        lbz = (rem + 2) // 2 if rem > 0 else 1
        rbz = p + q - (t + 1) // 2 if t > 0 else p + q
        if rr > 0:
            num = R[br] - R[ar]
            for k in xrange(p, mlr, -1):
                alf[k-mlr-1] = num / (R[ar+k] - R[ar])
            for j in xrange(1, rr + 1):
                save = rr - j
                s = mlr + j
                for k in xrange(p, s - 1, -1):
                    fb[k] = alf[k-s] * fb[k] + (1.0 - alf[k-s]) * fb[k-1]
                nextfb[save] = fb[p]
        if rs > 0:
            num = S[bs] - S[as2]
            for k in xrange(q, mls, -1):
                bet[k-mls-1] = num / (S[as2+k] - S[as2])
            for j in xrange(1, rs + 1):
                save = rs - j
                s = mls + j
                for k in xrange(q, s - 1, -1):
                    gb[k] = bet[k-s] * gb[k] + (1.0 - bet[k-s]) * gb[k-1]
                nextgb[save] = gb[q]

        # compute product of Bezier segments
        hb = bezier_product(fb, p, gb, q, M, lbz, p + q)

        # remove knot R[ar] = S[as2]
        if rem > 1:
            first, last = mt - 2, mt
            den = S[bs] - S[as2]
            beta = (S[bs] - T[mt-1]) / den
            for k in xrange(1, rem):
                i = first; j = last; l = j - mt + 1
                while j - i > k:
                    if i < nh:
                        delta = (S[bs] - T[i]) / (S[as2] - T[i])
                        h[i] = delta * h[i] + (1 - delta) * h[i-1]
                    if j >= lbz:
                        if j - k <= mt - p - q + rem:
                            gamma = (S[bs] - T[j-k]) / den
                            hb[l] = gamma * hb[l] + (1 - gamma) * hb[l+1]
                        else:
                            hb[l] = beta * hb[l] + (1 - beta) * hb[l+1]
                    i += 1; j -= 1; l -= 1
                first -= 1; last += 1

        # load knots and coefficients
        if ar != p and as2 != q:
            mt += p + q - rem
            T = np.append(T, (p + q - rem) * [S[as2]])
        nh += rbz - lbz + 1
        h = np.append(h, hb[lbz:rbz+1], axis=0)

        # prepare for next pass through
        if br < mr and bs < ms:
            for i in xrange(rr):
                fb[i] = nextfb[i]
            for i in xrange(rr, p + 1):
                fb[i] = f[br-p+i]
            for j in xrange(rs):
                gb[j] = nextgb[j]
            for j in xrange(rs, q + 1):
                gb[j] = gw[bs-q+j]
            ar = br; br += 1; as2 = bs; bs += 1
            rem = p + q - mi
        else:
            T = np.append(T, (p + q + 1) * [S[bs]])
    return Curve(ControlPolygon(Pw=h), (p + q,), (T,))


# TOOLBOX


def make_linear_curve(P0, P1):

    ''' Construct a linear Curve based on two three-dimensional space
    Points.

    Parameters
    ----------
    P0, P1 = the two end Points

    Returns
    -------
    Curve = the linear Curve

    '''

    return Curve(ControlPolygon([P0, P1]), (1,))


def make_composite_curve(Cs, remove=True, d=0.0):

    ''' Link two or more Curves to form one composite Curve.

    Parameters
    ----------
    Cs = the connected Curves to unite
    remove = whether or not to remove as many end knots as possible
    d = if remove is True, the maximum deviation allowed during the
        removal process

    Returns
    -------
    Curve = the composite Curve

    '''

    Cs = reorient_curves(Cs)
    Cs = make_curves_compatible2(Cs)
    n, dummy, U, Pw = Cs[0].var()
    UQ, Qw = U[:n+1], Pw[:-1]
    for c in Cs[1:]:
        n, dummy, U, Pw = c.var()
        UQ = np.append(UQ, U[1:n+1])
        Qw = np.append(Qw, Pw[:-1], axis=0)
    n, p, U, Pw = Cs[-1].var()
    UQ = np.append(UQ, U[n+1:])
    Qw = np.append(Qw, Pw[-1:], axis=0)
    Cc = Curve(ControlPolygon(Pw=Qw), (p,), (UQ,))
    if remove:
        rus = [c.U[0][0] for c in Cs[1:]]
        mult = knot.find_int_mult_knot_vec(p, Cc.U[0])
        for ru in rus:
            Cc = Cc.remove(ru, mult[ru] - 1, d)[1]
    return Cc


def reparam_arc_length_curve(C):

    ''' Reparameterize the Curve C according to its arc length
    (IN-PLACE).

    Parameters
    ----------
    C = the Curve to reparameterize

    '''

    U, = C.U
    a = param_to_arc_length(C, U[-1])
    knot.remap_knot_vec(U, 0, a)


# UTILITIES


def make_curves_compatible1(Cs):

    ''' Ensure that the Curves are defined on the same parameter range,
    be of common degree and share the same knot vector.

    Source: The NURBS Book (2nd Ed.), Pg. 237-238.

    '''

    p = max([c.p[0] for c in Cs])
    Umin = min([c.U[0][ 0] for c in Cs])
    Umax = max([c.U[0][-1] for c in Cs])
    Cs1 = []
    for c in Cs:
        dp = p - c.p[0]
        c = c.elevate(dp)
        knot.remap_knot_vec(c.U[0], Umin, Umax)
        Cs1.append(c)
    U = knot.merge_knot_vecs(*[c.U[0] for c in Cs1])
    Cs2 = []
    for c in Cs1:
        c = c.refine(knot.missing_knot_vec(U, c.U[0]))
        Cs2.append(c)
    n, = Cs2[0].cobj.n
    return n, p, U, Cs2


def make_curves_compatible2(Cs):

    ''' Make Curves of common degree, and force the end parameter value
    of the ith Curve to be equal to the start parameter of the (i + 1)th
    Curve.  Also check if the end control points match.

    Source: The NURBS Book (2nd Ed.), Pg. 181.

    '''

    p = max([c.p[0] for c in Cs])
    Cs1 = []
    for c in Cs:
        dp = p - c.p[0]
        c = c.elevate(dp)
        Cs1.append(c)
    Cl = Cs1[0]
    U, = Cl.U; U -= U[0]
    knot.clean_knot_vec(U)
    for Cr in Cs1[1:]:
        lP = Cl.cobj.cpts[-1].xyz
        fP = Cr.cobj.cpts[ 0].xyz
        l2n = util.norm(lP - fP)
        if l2n > 1e-3:
            print('nurbs.curve.make_curves_compatible2 :: '
                  'control point mismatch ({})'.format(l2n))
        U, = Cr.U; U += Cl.U[0][-1] - U[0]
        knot.clean_knot_vec(U)
        Cl = Cr
    return Cs1


def reorient_curves(Cs):

    ''' Reorient all Curves in terms of increasing parameter value.

    '''

    def find_common_ends(cl, cr):
        for k, i in enumerate([0, -1]):
            el[k] = cl.cobj.cpts[i].xyz
            er[k] = cr.cobj.cpts[i].xyz
        if np.allclose(el[1], er[0]):
            return 1, 0
        for l in xrange(2):
            for r in xrange(2):
                l2n = util.norm(el[l] - er[r])
                if l2n < 1e-3:
                    return l, r
        raise NoCommonEndCouldBeFound(cl, cr)

    el, er = np.zeros((2, 3)), np.zeros((2, 3))
    Cs = [c.copy() for c in Cs]
    for k in xrange(len(Cs) - 1):
        l, r = find_common_ends(Cs[k], Cs[k+1])
        if k == 0 and l == 0:
            Cs[0] = Cs[0].reverse()
        if r == 1:
            Cs[k+1] = Cs[k+1].reverse()
    return Cs


def calc_tol_removal(Pw, d):

    ''' Calculate the tolerance a NURBS curve can deviate from.

    Source: The NURBS Book (2nd Ed.), Pg. 185.

    '''

    if (Pw[...,-1] == 1.0).all():
        return d
    wmin = np.min(Pw[...,-1])
    Pmax = 0.0
    s = Pw.shape[:-1]
    for i in np.ndindex(s):
        xyzw = Pw[i]
        dist = util.norm(xyzw[:3] / xyzw[-1])
        if dist > Pmax:
            Pmax = dist
    return d * wmin / (1.0 + Pmax)


def product_matrix(p, q):

    ''' Compute the coefficients for taking the product of two Bezier
    segments.

    Source: The NURBS Book (2nd Ed.), Pg. 206.

    '''

    bezalfs = np.zeros((p + q + 1, p + 1))
    ph = p + q; ph2 = ph // 2
    bezalfs[0,0] = bezalfs[ph,p] = 1.0
    for i in xrange(1, ph2 + 1):
        inv = 1.0 / comb(ph, i)
        mpi = min(p, i)
        for j in xrange(max(0, i - q), mpi + 1):
            bezalfs[i,j] = inv * comb(p, j) * comb(q, i - j)
    for i in xrange(ph2 + 1, ph):
        mpi = min(p, i)
        for j in xrange(max(0, i - q), mpi + 1):
            bezalfs[i,j] = bezalfs[ph-i,p-j]
    return bezalfs


def mult_degree_elevate(n, p, U, t):

    ''' For degree elevation, return the new number of control points nh
    and knot vector Uh.

    '''

    nh = n + (np.unique(U[p+1:-p-1]).size + 1) * t
    mult = knot.find_mult_knot_vec(U)
    Uh = np.zeros(0)
    for u, m in sorted(mult.items()):
        Uh = np.append(Uh, np.array((m + t) * [u]))
    return nh, Uh


def mult_degree_reduce(n, p, U):

    ''' For degree reduction, return the new number of control points
    nh.

    '''

    return n - np.unique(U[p+1:-p-1]).size - 1


def bezier_product(fb, p, gb, q, M, first, last):

    ''' Given two Bezier functions

        F^B(u) = sum_(i=0)^p B_(i,p) f_i^b

        G^B(u) = sum_(j=0)^q B_(j,q) g_j^b

    where f_i^b and g_i^b are the Bezier coefficients, and B_(i,p)(u)
    and B_(j,q)(u) are the Bernstein polynomials of degree p and q,
    respectively.  Their product is computed as follows:

        H^B(u) = sum_(k=0)^(p+q) B_(k,p+q) h_k^b

    This function computes the coefficients

        h_i^b    i = first,...,last  first >= 0  last <= p + q

    of the product function H^B(u).

    Source: Piegl & Tiller, Algorithm for Computing the Product of two
            B-splines, Curves and surfaces with applications in CAGD,
            1997.

    '''

    hkb = np.zeros((p + q + 1, 4))
    for k in xrange(first, last + 1):
        for l in xrange(max(0, k - q), min(p, k) + 1):
            hkb[k,:3] += M[k,l] * fb[l] * gb[k-l,:3]
            hkb[k,-1] += M[k,l] * gb[k-l,-1]
    return hkb


# EXCEPTIONS


class CurveException(nurbs.NURBSException):
    pass

class ImproperInput(CurveException):
    pass

class MaximumToleranceReached(CurveException):
    pass

class NoCommonEndCouldBeFound(CurveException):
    pass
