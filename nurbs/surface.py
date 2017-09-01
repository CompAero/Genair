''' A NURBS surface of degree p in the u direction and degree q in the v
direction is a bivariate vector-valued piecewise rational function of
the form

             sum_(i=0)^(n) sum_(j=0)^(m) (Nip(u) * Njq(v) * wij * Pij)
    S(u,v) = ---------------------------------------------------------
                sum_(i=0)^(n) sum_(j=0)^(m) (Nip(u) * Njq(v) * wij)

(a <= u <= b), (c <= v <= d).  The {Pij} form a bidirectional control
net, the {wij} are the weights, and the {Nip(u)} and {Njq(v)} are the
nonrational B-spline basis functions defined on the knot vectors

    U = {a,...,a, u_(p+1),...,u_(r-p-1), b,...,b}

    V = {c,...,c, v_(q+1),...,u_(s-q-1), d,...,d}

where (r = n + p + 1) and (s = m + q + 1).

As with curves it is convenient to represent a NURBS surface using
homogeneous coordinates, that is

    Sw(u,v) = sum_(i=0)^(n) sum_(j=0)^(m) (Nip(u) * Njq(v) * Pwij)

where Pwij = (wij*xij, wij*yij, wij*zij, wij).  Then S(u,v) =
H{Sw(u,v)}.  Strictly speaking, Sw(u,v) is a tensor product, piecewise
polynomial surface in four-dimensional space while S(u,v), again, is a
piecewise rational surface in three-dimensional space; it is not a
tensor product surface.

'''

import numpy as np
from scipy.linalg import lstsq
from scipy.misc import comb
from scipy.optimize import fmin

import basis
import conics
import curve
import fit
import knot
import nurbs
import point
import transform
import util

import plot.pobject


__all__ = ['Surface', 'ControlNet',
           'make_bilinear_surface',
           'make_composite_surface',
           'make_coons_surface',
           'make_general_cone',
           'make_general_cylinder',
           'make_gordon_surface',
           'make_nsided_region',
           'make_revolved_surface_nrat',
           'make_revolved_surface_rat',
           'make_ruled_surface',
           'make_skinned_surface',
           'make_swept_surface']


class ControlNet(nurbs.ControlObject):

    def __init__(self, cpts=None, Pw=None):

        ''' See nurbs.nurbs.ControlObject.

        Parameters
        ----------
        cpts = the list of list of (control) Points
        Pw = the object matrix

        Examples
        --------
        >>> P0 = Point(-5, 5, 1)
        >>> P1 = Point( 2, 6, 2)
        >>> P2 = Point(-3,-4, 0)
        >>> P3 = Point( 1, 2,-3)
        >>> cnet = ControlNet([[P0, P1],[P2, P3]])

        or, equivalently,

        >>> Pw = [[(-5, 5, 1, 1), (2, 6, 2, 1)],
        ...       [(-3,-4, 0, 1), (1, 2,-3, 1)]]
        >>> cnet = ControlNet(Pw=Pw)

        '''

        super(ControlNet, self).__init__(cpts, Pw)


class Surface(nurbs.NURBSObject, plot.pobject.PlotSurface):

    def __init__(self, cnet, p, U=None):

        ''' See nurbs.nurbs.NURBSObject.

        Parameters
        ----------
        cnet = the ControlNet
        p = the u, v degrees (order p + 1) of the Surface
        U = the u, v knot vectors

        Examples
        --------

          v|
           |
        P1 ._________. P3
           |         |
           |         |
           |         |
           |         | P2
        P0 ._________. ____ u

        >>> P0 = Point( 1,-1)
        >>> P1 = Point(-1,-1)
        >>> P2 = Point( 1, 1)
        >>> P3 = Point(-1, 1,)
        >>> cnet = ControlNet([[P0, P1],[P2, P3]])
        >>> s = Surface(cnet, (1,1))

        '''

        n, m = cnet.n
        p, q = p
        if n < p or m < q:
            raise nurbs.TooFewControlPoints((n, p), (m, q))
        self._cobj = cnet.copy()
        self._p = p, q
        if not U:
            U = (knot.uni_knot_vec(n, p),
                 knot.uni_knot_vec(m, q))
        self.U = U
        self._trimcurves = []
        super(Surface, self).__init__()

    def __getstate__(self):
        ''' Pickling. '''
        d = super(Surface, self).__getstate__()
        d['_trimcurves'] = self._trimcurves
        return d

    def __setstate__(self, d):
        ''' Unpickling. '''
        self.__dict__.update(d)
        super(Surface, self).__init__()

    def copy(self):
        ''' Self copy. '''
        S = super(Surface, self).copy()
        S._trimcurves = [C.copy() for C in self._trimcurves]
        return S

# EVALUATION OF POINTS AND DERIVATIVES

    def eval_point(self, u, v):

        ''' Evaluate a point.

        Parameters
        ----------
        u, v = the parameter values of the point

        Returns
        -------
        S = the xyz coordinates of the point

        '''

        n, p, U, m, q, V, Pw = self.var()
        return rat_surface_point(n, p, U, m, q, V, Pw, u, v)

    def eval_points(self, us, vs):

        ''' Evaluate multiple points.

        Parameters
        ----------
        us, vs = the parameter values of each point

        Returns
        -------
        S = the xyz coordinates of all points

        '''

        n, p, U, m, q, V, Pw = self.var()
        return rat_surface_point_v(n, p, U, m, q, V, Pw, us, vs, len(us))

    def eval_derivatives(self, u, v, d):

        ''' Evaluate derivatives at a point.

        Parameters
        ----------
        u, v = the parameter values of the point
        d = the number of derivatives to evaluate

        Returns
        -------
        SKL = all derivatives, where SKL[k,l,:] is the derivative of
              S(u,v) with respect to u k times and v l times (0 <= k +
              l <= d)

        '''

        n, p, U, m, q, V, Pw = self.var()
        Aders = surface_derivs_alg1(n, p, U, m, q, V, Pw[:,:,:-1], u, v, d)
        if self.isrational:
            wders = surface_derivs_alg1(n, p, U, m, q, V, Pw[:,:,-1], u, v, d)
            return rat_surface_derivs(Aders, wders, d)
        return Aders

    def eval_curvature(self, u, v):

        ''' Evaluate the mean curvature at a point.

        Parameters
        ----------
        u, v = the parameter values of the point

        Returns
        -------
        H = the mean curvature

        '''

        SKL = self.eval_derivatives(u, v, 2)
        SU, SUU = SKL[1,0], SKL[2,0]
        SV, SVV = SKL[0,1], SKL[0,2]
        SUV = SKL[1,1]
        N = np.cross(SU, SV); N /= util.norm(N)
        E, F, G = np.dot(SU, SU), np.dot(SU, SV), np.dot(SV, SV)
        e, f, g = np.dot(SUU, N), np.dot(SUV, N), np.dot(SVV, N)
        num = e * G - 2 * f * F + g * E
        den = 2 * (E * G - F**2)
        return num / den

# KNOT INSERTION

    def insert(self, u, e, di):

        ''' Insert a knot in one direction multiple times.

        Parameters
        ----------
        u = the knot value to insert
        e = the number of times to insert u
        di = the parametric direction in which to insert u (0 or 1)

        Returns
        -------
        Surface = the new Surface with u inserted e times in di

        '''

        n, p, U, m, q, V, Pw = self.var()
        if e > 0:
            u = knot.clean_knot(u)
            if di == 0:
                k, s = basis.find_span_mult(n, p, U, u)
            elif di == 1:
                k, s = basis.find_span_mult(m, q, V, u)
            U, V, Pw = surface_knot_ins(n, p, U, m, q, V, Pw, u, k, s, e, di)
        return Surface(ControlNet(Pw=Pw), (p,q), (U,V))

    def split(self, u, di):

        ''' Split the Surface in one direction.

        Parameters
        ----------
        u = the parameter value at which to split the Surface
        di = the parametric direction in which to split the Surface (0
             or 1)

        Returns
        -------
        [Surface, Surface] = the two split Surfaces

        '''

        n, p, U, m, q, V, Pw = self.var()
        u = knot.clean_knot(u)
        if di == 0:
            if u == U[0] or u == U[-1]:
                return [self.copy()]
            k, s = basis.find_span_mult(n, p, U, u)
            r = p - s
            if r > 0:
                U, V, Pw = surface_knot_ins(n, p, U, m, q, V,
                                            Pw, u, k, s, r, 0)
            Ulr = (np.append(U[:k+r+1], u), np.insert(U[k-s+1:], 0, u))
            Vlr = V, V
            Pwlr = Pw[:k-s+1,:], Pw[k-s:,:]
        elif di == 1:
            if u == V[0] or u == V[-1]:
                return [self.copy()]
            k, s = basis.find_span_mult(m, q, V, u)
            r = q - s
            if r > 0:
                U, V, Pw = surface_knot_ins(n, p, U, m, q, V,
                                            Pw, u, k, s, r, 1)
            Ulr = U, U
            Vlr = (np.append(V[:k+r+1], u), np.insert(V[k-s+1:], 0, u))
            Pwlr = Pw[:,:k-s+1], Pw[:,k-s:]
        return [Surface(ControlNet(Pw=Pw), (p,q), (U,V))
                for Pw, U, V in zip(Pwlr, Ulr, Vlr)]

    def extract(self, u, di):

        ''' Extract an isoparametric Curve from the Surface.

        Parameters
        ----------
        u = the parameter value at which to extract the Curve
        di = the parametric direction in which to extract the Curve (0
             or 1)

        Returns
        -------
        Curve = the extracted Curve

        '''

        n, p, U, m, q, V, Pw = self.var()
        u = knot.clean_knot(u)
        if di == 0:
            if u == U[0]:
                Pwc = Pw[0,:]
            elif u == U[-1]:
                Pwc = Pw[-1,:]
            else:
                k, s = basis.find_span_mult(n, p, U, u)
                r = p - s
                if r > 0:
                    U, V, Pw = surface_knot_ins(n, p, U, m, q, V,
                                                Pw, u, k, s, r, 0)
                Pwc = Pw[k-s,:]
            p = q
            U = V
        elif di == 1:
            if u == V[0]:
                Pwc = Pw[:,0]
            elif u == V[-1]:
                Pwc = Pw[:,-1]
            else:
                k, s = basis.find_span_mult(m, q, V, u)
                r = q - s
                if r > 0:
                    U, V, Pw = surface_knot_ins(n, p, U, m, q, V,
                                                Pw, u, k, s, r, 1)
                Pwc = Pw[:,k-s]
        return curve.Curve(curve.ControlPolygon(Pw=Pwc), (p,), (U,))

    def extend(self, l, di, end=False):

        ''' Extend the Surface.

        Tangent-plane and curvature continuities are only guaranteed if
        the Surface is nonrational (all weights equal 1.0).

        Parameters
        ----------
        l = the (estimated) length of the extension
        di = the parametric direction in which to extend the Surface (0
             or 1)
        end = whether to extend the start or the end part of the Surface

        Returns
        -------
        Surface = the Surface extension

        Source
        ------
        Shetty and White, Curvature-continuous extensions for rational
        B-spline curves and surfaces, Computer-aided design, 1991.

        '''

        if di == 1:
            Sext = self.swap().extend(l, 0, end)
            return Sext.swap() if Sext else None
        if end:
            Sext = self.reverse(0).extend(l, 0)
            return Sext.reverse(0) if Sext else None
        n, p, U, m, q, V, Pw = self.var()
        C = self.extract(V[0], 1)
        u = curve.arc_length_to_param(C, l)
        Ss = self.split(u, 0)
        if len(Ss) == 1:
            return None
        Sext, dummy = Ss
        Pw, cpts = Sext.cobj.Pw, Sext.cobj.cpts
        for col in xrange(m + 1):
            Q, N = [cpt.xyz for cpt in cpts[:2,col]]
            transform.mirror(Pw[:,col], N - Q, Q)
        U = Sext.U[0]; U -= U[0]
        return Sext.reverse(0)

# KNOT REFINEMENT

    def refine(self, X, di):

        ''' Refine the knot vector in one direction.

        Parameters
        ----------
        X = a list of the knots to insert
        di = the parametric direction in which to refine the Surface (0
             or 1)

        Returns
        -------
        Surface = the refined Surface

        '''

        n, p, U, m, q, V, Pw = self.var()
        if len(X) != 0:
            if X == 'mid':
                if di == 0:
                    X = knot.midpoints_knot_vec(U)
                elif di == 1:
                    X = knot.midpoints_knot_vec(V)
            X = knot.clean_knot(X)
            if di == 0:
                CU = U
            elif di == 1:
                CU = V
            U, V, Pw = refine_knot_vect_surface(n, p, U, m, q, V, Pw, X, di)
        return Surface(ControlNet(Pw=Pw), (p,q), (U,V))

    def decompose(self):

        ''' Decompose the Surface into Bezier patches.

        Returns
        -------
        Ss = a list of Bezier patches

        '''

        n, p, U, m, q, V, Pw = self.var()
        nbs, Ubs, Pws = decompose_surface(n, p, U, m, q, V, Pw, 0)
        Ss = []
        for bs in xrange(nbs):
            S = Surface(ControlNet(Pw=Pws[bs]), (p,q), (Ubs[bs],V))
            n, p, U, m, q, V, Pw = S.var()
            nb, Ub, Pw = decompose_surface(n, p, U, m, q, V, Pw, 1)
            for b in xrange(nb):
                S = Surface(ControlNet(Pw=Pw[b]), (p,q), (Ubs[bs], Ub[b]))
                Ss.append(S)
        return Ss

    def segment(self, u1, u2, v1, v2):

        ''' Segment the Surface in two directions.

        Let U and V be the new segmented knot vectors, then moving any
        of the control Points ij satisfying (u1 <= U[i]), (v1 <= V[j]),
        (U[i+p+1] <= u2) and (V[j+q+1] <= v2) would only modify the part
        of the new Surface whose parametric region lies between u1, u2,
        v1 and v2.

        Parameters
        ----------
        u1, u2, v1, v2 = the parametric bounds

        Returns
        -------
        Surface = the segmented Surface

        '''

        n, p, U, m, q, V, dummy = self.var()
        u1, u2 = knot.clean_knot((u1, u2))
        v1, v2 = knot.clean_knot((v1, v2))
        ru = knot.segment_knot_vec(n, p, U, u1, u2)
        rv = knot.segment_knot_vec(m, q, V, v1, v2)
        S = self.refine(ru, 0)
        return S.refine(rv, 1)

# KNOT REMOVAL

    def remove(self, u, num, di, d=1e-3):

        ''' Try to remove an interior knot in one direction multiple
        times.

        Parameters
        ----------
        u = the knot to remove
        num = the number of times to remove u (1 <= num <= s), where s
              is the multiplicity of u
        di = the parametric direction in which to remove u (0 or 1)
        d = the maximum deviation allowed

        Returns
        -------
        e = the number of times u has been successfully removed
        Surface = the new Surface with u removed at most num times in di

        '''

        n, p, U, m, q, V, Pw = self.var()
        e = 0
        if num > 0:
            u = knot.clean_knot(u)
            if di == 0:
                r, s = basis.find_span_mult(n, p, U, u)
            elif di == 1:
                r, s = basis.find_span_mult(m, q, V, u)
            e, U, V, Pw = remove_surface_knot(n, p, U, m, q, V,
                                              Pw, u, r, s, num, d, di)
        return e, Surface(ControlNet(Pw=Pw), (p,q), (U,V))

    def removes(self, di, d=1e-3):

        ''' Remove all removable interior knots in one or both
        directions at once.

        Parameters
        ----------
        di = the parametric direction in which to remove all removable
             knots (0, 1 or 2)
        d = the maximum deviation allowed

        Returns
        -------
        e = the number of knots that has been successfully removed
        Surface = the new Surface with all removable knots removed

        '''

        n, p, U, m, q, V, Pw = self.var()
        if di == 0:
            e, U, Pw = remove_surface_uknots(n, p, U, m, Pw, d)
        elif di == 1:
            Pws = np.transpose(Pw, (1, 0, 2))
            e, V, Pws = remove_surface_uknots(m, q, V, n, Pws, d)
            Pw = np.transpose(Pws, (1, 0, 2))
        elif di == 2:
            nr0, s0 = self.removes(0, d)
            nr1, s1 = s0.removes(1, d)
            return (nr0, nr1), s1
        return e, Surface(ControlNet(Pw=Pw), (p,q), (U,V))

# DEGREE ELEVATION

    def elevate(self, t, di):

        ''' Elevate the Surface's degree in one direction.

        Parameters
        ----------
        t = the number of degrees to elevate the Surface with
        di = the parametric direction in which to degree elevate the
             Surface (0 or 1)

        Returns
        -------
        Surface = the degree elevated Surface

        '''

        n, p, U, m, q, V, Pw = self.var()
        if t > 0:
            U, V, Pw = degree_elevate_surface(n, p, U, m, q, V, Pw, t, di)
            if di == 0:
                p += t
            elif di == 1:
                q += t
        return Surface(ControlNet(Pw=Pw), (p,q), (U,V))

# DEGREE REDUCTION

    def reduce(self, di, d=1e-3):

        ''' Try to reduce the Surface's degree in one direction by one.

        Parameters
        ----------
        di = the parametric direction in which to degree reduce the
             Surface (0 or 1)
        d = the maximum deviation allowed

        Returns
        -------
        success = whether or not degree reduction was successful
        Surface = the degree reduced Surface

        '''

        n, p, U, m, q, V, Pw = self.var()
        try:
            U, V, Pw = degree_reduce_surface(n, p, U, m, q, V, Pw, d, di)
            success = True
            if di == 0:
                p -= 1
            elif di == 1:
                q -= 1
        except curve.MaximumToleranceReached:
            success = False
        return success, Surface(ControlNet(Pw=Pw), (p,q), (U,V))

# TRIMMING

    def trim(self, outer, inners=None):

        ''' Trim the Surface with boundary Curves.

        Exerpt from the IGES 5.3 specification standard (Type 144):

        <<< A simple closed Curve in the Euclidean plane divides the
        plane into two disjoint open connected components, one bounded
        and one unbounded.  The bounded component is called the interior
        region to the Curve and the unbounded component is called the
        exterior region to the Curve.

        The domain of the trimmed Surface is defined as the common
        region of the interior of the outer boundary and the exterior of
        each of the inner boundaries and includes the boundary Curves.
        Note that the trimmed Surface has the same mapping S(u,v) as the
        original but a different domain.

        Let S(u,v) be a regular parameterized Surface, whose untrimmed
        domain is a rectangle D.  Two types of simpled closed Curves are
        utilized to define the domain of the trimmed Surface:

          1) Outer boundary: there is exactly one.  It lies in D, and,
             in particular, it can be the boundary Curve of D.

          2) Inner boundary: there can be any number of them, including
             zero.  The Curves, as well as their interiors, must be
             mutually disjoint.  Also, each Curve must lie in the
             interior of the outer boundary.

        If the outer boundary of the Surface being defined is the
        boundary of D and there are no inner boundaries, the trimmed
        Surface being defined is untrimmed. >>>

        Further, note that for visualization purposes OpenGL requires
        the orientation of the trimming Curves to be considered.
        Basically, if you imagine walking along a Curve, everything to
        the left is included and everything to the right is trimmed
        away.

        Parameters
        ----------
        outer = the user defined outer boundary Curve, or the integer 0
                in which case the outer boundary of D is used
        inners = a list of inner boundary Curves, if any

        Examples
        --------
        >>> P0 = Point( 1, -1)
        >>> P1 = Point(-1, -1)
        >>> P2 = Point( 1,  1)
        >>> P3 = Point(-1,  1)
        >>> S = nurbs.tb.make_bilinear_surface(P0, P1, P2, P3)

        >>> O, X, Y = ([0.5, 0.5, 0], [0, 1, 0], [1, 0, 0])
        >>> C = nurbs.tb.make_circle_rat(O, X, Y, 0.3, 0, 360)

        >>> S.trim(0, [C])

        '''

        inners = list(inners) if inners is not None else []
        if outer == 0:
            U, V = self.U
            P0 = point.Point(U[ 0], V[ 0])
            P1 = point.Point(U[-1], V[ 0])
            P2 = point.Point(U[-1], V[-1])
            P3 = point.Point(U[ 0], V[-1])
            P4 = point.Point(U[ 0], V[ 0])
            cpol = curve.ControlPolygon([P0, P1, P2, P3, P4])
            outer = curve.Curve(cpol, (1,))
        self._trimcurves = [outer] + inners

    @property
    def istrimmed(self):

        ''' Is the Surface trimmed?

        '''

        return True if self._trimcurves else False

    def untrim(self):

        ''' Untrim the Surface.

        '''

        del self._trimcurves[:]

# MISCELLANEA

    def project(self, xyz, uvi=None):

        ''' Project a point.

        Parameters
        ----------
        xyz = the xyz coordinates of a point to project
        uvi = the initial guess for Newton's method

        Returns
        -------
        u, v = the parameter values of the projected point

        '''

        n, p, U, m, q, V, Pw = self.var()
        return surface_point_projection(n, p, U, m, q, V, Pw, xyz, uvi)

    def reverse(self, di):

        ''' Reverse (flip) the Surface's direction.

        Parameters
        ----------
        di = the reversal direction (0 or 1)

        Returns
        -------
        Surface = the reversed Surface

        '''

        n, p, U, m, q, V, Pw = self.var()
        U, V, Pw = reverse_surface_direction(n, p, U, m, q, V, Pw, di)
        return Surface(ControlNet(Pw=Pw), (p,q), (U,V))

    def swap(self):

        ''' Swap the u and v directions.

        Returns
        -------
        Surface = the swapped Surface

        '''

        n, p, U, m, q, V, Pw = self.var()
        Pw = np.transpose(Pw, (1, 0, 2))
        return Surface(ControlNet(Pw=Pw), (q,p), (V,U))


# HEAVY LIFTING FUNCTIONS


def surface_deriv_cpts(n, p, U, m, q, V, P, d, r1, r2, s1, s2):

    ''' Compute all (or optionally some) of the control points surface
    derivatives P_(i,j)^(k,l) up to order d (0 <= k + l <= d).  Output
    is the array PKL, where PKL[k,l,i,j,:] is the (i,j)th control point
    on the surface, differentiated k times with respect to u and l times
    with respect to v.

    Source: The NURBS Book (2nd Ed.), Pg. 114.

    '''

    r, s = r2 - r1, s2 - s1
    PKL = np.zeros((d + 1, d + 1, r + 1, s + 1, 3))
    du, dv = min(d, p), min(d, q)
    for j in xrange(s1, s2 + 1):
        tmp = curve.curve_deriv_cpts(n, p, U, P[:,j], du, r1, r2)
        for k in xrange(du + 1):
            for i in xrange(r - k + 1):
                PKL[k,0,i,j-s1] = tmp[k,i]
    for k in xrange(du):
        for i in xrange(r - k + 1):
            dd = min(d - k, dv)
            tmp = curve.curve_deriv_cpts(m, q, V[s1:], PKL[k,0,i,:],
                                          dd, 0, s)
            for l in xrange(1, dd + 1):
                for j in xrange(s - l + 1):
                    PKL[k,l,i,j] = tmp[l,j]
    return PKL


def surface_derivs_alg1(n, p, U, m, q, V, P, u, v, d):

    ''' Compute a point on a B-spline surface and all partial
    derivatives up to and including order d (0 <= k + l <= d), (d > p,
    q) is allowed, although the derivatives are 0 in this case (for
    nonrational surfaces); these derivatives are necessary for rational
    surfaces.  Output is the array SKL, where SKL[k,l,:] is the
    derivative of S(u,v) with respect to u k times, and v l times.

    Source: The NURBS Book (2nd Ed.), Pg. 111.

    '''

    SKL = np.zeros((d + 1, d + 1, 3))
    tmp = np.zeros((q + 1, 3))
    du, dv = min(d, p), min(d, q)
    uspan = basis.find_span(n, p, U, u)
    Nu = basis.ders_basis_funs(uspan, u, p, du, U)
    vspan = basis.find_span(m, q, V, v)
    Nv = basis.ders_basis_funs(vspan, v, q, dv, V)
    for k in xrange(du + 1):
        tmp[:] = 0.0
        for s in xrange(q + 1):
            for r in xrange(p + 1):
                tmp[s] += Nu[k,r] * P[uspan-p+r,vspan-q+s]
        dd = min(d - k, dv)
        for l in xrange(dd + 1):
            for s in xrange(q + 1):
                SKL[k,l] += Nv[l,s] * tmp[s]
    return SKL


def surface_derivs_alg2(n, p, U, m, q, V, P, u, v, d):

    ''' Idem surface_derivs_alg1, but by using the controls of surface
    derivatives yielded by surface_deriv_cpts.

    Source: The NURBS Book (2nd Ed.), Pg. 115.

    '''

    SKL = np.zeros((d + 1, d + 1, 3))
    du, dv = min(d, p), min(d, q)
    uspan = basis.find_span(n, p, U, u)
    Nu = basis.all_basis_funs(uspan, u, p, U)
    vspan = basis.find_span(m, q, V, v)
    Nv = basis.all_basis_funs(vspan, v, q, V)
    PKL = surface_deriv_cpts(n, p, U, m, q, V, P, d,
                             uspan - p, uspan, vspan - q, vspan)
    for k in xrange(du + 1):
        dd = min(d - k, dv)
        for l in xrange(dd + 1):
            SKL[k,l] = 0.0
            for i in xrange(q - l + 1):
                tmp = 0.0
                for j in xrange(p - k + 1):
                    tmp += Nu[j,p-k] * PKL[k,l,j,i]
                SKL[k,l] += Nv[i,q-l] * tmp
    return SKL


def rat_surface_point(n, p, U, m, q, V, Pw, u, v):

    ''' Compute a point on a rational B-spline surface at fixed u and v
    parameter values.

    Source: The NURBS Book (2nd Ed.), Pg. 134.

    '''

    Sw = np.zeros(4)
    tmp = np.zeros((q + 1, 4))
    uspan = basis.find_span(n, p, U, u)
    Nu = basis.basis_funs(uspan, u, p, U)
    vspan = basis.find_span(m, q, V, v)
    Nv = basis.basis_funs(vspan, v, q, V)
    for l in xrange(q + 1):
        for k in xrange(p + 1):
            tmp[l] += Nu[k] * Pw[uspan-p+k,vspan-q+l]
    for l in xrange(q + 1):
        Sw += Nv[l] * tmp[l]
    return Sw[:3] / Sw[-1]


def rat_surface_point_v(n, p, U, m, q, V, Pw, u, v, num):

    ''' Idem rat_surface_point, vectorized in u, v.

    '''

    u, v = [np.asfarray(u) for u in u, v]
    Sw = np.zeros((4, num))
    tmp = np.zeros((q + 1, 4, num))
    uspan = basis.find_span_v(n, p, U, u, num)
    Nu = basis.basis_funs_v(uspan, u, p, U, num)
    vspan = basis.find_span_v(m, q, V, v, num)
    Nv = basis.basis_funs_v(vspan, v, q, V, num)
    for l in xrange(q + 1):
        for k in xrange(p + 1):
            tmp[l] += Nu[k] * Pw[uspan-p+k,vspan-q+l].T
    for l in xrange(q + 1):
        Sw += Nv[l] * tmp[l]
    return Sw[:3] / Sw[-1]


def rat_surface_derivs(Aders, wders, d):

    ''' Given that (u,v) is fixed, and that all derivatives A^(k,l),
    w^(k,l) for (k,l >= 0) and (0 <= k + l <= d) have been computed and
    loaded into the arrays Aders and wders, respectively, this algorithm
    computes the point, S(u,v) and the derivatives, S^(k,l)(u,v), (0 <=
    k + l <= d).  The surface point is returned in SKL[0,0,:] and the
    k,lth derivative is returned in SKL[k,l,:].

    Source: The NURBS Book (2nd Ed.), Pg. 137.

    '''

    SKL = np.zeros((d + 1, d + 1, 3))
    for k in xrange(d + 1):
        for l in xrange(d - k + 1):
            v = Aders[k,l]
            for j in xrange(1, l + 1):
                v -= comb(l, j) * wders[0,j] * SKL[k,l-j]
            for i in xrange(1, k + 1):
                for j in xrange(l + 1):
                    v -= (comb(k, i) * comb(l, j) *
                          wders[i,j] * SKL[k-i,l-j])
            SKL[k,l] = v / wders[0,0]
    return SKL


# FUNDAMENTAL GEOMETRIC ALGORITHMS


def surface_knot_ins(n, p, U, m, q, V, Pw, u, k, s, r, di):

    ''' Knots are inserted into surfaces by simply applying curve knot
    insertion to the rows and/or columns of control points.  In
    particular, ubar is added to the knot vector U by doing a ubar knot
    insertion on each of m + 1 columns of control points.  Analogously,
    vbar must be inserted on each of the n + 1 rows of control points.

    Source: The NURBS Book (2nd Ed.), Pg. 155.

    '''

    if di == 0:
        VQ = V.copy()
        Qw = np.zeros((n + r + 1, m + 1, 4))
        for col in xrange(m + 1):
            UQ, Qw[:,col] = \
                    curve.curve_knot_ins(n, p, U, Pw[:,col], u, k, s, r)
    elif di == 1:
        UQ = U.copy()
        Qw = np.zeros((n + 1, m + r + 1, 4))
        for row in xrange(n + 1):
            VQ, Qw[row,:] = \
                    curve.curve_knot_ins(m, q, V, Pw[row,:], u, k, s, r)
    return UQ, VQ, Qw


def refine_knot_vect_surface(n, p, U, m, q, V, Pw, X, di):

    ''' Let Sw(u,v) be a NURBS surface on U and V.  A U (V) knot vector
    refinement is accomplished by simply applying curve knot refinement
    to the m + 1 (n + 1) columns (rows) of control points.

    Source: The NURBS Book (2nd Ed.), Pg. 167.

    '''

    r = len(X) - 1
    if di == 0:
        VQ = V.copy()
        Qw = np.zeros((n + r + 2, m + 1, 4))
        for col in xrange(m + 1):
            UQ, Qw[:,col] = curve.refine_knot_vect_curve(n, p, U, Pw[:,col], X)
    elif di == 1:
        UQ = U.copy()
        Qw = np.zeros((n + 1, m + r + 2, 4))
        for row in xrange(n + 1):
            VQ, Qw[row,:] = curve.refine_knot_vect_curve(m, q, V, Pw[row,:], X)
    return UQ, VQ, Qw


def decompose_surface(n, p, U, m, q, V, Pw, di):

    ''' The function computes a Bezier strip, i.e., a NURBS surface that
    is Bezier in one direction and B-spline in the other.  The function
    must be called twice, once in the u direction to get the Bezier
    strips, and then the strips must be fed into the function in the v
    direction to get the Bezier patches.

    Source: The NURBS Book (2nd Ed.), Pg. 177.

    '''

    if di == 0:
        ni = np.unique(U[p+1:n+1]).size
        Qw = np.zeros((ni + 1, p + 1, m + 1, 4))
        for col in xrange(m + 1):
            nb, Ub, Qw[:,:,col] = curve.decompose_curve(n, p, U, Pw[:,col])
    elif di == 1:
        ni = np.unique(V[q+1:m+1]).size
        Qw = np.zeros((ni + 1, n + 1, q + 1, 4))
        for row in xrange(n + 1):
            nb, Ub, Qw[:,row,:] = curve.decompose_curve(m, q, V, Pw[row,:])
    return nb, Ub, Qw


def remove_surface_knot(n, p, U, m, q, V, Pw, u, r, s, num, d, di):

    ''' A u knot (v knot) is removed from Sw(u,v) by applying the curve
    knot removal algorithm to the m + 1 columns (n + 1 rows) of control
    points.  But the knot can be removed only the removal is successful
    for all m + 1 columns (n + 1 rows).

    Source: The NURBS Book (2nd Ed.), Pg. 186.

    '''

    if di == 0:
        VQ = V.copy()
        for col in xrange(m + 1):
            nr, UQ, Q = curve.remove_curve_knot(n, p, U, Pw[:,col],
                                                u, r, s, num, d)
            if col == 0:
                nr0 = nr
                Qw = np.zeros((n - nr + 1, m + 1, 4))
            if nr == 0 or not nr == nr0:
                return 0, U, V, Pw
            Qw[:,col] = Q
    elif di == 1:
        UQ = U.copy()
        for row in xrange(n + 1):
            nr, VQ, Q = curve.remove_curve_knot(m, q, V, Pw[row,:],
                                                u, r, s, num, d)
            if row == 0:
                nr0 = nr
                Qw = np.zeros((n + 1, m - nr + 1, 4))
            if nr == 0 or not nr == nr0:
                return 0, U, V, Pw
            Qw[row,:] = Q
    return nr, UQ, VQ, Qw


def remove_surface_uknots(n, p, U, m, Pw, d=1e-3):

    ''' Remove as many u knots from a surface as possible.  The v-knot
    case is analogous.

    Source: Tiller, Knot-removal algorithms for NURBS curves and
            surfaces, CAD, 1992.

    '''

    if n < p + 1:
        return 0, U, Pw
    r = U.shape[0] - 1
    TOL = curve.calc_tol_removal(Pw, d)
    alfas = np.zeros((2 * p + 1))
    tmp = np.zeros((m + 1, 2 * p + 1, 4))
    U = U.copy(); Pw = Pw.copy()
    o = p + 1; hispan = r - o; hiu = U[hispan]
    gap = 0; u = U[o]; r = o
    while u == U[r+1]:
        r += 1
    s = r - p; fout = (2 * r - s - p) // 2
    first = s; last = r - s
    bgap = r; agap = bgap + 1
    while True:
        nr = 0
        for t in xrange(s):
            remflag = 1
            lf2 = last - first + 2
            i = first; j = last
            while j - i > t:
                alfas[i-first+1] = (u - U[i]) / (U[i+o+gap+t] - U[i])
                alfas[j-first+1] = (u - U[j-t]) / (U[j+o+gap] - U[j-t])
                i += 1; j -= 1
            if j - i == t:
                alfas[i-first+1] = (u - U[i]) / (U[i+o+gap+t] - U[i])
            for k in xrange(m + 1):
                ki = first - 1; kj = last + 1
                tmp[k,0] = Pw[ki,k]; tmp[k,lf2] = Pw[kj,k]
                ki += 1; kj -= 1; i = 1; j = lf2 - 1
                while j - i > t:
                    alfi, alfj = alfas[i], alfas[j]
                    tmp[k,i] = (Pw[ki,k] - (1.0 - alfi) * tmp[k,i-1]) / alfi
                    tmp[k,j] = (Pw[kj,k] - alfj * tmp[k,j+1]) / (1.0 - alfj)
                    i += 1; ki += 1; j -= 1; kj -= 1
                if j - i < t:
                    if util.distance(tmp[k,i-1], tmp[k,j+1]) > TOL:
                        remflag = 0
                        break
                else:
                    alfi = alfas[i]
                    if (util.distance(Pw[ki,k], alfi * tmp[k,i+t+1] +
                                      (1.0 - alfi) * tmp[k,i-1])) > TOL:
                        remflag = 0
                        break
            if remflag == 0:
                break
            nr += 1
            for k in xrange(m + 1):
                ki, kj = first, last
                i, j = 1, lf2 - 1
                while j - i > t:
                    Pw[ki,k], Pw[kj,k] = tmp[k,i], tmp[k,j]
                    ki += 1; kj -= 1; i += 1; j -= 1
            first -= 1; last += 1
        if nr > 0:
            j = fout; i = j
            for k in xrange(1, nr):
                if (np.mod(k, 2) == 1):
                    i += 1
                else:
                    j -= 1
            i += 1
            for k in xrange(m + 1):
                kj = j
                for ki in xrange(i, bgap + 1):
                    Pw[kj,k] = Pw[ki,k]
                    kj += 1
            bgap -= nr
        if u == hiu:
            gap += nr
            break
        else:
            j = i = r - nr + 1; k = r + gap + 1
            u = U[k]
            while u == U[k]:
                U[i] = U[k]
                i += 1; k += 1
            s = i - j; r = i - 1
            gap += nr
            ki = bgap + 1
            for k in xrange(m + 1):
                i = ki
                for j in xrange(s):
                    Pw[i,k] = Pw[agap+j,k]
                    i += 1
            bgap += s; agap += s
            fout = (2 * r - p - s) // 2
            first = r - p; last = r - s
    if gap == 0:
        return gap, U, Pw
    i = hispan + 1; k = i - gap
    for j in xrange(1, o + 1):
        U[k] = U[i]
        k += 1; i += 1
    UQ = U[:-gap]; Qw = Pw[:-gap,:]
    return gap, UQ, Qw


def degree_elevate_surface(n, p, U, m, q, V, Pw, t, di):

    '''  Degree elevation is accomplished for surfaces by applying it to
    the rows/columns of control points.  In particular, degree p (u
    direction) is elevated by applying curve degree elevation to each of
    the (m + 1) columns of control points.  The v direction degree q is
    elevated by applying the same algorithm to each of (n + 1) rows of
    control points.

    Source: The NURBS Book (2nd Ed.), Pg. 209.

    '''

    if di == 0:
        VQ = V.copy()
        nh = curve.mult_degree_elevate(n, p, U, t)[0]
        Qw = np.zeros((nh + 1, m + 1, 4))
        for col in xrange(m + 1):
            UQ, Qw[:,col] = curve.degree_elevate_curve(n, p, U, Pw[:,col], t)
    elif di == 1:
        UQ = U.copy()
        mh = curve.mult_degree_elevate(m, q, V, t)[0]
        Qw = np.zeros((n + 1, mh + 1, 4))
        for row in xrange(n + 1):
            VQ, Qw[row,:] = curve.degree_elevate_curve(m, q, V, Pw[row,:], t)
    return UQ, VQ, Qw


def degree_reduce_surface(n, p, U, m, q, V, Pw, d, di):

    ''' u and v-degree of a surface can be reduced by applying the curve
    degree reduction algorithm to the rows or columns of control points.

    Source: The NURBS Book (2nd Ed.), Pg. 227.

    '''

    if di == 0:
        VQ = V.copy()
        nh = curve.mult_degree_reduce(n, p, U)
        Qw = np.zeros((nh + 1, m + 1, 4))
        for col in xrange(m + 1):
            UQ, Qw[:,col] = curve.degree_reduce_curve(n, p, U, Pw[:,col], d)
    elif di == 1:
        UQ = U.copy()
        mh = curve.mult_degree_reduce(m, q, V)
        Qw = np.zeros((n + 1, mh + 1, 4))
        for row in xrange(n + 1):
            VQ, Qw[row,:] = curve.degree_reduce_curve(m, q, V, Pw[row,:], d)
    return UQ, VQ, Qw


# ADVANCED GEOMETRIC ALGORITHMS


def surface_point_projection(n, p, U, m, q, V, Pw, Pi,
                             uvi=None, eps1=1e-15, eps2=1e-12):

    ''' Find the parameter values ui, vi for which S(ui,vi) is closest
    to Pi.  This is achieved by solving the two following functions
    simultaneously: f(u,v) = Su(u,v) * (S(u,v) - Pi) = 0 and g(u,v) =
    Sv(u,v) * (S(u,v) - Pi) = 0.  Two zero tolerances are used to
    indicate convergence: (1) eps1, a measure of Euclidean distance and
    (2) eps2, a zero cosine measure.

    Source: The NURBS Book (2nd Ed.), Pg. 229.

    '''

    if uvi is None:
        num = 100 # knob
        us, vs = util.construct_flat_grid((U, V), 2 * (num,))
        S = rat_surface_point_v(n, p, U, m, q, V, Pw, us, vs, num**2)
        i = np.argmin(util.distance_v(S, Pi))
        ui, vi = us[i], vs[i]
    else:
        ui, vi = uvi
    P, w = Pw[:,:,:-1], Pw[:,:,-1]
    rat = (w != 1.0).any()
    J = np.zeros((2, 2))
    K = np.zeros(2)
    for ni in xrange(20): # knob
        SKL = surface_derivs_alg1(n, p, U, m, q, V, P, ui, vi, 2)
        if rat:
            wders = surface_derivs_alg1(n, p, U, m, q, V, w, ui, vi, 2)
            SKL = rat_surface_derivs(SKL, wders, 2)
        S = SKL[0,0]
        SU, SUU = SKL[1,0], SKL[2,0]
        SV, SVV = SKL[0,1], SKL[0,2]
        SUV = SVU = SKL[1,1]
        R = S - Pi; RN = util.norm(R)
        SUN = util.norm(SU)
        SVN = util.norm(SV)
        if RN <= eps1 or SUN == 0.0 or SVN == 0.0:
            return ui, vi
        SUR = np.dot(SU, R)
        SVR = np.dot(SV, R)
        zero_cosine1 = abs(SUR) / SUN / RN
        zero_cosine2 = abs(SVR) / SVN / RN
        if zero_cosine1 <= eps2 and zero_cosine2 <= eps2:
            return ui, vi
        SUVR = np.dot(SUV, R)
        SUV = np.dot(SU, SV)
        J[:,:] = [[util.norm(SU)**2 + np.dot(SUU, R), SUV + SUVR],
                  [SUV + SUVR, util.norm(SV)**2 + np.dot(SVV, R)]]
        K[:] = [SUR, SVR]
        d0, d1 = np.linalg.solve(J, - K)
        uii, vii = d0 + ui, d1 + vi
        if uii < U[0]:
            uii = U[0]
        elif uii > U[-1]:
            uii = U[-1]
        if vii < V[0]:
            vii = V[0]
        elif vii > V[-1]:
            vii = V[-1]
        if util.norm((uii - ui) * SU + (vii - vi) * SV) <= eps1:
            return uii, vii
        ui, vi = uii, vii
    raise nurbs.NewtonLikelyDiverged(ui, vi)


def reverse_surface_direction(n, p, U, m, q, V, Pw, di):

    ''' Reverse one direction of a surface while maintaining its overall
    parameterization.

    Source: The NURBS Book (2nd Ed.), Pg. 263.

    '''

    if di == 0:
        T = V.copy()
        Qw = np.zeros((n + 1, m + 1, 4))
        for col in xrange(m + 1):
            S, Qw[:,col] = curve.reverse_curve_direction(n, p, U, Pw[:,col])
    elif di == 1:
        S = U.copy()
        Qw = np.zeros((n + 1, m + 1, 4))
        for row in xrange(n + 1):
            T, Qw[row,:] = curve.reverse_curve_direction(m, q, V, Pw[row,:])
    return S, T, Qw


def surface_surface_intersect(S0, S1, P0, CRT=5e-3, ANT=1e-2):

    ''' Find the approximation curve that intersects two Surfaces.

    Given two Surfaces S0 and S1, the starting point P0 (at which
    originally S0(s0,t0) ~= S1(u0,v0)) is marched along the intersection
    curve until a boundary is reached.  Each marched point follows a
    two-step process: (1) determine a local, unit step direction, V, and
    step length, L, to guess a new approximation point, P1h (= P0 +
    L*V), and (2) relax P1h onto the "true" intersection curve.

    1. For the very first point, V is determined by intersecting the
    tangent planes of the two surfaces at P0; all remaining Vs are taken
    as the difference of previous intersection points (backtracking).
    As for L, it is obtained from an adaptive method that is based on
    curvature, rho, and an angle tolerance, ANT, i.e.:

                        L = min(rho * ANT, CRT)

    where CRT denotes a Curve Refinement Tolerance.

    2. Once P1h has been calculated, it is "relaxed" onto the true
    intersection by solving the following three nonlinear equations in
    four unknowns:

                         S1(u,v) - S0(s,t) = 0

    Given a first iterate, (s1h, t1h, u1h, v1h), the above can be
    linearized and interpreted geometrically as the intersection between
    the tangent plane to S0 at (s1h, t1h) and the tangent plane to S1 at
    (u1h, v1h).  The midpoint of that intersection is chosen as the next
    iterate.  Convergence is reached when |S1(u,v) - S0(s,t)| <= Same
    Point Tolerance (SPT).

    Note that this function assumes that P0 originally lies on (or is
    very close to) one and only one of the four boundary curves of S0
    and/or S1.  Thus, intersections starting from the crossing of two
    boundaries are allowed (see example below).  However, intersections
    arising from self-intersecting Surfaces are not supported.

    Parameters
    ----------
    S0, S1 = the two Surface to intersect
    P0 = the starting (boundary) point (in x, y, z format)
    CRT = the Curve Refinement Tolerance
    ANT = the ANgular Tolerance

    Returns
    -------
    Q = the approximated points lying on the intersection curve
    s, t, u, v = the parametric values corresponding to Q

    Source
    ------
    Barnhill & Kersey, A marching method for parametric surface/surface
    intersection, CAGD, 1990.

    Examples
    --------
    >>> P00 = Point( 1, -1)
    >>> P01 = Point(-1, -1)
    >>> P10 = Point( 1,  1)
    >>> P11 = Point(-1,  1)
    >>> S0 = nurbs.tb.make_bilinear_surface(P00, P01, P10, P11)
    >>> S1 = S0.copy()
    >>> S1.rotate(90, L=[1,0,0])
    >>> Q = nurbs.surface.surface_surface_intersect(S0, S1, (-1,0,0))

    '''

    def find_closest_edge():
        ''' Find which boundary (edge) the current P0 is closest to. '''

        global s, t, u, v

        l2ns = []
        for S in S0, S1:
            U, V = S.U
            uedi = zip([U[0], U[-1], V[0], V[-1]],
                       [0, 0, 1, 1])
            for ue, di in uedi:
                C = S.extract(ue, di)
                try:
                    up, = C.project(P0)
                except nurbs.NewtonLikelyDiverged:
                    l2n = np.inf
                else:
                    P = C.eval_point(up)
                    l2n = util.distance(P0, P)
                l2ns.append(l2n)
        ed = np.argmin(l2ns)
        S = S0 if ed < 4 else S1
        U, V = S.U
        mask = np.array(True).repeat(4)
        if S is S0:
            if ed in (0, 1):
                mask[0] = False
                s = U[0] if ed == 0 else U[-1]
            elif ed in (2, 3):
                mask[1] = False
                t = V[0] if ed == 2 else V[-1]
        elif S is S1:
            if ed in (4, 5):
                mask[2] = False
                u = U[0] if ed == 4 else U[-1]
            elif ed in (6, 7):
                mask[3] = False
                v = V[0] if ed == 6 else V[-1]
        return mask

    def relax_point_bnd():
        ''' Relax the current P0 onto the closest boundary by minimizing
        the distance between the two Surfaces. '''

        def dist(stu):
            try:
                stuv[mask] = stu
                return util.distance(S0.eval_point(stuv[0], stuv[1]),
                                     S1.eval_point(stuv[2], stuv[3]))
            except knot.KnotOutsideKnotVectorRange:
                return np.inf

        global s, t, u, v

        mask = find_closest_edge()
        stuv = np.array((s, t, u, v))
        stuv[mask] = fmin(dist, stuv[mask], xtol=PBT, ftol=PBT, disp=False)
        s, t, u, v = stuv
        return S0.eval_point(s, t)

    def relax_point(L, V):
        ''' Update the current P0 by (L * V) and then relax it onto the
        true intersection curve. '''

        global s, t, u, v

        Pi = P0 + L * V
        for ni in xrange(50): # knob
            try:
                arg0 = args0 + (Pi, (s, t)) + PST
                arg1 = args1 + (Pi, (u, v)) + PST
                s, t = surface_point_projection(*arg0)
                u, v = surface_point_projection(*arg1)
            except nurbs.NewtonLikelyDiverged:
                return P0
            SD0 = S0.eval_derivatives(s, t, 1)
            SD1 = S1.eval_derivatives(u, v, 1)
            Q0 = SD0[0,0]
            Q2 = SD1[0,0]
            Q1 = (Q0 + Q2) / 2.0
            l2n = util.distance(Q0, Q2)
            if l2n <= SPT:
                return Q1
            N0 = np.cross(SD0[1,0], SD0[0,1])
            N2 = np.cross(SD1[1,0], SD1[0,1])
            N1 = np.cross(N0, N2)
            if (N1 == 0.0).all(): # parallel tangent planes
                return P0
            Pi = util.intersect_three_planes(Q0, N0, Q2, N2, Q1, N1)
        raise SurfaceSurfaceLikelyDiverged()

    global s, t, u, v

    SPT = 1e-7       # Same Point Tolerance
    PST = 1e-8, 1e-7 # Point coincidence and zero cosine Tolerances (Surface)
    PBT = 1e-15      # Point coincidence Tolerance (Boundary)
    SRT = 1e-4       # Significant pRogress Tolerance

    args0 = S0.var()
    args1 = S1.var()

    arg0 = args0 + (P0, None) + PST
    arg1 = args1 + (P0, None) + PST
    s, t = surface_point_projection(*arg0)
    u, v = surface_point_projection(*arg1)

    # find a close boundary and relax P0 onto it
    P0 = relax_point_bnd()
    Q, stuv = [P0], [(s, t, u, v)]

    # get an initial V
    degs = np.linspace(0, 2 * np.pi, 17)
    D, V = np.inf, None
    for deg in degs[:-1]:
        ds, dt = CRT * np.array((np.cos(deg), np.sin(deg)))
        try:
            Vi = S0.eval_point(s + ds, t + dt) - P0
        except knot.KnotOutsideKnotVectorRange:
            continue
        Vi = util.normalize(Vi)
        try:
            P1 = relax_point(CRT, Vi)
        except SurfaceSurfaceLikelyDiverged:
            continue
        Pi = P0 + CRT * Vi
        Di = util.distance(Pi, P1)
        if Di < D:
            D, V = Di, Vi
    if V is None:
        raise NoInitialStepDirectionCouldBeFound()

    # start marching
    nm = 0
    while True:

        # adaptive step length
        rho0 = abs(S0.eval_curvature(s, t))
        rho1 = abs(S1.eval_curvature(u, v))
        rho0 = 1.0 / rho0 if rho0 else np.inf
        rho1 = 1.0 / rho1 if rho1 else np.inf
        rho = min(rho0, rho1)
        L = min(rho * ANT, CRT)
        L = max(L, 5e-4) # knob

        # update and relax P0 onto the true intersection curve
        try:
            P1 = relax_point(L, V)
        except SurfaceSurfaceLikelyDiverged:
            print('nurbs.surface.surface_surface_intersect :: '
                  'march likely diverged, aborting')
            break
        l2n = util.distance(P0, P1)
        if l2n < SRT or np.dot(P1 - P0, V) < 0.0:
            # no significant progress has been made or marching
            # direction has reversed; assume we've reached a boundary
            P1 = relax_point_bnd()
            Q[-1], stuv[-1] = P1, (s, t, u, v)
            break
        Q.append(P1)
        stuv.append((s, t, u, v))

        # backtracking
        V = util.normalize(Q[-1] - Q[-2])
        P0 = P1

        nm += 1
        if np.mod(nm, 100) == 0:
            print('nurbs.surface.surface_surface_intersect :: '
                  'marched {} points'.format(nm))

    return np.array(Q), np.array(stuv)


# TOOLBOX


def make_bilinear_surface(P00, P01, P10, P11):

    ''' Construct a bilinear Surface.

    Let P00, P01, P10 and P11 be four Points in three-dimensional space:

                             v|
                              |
                          P01 ._________. P11
                              |         |
                              |         |
                              |         |
                          P00 |         | P10
                              ._________. ____
                                             u

    Clearly, the (nonrational) surface given by

       S(u,v) = sum_(i=0)^(1) sum_(j=0)^(1) Ni1(u) * Nj1(v) * Pij

    with U = V = {0,0,1,1} represents a bilinear interpolation between
    the four line segments P00-P10, P01-P11, P00-P01 and P10-P11.

    Parameters
    ----------
    P00, P01, P10, P11 = the four corner Points

    Returns
    -------
    Surface = the bilinear Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 333.

    '''

    cnet = ControlNet([[P00, P01], [P10, P11]])
    return Surface(cnet, (1,1))


def make_general_cylinder(C, d, W):

    ''' Construct a general cylinder.

    Let W be a vector of unit length and C(u) = sum_(i=0)^(n) (Rip(u) *
    Pi) be a pth-degree NURBS curve on the knot vector, U, with weights
    wi.  A general cylinder S(u,v) is obtained by sweeping C(u) a
    distance d along W.  Denoting the parameter for the sweep direction
    by v (0 <= v <= 1), the desired representation is

        S(u,v) = sum_(i=0)^(n) sum_(j=0)^(1) (Rip;j1(u,v) * Pij)

    on knot vectors U and V, where V = {0,0,1,1} and U is the knot
    vector of C(u).  The control points are given by Pi0 = Pi and Pi1 =
    Pi + d * W, and the weights are wi0 = wi1 = wi.

    Parameters
    ----------
    C = the Curve to sweep (rational or not)
    d = the distance to sweep
    W = the direction of the sweep

    Returns
    -------
    Surface = the general cylinder

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 334.

    '''

    n, p, U, Pw0 = C.var()
    Pw1 = Pw0.copy()
    dW = d * util.normalize(W)
    transform.translate(Pw1, dW)
    Pw = np.zeros((n + 1, 2, 4))
    Pw[:,0], Pw[:,1] = Pw0, Pw1
    return Surface(ControlNet(Pw=Pw), (p,1), (U,[0,0,1,1]))


def make_ruled_surface(C0, C1):

    ''' Construct a ruled Surface.

    Assume two NURBS curves

       C_k(u) = sum_(i=0)^(n_k) (R_(i,p_k)(u) * Pi^k)    k = 1,2

    defined on the knot vectors U^k = {u_0^k,...,u_(m_k)^k}.  A ruled
    surface in the v direction is a linear interpolation between C_1(u)
    and C_2(u).  The interpolation shall be between points of equal
    parameter value, i.e., for fixed ub, S(ub,v) is a straight line
    segment connecting the points C_1(ub) and C_2(ub).  The desired
    surface is of the form

        S(u,v) = sum_(i=0)^(n) sum_(j=0)^(1) (Rip;j1(u,v) * Pij)

    where V = {0,0,1,1} and n, U, p, wij and Pij must all be determined.

    Parameters
    ----------
    C0, C1 = the two Curves to rule (rational or not)

    Returns
    -------
    Surface = the ruled Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 337.

    '''

    n, p, U, Cs = curve.make_curves_compatible1([C0, C1])
    C0, C1 = Cs
    Pw = np.zeros((n + 1, 2, 4))
    Pw[:,0], Pw[:,1] = C0.cobj.Pw, C1.cobj.Pw
    return Surface(ControlNet(Pw=Pw), (p,1), (U,[0,0,1,1]))


def make_general_cone(C0, xyz):

    ''' Construct a general cone.

    The general cone is a type of ruled surface.  Let P be the vertex
    point of the cone, and let C_1(u) = sum_(i=0)^(n) (Rip(u) * Pi^1) be
    its base curve.  Define the curve C_2(u) = sum_(i=0)^(n) (Rip(u) *
    Pi^2) with Pi^2 = P and wi^2 = wi^1 for all i (a degenerate curve).
    The ruled surface between C_1(u) and C_2(u) is the desired cone.

    Parameters
    ----------
    C0 = the base Curve (rational or not)
    xyz = the xyz coordinates of the apex

    Returns
    -------
    Surface = the general cone

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 338.

    '''

    n, p, U, Pw0 = C0.var()
    Pw1 = Pw0.copy()
    w = Pw1[:,-1]
    Pw1[:,:-1] = xyz * w[:,np.newaxis]
    Pw = np.zeros((n + 1, 2, 4))
    Pw[:,0], Pw[:,1] = Pw0, Pw1
    return Surface(ControlNet(Pw=Pw), (p,1), (U,[0,0,1,1]))


def make_revolved_surface_rat(G, S, T, theta):

    ''' Construct a Surface of revolution.

    Let C(v) = sum_(j=0)^(m) (Rjq(v) * Pj) be a qth-degree NURBS curve
    on the knot vector V.  C(v) is called the generatrix, G, and it is
    to be revolved through an arbitrary angle, theta, about an arbitrary
    axis.  The axis is specified by a point, S, and a unit length
    vector, T.  S, T, and the control points of G should all lie in the
    same plane.

    Parameters
    ----------
    G = the generatrix Curve (rational or not)
    S = the xyz coordinates of a point on the axis of revolution
    T = a vector along the axis of revolution
    theta = the angle of revolution (in degrees)

    Returns
    -------
    Surface = the revolved Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 346.

    '''

    if not 0 < theta <= 360:
        raise ImproperInput(theta)
    if theta <= 90.0:
        narcs = 1
    elif theta <= 180.0:
        narcs = 2
    elif theta <= 270.0:
        narcs = 3
    else:
        narcs = 4
    m, q, V, Pw = G.var()
    n = 2 * narcs
    Pj = nurbs.obj_mat_to_3D(Pw)
    wj = Pw[:,-1]
    Pij = np.zeros((n + 1, m + 1, 3))
    wij = np.zeros((n + 1, m + 1, 1))
    U = np.zeros(n + 4)
    theta = np.deg2rad(theta)
    dtheta = theta / narcs
    wm = np.cos(dtheta / 2.0)
    angle = 0.0
    cosines = np.zeros(narcs + 1)
    sines = np.zeros(narcs + 1)
    for i in xrange(1, narcs + 1):
        angle += dtheta
        cosines[i] = np.cos(angle)
        sines[i] = np.sin(angle)
    for j in xrange(m + 1):
        O = util.point_to_line(S, T, Pj[j])
        X = Pj[j] - O
        if (X == 0.0).all():
            Pij[:,j] = O
            wij[0::2,j] = wj[j]
            wij[1::2,j] = wm * wj[j]
            continue
        r = util.norm(X)
        X = util.normalize(X)
        Y = np.cross(T, X)
        Pij[0,j] = P0 = Pj[j]
        wij[0,j] = wj[j]
        T0 = Y
        ind = 0
        for i in xrange(1, narcs + 1):
            P2 = O + r * cosines[i] * X + r * sines[i] * Y
            Pij[ind+2,j] = P2
            wij[ind+2,j] = wj[j]
            T2 = - sines[i] * X + cosines[i] * Y
            Pij[ind+1,j] = util.intersect_3D_lines(P0, T0, P2, T2)
            wij[ind+1,j] = wm * wj[j]
            ind += 2
            if i < narcs:
                P0, T0 = P2, T2
    j = 3 + 2 * (narcs - 1)
    for i in xrange(3):
        U[i+j] = 1.0
    if narcs == 2:
        U[3] = U[4] = 0.5
    elif narcs == 3:
        U[3] = U[4] = 1.0 / 3.0
        U[5] = U[6] = 2.0 / 3.0
    elif narcs == 4:
        U[3] = U[4] = 0.25
        U[5] = U[6] = 0.5
        U[7] = U[8] = 0.75
    Pw = np.concatenate((Pij * wij, wij), axis=2)
    return Surface(ControlNet(Pw=Pw), (2,q), (U,V))


def make_revolved_surface_nrat(G, S, T, theta, p=3, TOL=1e-5):

    ''' Construct an approximation to a Surface of revolution.

    Let C(v) = sum_(j=0)^(m) (Njq(v) * Pj) be a qth-degree nonuniform,
    nonrational curve on the knot vector V.  C(v) is called the
    generatrix, G, and it is to be revolved through an arbitrary angle,
    theta, about an arbitrary axis.  The axis is specified by a point,
    S, and a unit length vector, T.  S, T, and the control points of G
    must all lie in the same plane.

    Parameters
    ----------
    G = the generatrix Curve (nonrational)
    S = the xyz coordinates of a point on the axis of revolution
    T = a vector along the axis of revolution
    theta = the angle of revolution (in degrees)

    See conics.make_circle_nrat for a description of p and TOL.

    Returns
    -------
    Surface = the revolved Surface

    Source
    ------
    Piegl & Tiller, Approximating surfaces of revolution by nonrational
    B-splines, IEEE Computer Graphics and Applications, 2003.

    '''

    if not 0 < theta <= 360:
        raise ImproperInput(theta)
    if G.isrational:
        raise nurbs.RationalNURBSObjectDetected(G)
    m, q, V, Pw = G.var()
    O = np.zeros((m + 1, 3)); X, Y = O.copy(), O.copy()
    r = np.zeros(m + 1)
    Pj = nurbs.obj_mat_to_3D(Pw)
    for j in xrange(m + 1):
        O[j] = util.point_to_line(S, T, Pj[j])
        X[j] = Pj[j] - O[j]
        r[j] = util.norm(X[j])
        if not (r[j] == 0.0).all():
            X[j] = X[j] / r[j]
        Y[j] = np.cross(T, X[j])
    jm, rm = np.argmax(r), np.max(r)
    Mc = conics.make_circle_nrat(O[jm], X[jm], Y[jm], rm, 0, theta, p, TOL)
    n, p, U, Pw = Mc.var()
    Pij = np.zeros((n + 1, m + 1, 3))
    for j in xrange(m + 1):
        Qw = Pw.copy()
        transform.translate(Qw, O[j] - O[jm])
        transform.scale(Qw, r[j]/rm, O[j])
        Pij[:,j] = Qw[:,:-1]
    Pw = nurbs.obj_mat_to_4D(Pij)
    return Surface(ControlNet(Pw=Pw), (p,q), (U,V))


def make_skinned_surface(Cs, q, vb=None):

    ''' Skin a Surface through a set of lofted Curves.

    Let

              C_k(u) = sum_(i=0)^n (Nip(u) * Pij)    k = 0,...,K

    be a set of nonrational curves (all weights equal 1.0).  The C_k(u)
    are called section curves.  Skinning is a process of blending the
    section curves together to form a surface.  Skinning methods
    interpolate through the C_k(u), with the result that the C_k(u) are
    isoparametric curves on the resulting skinned surface.

    If necessary, all C_k(u) are made compatible (brought to the same
    knot vector U and degree p).  Then for the v direction a degree q is
    chosen, and parameters {vb}, k=0,...,K, and a knot vector, V, are
    computed.  These are then used to do (n + 1) curve interpolations
    across the control points of the section curves.

    Interactively designing skinned surfaces is often a tedious and
    iterative process.  Surface shape is difficult to control; it
    depends on the number, shape, and positioning of the section curves,
    as well as on the method used for the v-directional interpolation.
    Often the design process consists of starting with a small number of
    section curves and then iteratively skinning, taking plane cuts to
    obtain additional section curves, modifying the new curces to obtain
    more desirable shapes, and reskinning.

    Parameters
    ----------
    Cs = the cross sections to skin (a list of nonrational Curves)
    q = the degree of the interpolation in v
    vb = the parameter values at the cross-sections (if available)

    Returns
    -------
    Surface = the skinned Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 457.

    '''

    if any([c.isrational for c in Cs]):
        raise nurbs.RationalNURBSObjectDetected()
    K = len(Cs) - 1
    if K < q:
        raise ImproperInput(K, q)
    n, p, U, Cs = curve.make_curves_compatible1(Cs)
    if vb is None:
        Q = [C.cobj.Pw[np.newaxis,:,:-1] for C in Cs]
        Q = np.vstack(Q)
        vb = knot.skinned_param(n, K, Q)
    V = knot.averaging_knot_vec(K, q, vb)
    Qw = np.zeros((n + 1, K + 1, 4))
    for i in xrange(n + 1):
        P = np.zeros((K + 1, 3))
        for k in xrange(K + 1):
            P[k] = Cs[k].cobj.Pw[i,:-1]
        V, Qw[i,:] = fit.global_curve_interp(K, P, q, vb, V)
    return Surface(ControlNet(Pw=Qw), (p,q), (U,V))


def make_swept_surface(C, T, Bv, K, scs=None, Ce=None, local=None):

    ''' Sweep a section Curve along an arbitrary trajectory Curve.

    Denote the trajectory curve by T(v) and the section Curve by C(u).
    A general form of the swept surface is given by

                   S(u,v) = T(v) + A(v) * S(v) * C(u)

    where A(v) is a general transformation matrix and S(v) is a scaling
    matrix.  The isoparametric curves on S(u,v) at fixed v values are
    instances of C(u), first transformed by A(v) * S(v) and then
    translated by T(v).  If C(u) passes through the global origin at (u
    = ub), then T(v) is the isoparametric curve S(u=ub,v).

    In general, A(v) is not representable in NURBS form.  Hence, an
    approximation to S(u,v), denoted by Sh(u,v), is constructed by
    skinning (see make_skinned_surface) across (K + 1) instances of C(u)
    along T(v).  Thus, the trajectory's degree q and knot vector V are
    inherited by Sh(u,v), and its accuracy can be increased by
    increasing K.

    Note that this function relies on an appropriate orientation
    function, B(v), that, as its name suggests, orients C(u) along T(v).
    The determination of this function is a critical yet nontrivial part
    of the sweeping process.  You may use get_sweep_orient_func to get
    it for you.

    Parameters
    ----------
    C = the section Curve (rational or not)
    T = the trajectory Curve (nonrational)
    Bv = the orientation function (a B-spline function, rational or not)
    K + 1 = the (minimum) number of instances of C taken along T
    scs = the B-spline scaling functions, if any, defined with respect
          to the parameter values v of T (a 3-tuple of B-spline functions
          corresponding to scaling in X, Y and Z, respectively)
    Ce = the end section Curve, if any (rational or not)
    local = the function that evaluates the local coordinate system of C
            along T; defaults to get_sweep_local_sys

    Returns
    -------
    Surface = the swept Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 475.

    '''

    if T.isrational:
        raise nurbs.RationalNURBSObjectDetected(T)
    n, p, U, dummy = C.var()
    if Ce:
        n, p, U, Cs = curve.make_curves_compatible1([C, Ce])
        C, Ce = Cs
    dummy, q, V, dummy = T.var()
    ktv = V.size
    nsect = K + 1
    if ktv <= nsect + q:
        m = nsect + q - ktv + 1
        V = knot.midpoint_longest_knot_vec_ins(V, m)
    elif ktv > nsect + q + 1:
        nsect = ktv - q - 1
    vb = np.zeros(nsect)
    vb[0] = V[0]
    for k in xrange(1, nsect - 1):
        vb[k] = sum(V[k+1:k+q+1]) / q
    vb[nsect-1] = V[-1]
    local = get_sweep_local_sys if not local else local
    I = np.identity(3)
    Qw = np.zeros((n + 1, nsect, 4))
    for k in xrange(nsect):
        v = vb[k]
        if Ce:
            #BWB
            #if v > 0.4:
            #    vtmp = 1.0 / (1.0 - 0.4) * v - 2.0 / 3.0
            #    Qw[:,k] = (1.0 - vtmp) * C.cobj.Pw + vtmp * Ce.cobj.Pw
            #else:
            #    Qw[:,k] = C.cobj.Pw.copy()
            v0 = (v - V[0]) / (V[-1] - V[0])
            #v0 = np.sin(v0*np.pi/2.0) # TMP!!!
            Qw[:,k] = (1.0 - v0) * C.cobj.Pw + v0 * Ce.cobj.Pw
        else:
            Qw[:,k] = C.cobj.Pw.copy()
        if scs:
            for di, sc in zip(I, scs):
                if sc:
                    sf = sc.eval_point(v)[0]
                    transform.scale(Qw[:,k], sf, L=di)
        O, X, Y, Z = local(v, T, Bv)
        transform.custom(Qw[:,k], R=np.column_stack((X, Y, Z)), T=O)
    for i in xrange(n + 1):
        V, Qw[i,:] = fit.global_curve_interp(nsect - 1, Qw[i,:], q, vb, V)
    return Surface(ControlNet(Pw=Qw), (p,q), (U,V))


def make_gordon_surface(Ck, Cl, ul, vk, pc=None, qc=None):

    ''' Interpolate a bidirectional Curve network.

    Let

     Ck(u) = sum_(i=0)^(n) (Nip(u) * Pki)  k = 0,...,r   u in [0,1]
     Cl(v) = sum_(j=0)^(m) (Njq(v) * Plj)  l = 0,...,s   v in [0,1]

    be two sets of nonrational B-spline Curves satisfying the
    compatibility conditions:

      - as independent sets they are compatible in the B-spline sense,
        that is, all the Ck(u) are defined on a common vector, UC, and
        all the Cl(v) are defined on a common knot vector, VC;

      - there exist parameters (0 = u_0 < u_1 < ... < u_(s-1) < u_s =
        1) and (0 = v_0 < v_1 < ... < v_(r-1) < v_r = 1) such that

            Qlk = Ck(ul) = Cl(vk)  k = 0,...,r  l = 0,...,s

    The desired NURBS surface, S(u,v), interpolates the two sets of
    curves, that is

                     S(ul,v) = Cl(v)    l = 0,...,s
                     S(u,vk) = Ck(u)    k = 0,...,r

    Gordon showed that the surface S(u,v) = L1(u,v) + L2(u,v) - T(u,v)
    satisfies these conditions, where L1(u,v) and L2(u,v) are
    respectively the u and v-directional skinned surfaces and T(u,v) is
    a surface interpolating the points Qlk.

    Parameters
    ----------
    Ck, Cl = the input Curves (nonrational)
    ul, vk = the parameters of the curve intersection points
    pc, qc = the degrees for interpolations

    Returns
    -------
    Surface = the Gordon Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 494.

    '''

    s = len(Cl) - 1; r = len(Ck) - 1
    if len(ul) - 1 != s or len(vk) - 1 != r:
        raise ImproperInput(s, r)
    Q = np.zeros((s + 1, r + 1, 3))
    for l in xrange(s + 1):
        cl = Cl[l]
        for k in xrange(r + 1):
            ck = Ck[k]
            Qlk1, Qlk2 = cl.eval_point(vk[k]), ck.eval_point(ul[l])
            l2n = util.distance(Qlk1, Qlk2)
            if l2n > 1e-3:
                print('nurbs.surface.make_gordon_surface :: '
                      'point inconsistency ({})'.format(l2n))
            Q[l,k] = Qlk1
    dummy, ppc, dummy, Ck = curve.make_curves_compatible1(Ck)
    dummy, qqc, dummy, Cl = curve.make_curves_compatible1(Cl)
    pc = pc if pc else min(ppc, s)
    qc = qc if qc else min(qqc, r)
    L1 = make_skinned_surface(Cl, pc, ul); L1 = L1.swap()
    L2 = make_skinned_surface(Ck, qc, vk)
    UT, VT, PT = fit.global_surf_interp(s, r, Q, pc, qc, ul, vk)
    T = Surface(ControlNet(Pw=PT), (pc,qc), (UT,VT))
    dummy, dummy, p, q, U, V, Ss = make_surfaces_compatible1([L1, L2, T])
    Pij = Ss[0].cobj.Pw + Ss[1].cobj.Pw - Ss[2].cobj.Pw
    return Surface(ControlNet(Pw=Pij), (p,q), (U,V))


def make_coons_surface(Ck, Cl, Dk, Dl, eps=None):

    ''' Create a bicubically blended Coons Surface.

    Let

          Ck(u) = sum_(i=0)^(n) (Nip(u) * Pki)  k = 0,1   u in [0,1]
          Cl(v) = sum_(j=0)^(m) (Njq(u) * Plj)  l = 0,1   v in [0,1]

    be two sets of nonrational B-spline curves satisfying the
    compatibility conditions:

      - as independent sets they are compatible in the B-spline sense,
        that is, the two Ck(u) are defined on a common vector, U, and
        the two Cl(v) are defined on a common knot vector, V;

      -            S00 = C_(k=0)(u=0) = C_(l=0)(v=0)
                   S10 = C_(k=0)(u=1) = C_(l=1)(v=0)
                   S01 = C_(k=1)(u=0) = C_(l=0)(v=1)
                   S11 = C_(k=1)(u=1) = C_(l=1)(v=1)

    Also assume four cross-boundary derivative fields

       Dk(u) = sum_(i=0)^(n) (Nip(u) * Qki)  k = 0,1   u in [0,1]
       Dl(v) = sum_(j=0)^(m) (Njq(u) * Qlj)  l = 0,1   v in [0,1]

    The desired surface has the Ck(u) and Cl(v) as its boundaries, and
    the Dk(u) and Dl(v) as its first partial derivatives along the
    boundaries, that is

           D_(k=0)(u) = S_(v)(u,0)    D_(k=1)(u) = S_(v)(u,1)
           D_(l=0)(v) = S_(u)(0,v)    D_(l=1)(v) = S_(u)(1,v)

    The compatibility conditions

      - the Ck(u) and the Dk(u) are compatible in the B-spline sense, as
        are the Cl(v) and Dl(v);

      -                d D_(k=0)(u=0)   d D_(l=0)(v=0)
                 T00 = -------------- = --------------
                             du               dv

                       d D_(k=0)(u=1)   d D_(l=1)(v=0)
                 T10 = -------------- = --------------
                             du               dv

                       d D_(k=1)(u=0)   d D_(l=0)(v=1)
                 T01 = -------------- = --------------
                             du               dv

                       d D_(k=1)(u=1)   d D_(l=1)(v=1)
                 T11 = -------------- = --------------
                             du               dv

    must hold.  The Tkl are the four twist vectors.

    The most common use of this patch is to attempt to fill a four-sided
    hole in a patchwork of surfaces, smoothly in the G1 sense.  The
    boundary is extracted from the neighboring surfaces.  However, in
    this case derivative magnitudes are generally not meaningful, twist
    vectors are not compatible, and hence G1 continuity between the
    patch and its neighboring surfaces is not possible.  In order to
    address this, an angular tolerance, eps, can be specified to make
    the cross-derivatives twist (and boundary) compatible.

    Parameters
    ----------
    Ck, Cl = the four input Curves (nonrational)
    Dk, Dl = the four cross-derivative fields (nonrational Curves)
    eps = the angular tolerance, if any (in degrees)

    Returns
    -------
    Surface = the Coons Surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 502.

    '''

    if any([c.isrational for c in Ck + Cl + Dk + Dl]):
        raise nurbs.RationalNURBSObjectDetected()
    if eps is not None:
        Dk, Dl = make_boundary_compatible(Ck, Cl, Dk, Dl)
        Dk, Dl = make_twist_compatible(Dk, Dl, eps)
    check_coons_compatibility(Ck, Cl, Dk, Dl)
    n, p, U, CDk = curve.make_curves_compatible1(Ck + Dk)
    m, q, V, CDl = curve.make_curves_compatible1(Cl + Dl)
    Ck, Cl, Dk, Dl = CDk[:2], CDl[:2], CDk[2:], CDl[2:]
    S1 = blend_cubic_bezier(n, p, U, Ck, Dk, 0)
    S2 = blend_cubic_bezier(m, q, V, Cl, Dl, 1)
    T = blend_bicubic_bezier(Ck, Cl, Dk, Dl)
    dummy, dummy, p, q, U, V, Ss = make_surfaces_compatible1([S1, S2, T])
    Pij = Ss[0].cobj.Pw + Ss[1].cobj.Pw - Ss[2].cobj.Pw
    return Surface(ControlNet(Pw=Pij), (p,q), (U,V))


def make_nsided_region(CLRk, DLRk, CIk, BLRk, eps=1):

    ''' Fill an n-sided region with NURBS patches.

    Given a set of boundary curves

      Ck(u) = sum_(i=0)^(n_k) (N_(ip_k)(u) * P_(k_i))  k = 0,...,N

    and a set of corresponding cross-boundary derivatives

      Dk(u) = sum_(i=0)^(n_k) (N_(ip_k)(u) * V_(k_i))  k = 0,...,N

    a set of (N + 1) NURBS patches are sought to fill the region with
    G^(eps) continuity, so that, along each edge shared by two patches,
    the angle of the surface normals does not deviate more than eps.
    The general approach is:

     1. Split the boundary curves.
     2. Construct a central point and normal vector.
     3. Generate inner curves running from the central point to the
        split points of each boundary.
     4. Put bicubically blended Coons patches in each rectangular
        region.

    To construct these patches, a few requirements must be satisfied
    according to how the output is to be presented:

     1. The parameterizations of the boundary curves are consistent:

                      C_(k)(umax) = C_(k+1)(umin)

        where all indexes are computed modulo (N + 1), and umin and umax
        are minimum and maximum parameter bounds of the respective
        curves.

     2. The boundary curves and the corresponding cross-derivatives are
        compatible in the B-spline sense - they are defined over the
        same knot vector and have the same degree.

     3. The cross-derivatives point toward the inside of the region.  If
        not, the control vectors of the offending derivative curve can
        be flipped.

    Once these consistency checks (and possible modifications) have been
    made, a collection of NURBS surfaces can be placed in the region in
    a consistent manner.  Each corner of the region is the lower left
    corner of a patch, each split point of a boundary is the lower right
    corner of one patch and upper left corner of another, and the
    central point is the upper right corner of every patch.  This
    arrangement parameterizes the inner curves inwards.

    Parameters
    ----------
    CLRk = a list of (N + 1) tuples containing the left and right
           portion of the split boundary Curves
    DLRk = Idem CLRk, but contains the cross-boundary derivatives
    CIk = a list of (N + 1) inner Curves
    BLRk = a list of (N + 1) tuples containing the left and right inner
           cross-boundary derivatives
    eps = the tolerance on geometric continuity (in degrees)

    Returns
    -------
    Ss = a list of (N + 1) bicubically blended Coons patches

    Source
    ------
    Piegl & Tiller, Filling n-sided regions with NURBS patches, The
    Visual Computer, 1999.

    '''

    N = len(CLRk) - 1
    Ss = []
    for k in xrange(N + 1):
        Ck = [CLRk[k][0], CIk[np.mod(k - 1, N + 1)]]
        Cl = [CLRk[np.mod(k - 1, N + 1)][1], CIk[k]]
        Dk = [DLRk[k][0], BLRk[np.mod(k - 1, N + 1)][0]]
        Dl = [DLRk[np.mod(k - 1, N + 1)][1], BLRk[k][1]]
        Ss.append(make_coons_surface(Ck, Cl, Dk, Dl, eps))
    return Ss


def make_composite_surface(Ss, di=1, reorient=True, remove=True, d=0.0):

    ''' Link two or more Surfaces aligned in the v-direction to form one
    composite Surface.  In general a reorientation of the Surfaces is
    necessary as a preprocessing step; as a result some Surfaces may be
    flipped or/and have their axes swapped.

    Parameters
    ----------
    Ss = the connected Surfaces to unite
    di = the parametric direction in which to unite the Surfaces (0 or
         1)
    reorient = whether or not to reorient the Surfaces before uniting
               them (see reorient_surfaces)
    remove = whether or not to remove as many end knots as possible
    d = if remove is True, the maximum deviation allowed during the
        removal process

    Returns
    -------
    Surface = the composite Surface

    '''

    t0 = np.mod(di + 1, 2)
    if reorient:
        if di != 1:
            raise ImproperInput()
        Ss = reorient_surfaces(Ss)
    Ss = make_surfaces_compatible2(Ss, di=di)
    s = Ss[0]
    n, U, Pw = s.cobj.n[di], s.U[di], s.cobj.Pw
    UQ = U[:n+1]
    if di == 0:
        Qw = Pw[:-1,:]
    elif di == 1:
        Qw = Pw[:,:-1]
    for s in Ss[1:]:
        n, U, Pw = s.cobj.n[di], s.U[di], s.cobj.Pw
        UQ = np.append(UQ, U[1:n+1])
        if di == 0:
            Qw = np.append(Qw, Pw[:-1,:], axis=0)
        elif di == 1:
            Qw = np.append(Qw, Pw[:,:-1], axis=1)
    s = Ss[-1]
    n, p, U, Pw = s.cobj.n[di], s.p[di], s.U[di], s.cobj.Pw
    UQ = np.append(UQ, U[-p-1:])
    if di == 0:
        Qw = np.append(Qw, Pw[-1:,:], axis=0)
    elif di == 1:
        Qw = np.append(Qw, Pw[:,-1:], axis=1)
    UV = [0, 0]
    UV[di], UV[t0] = UQ, s.U[t0]
    cs = Surface(ControlNet(Pw=Qw), s.p, UV)
    if remove:
        rvs = [s.U[di][0] for s in Ss[1:]]
        mult = knot.find_int_mult_knot_vec(cs.p[di], cs.U[di])
        for rv in rvs:
            cs = cs.remove(rv, mult[rv] - 1, di, d)[1]
    return cs


# UTILITIES


def make_surfaces_compatible1(Ss):

    ''' Ensure that the Surfaces are defined on the same parameter
    ranges, be of common degrees and share the same knot vectors.

    '''

    p = max([s.p[0] for s in Ss])
    q = max([s.p[1] for s in Ss])
    Umin = min([s.U[0][ 0] for s in Ss])
    Umax = max([s.U[0][-1] for s in Ss])
    Vmin = min([s.U[1][ 0] for s in Ss])
    Vmax = max([s.U[1][-1] for s in Ss])
    Ss1 = []
    for s in Ss:
        dp, dq = p - s.p[0], q - s.p[1]
        s = s.elevate(dp, 0).elevate(dq, 1)
        knot.remap_knot_vec(s.U[0], Umin, Umax)
        knot.remap_knot_vec(s.U[1], Vmin, Vmax)
        Ss1.append(s)
    U = knot.merge_knot_vecs(*[s.U[0] for s in Ss1])
    V = knot.merge_knot_vecs(*[s.U[1] for s in Ss1])
    Ss2 = []
    for s in Ss1:
        s = s.refine(knot.missing_knot_vec(U, s.U[0]), 0)
        s = s.refine(knot.missing_knot_vec(V, s.U[1]), 1)
        Ss2.append(s)
    n, m = Ss2[0].cobj.n
    return n, m, p, q, U, V, Ss2


def make_surfaces_compatible2(Ss, di=1):

    ''' Make Surfaces compatible (in the di-direction only), and force
    the end parameter value of the ith Surface to be equal to the start
    parameter of the (i + 1)th Surface.  Also check if the end control
    points match.

    '''

    t0 = np.mod(di + 1, 2)
    p = max([s.p[0] for s in Ss])
    q = max([s.p[1] for s in Ss])
    Umin = min([s.U[t0][ 0] for s in Ss])
    Umax = max([s.U[t0][-1] for s in Ss])
    Ss1 = []
    for s in Ss:
        dp, dq = p - s.p[0], q - s.p[1]
        s = s.elevate(dp, 0).elevate(dq, 1)
        knot.remap_knot_vec(s.U[t0], Umin, Umax)
        Ss1.append(s)
    U = knot.merge_knot_vecs(*[s.U[t0] for s in Ss1])
    Ss2 = []
    for s in Ss1:
        s = s.refine(knot.missing_knot_vec(U, s.U[t0]), t0)
        Ss2.append(s)
    sl = Ss2[0]
    U = sl.U[di]; U -= U[0]
    knot.clean_knot_vec(U)
    for sr in Ss2[1:]:
        for i in xrange(sr.cobj.n[t0] + 1):
            lP = sl.cobj.cpts[(-1,i) if di == 0 else (i,-1)].xyz
            fP = sr.cobj.cpts[( 0,i) if di == 0 else (i, 0)].xyz
            l2n = util.norm(lP - fP)
            if l2n > 1e-3:
                print('nurbs.surface.make_surfaces_compatible2 :: '
                      'control point mismatch ({})'.format(l2n))
        U = sr.U[di]; U += sl.U[di][-1] - U[0]
        knot.clean_knot_vec(U)
        sl = sr
    return Ss2


def make_surfaces_compatible3(Ss, di=1):

    ''' Idem make_surfaces_compatible1, but in the di-direction only.
    Note that only Surfaces are returned.

    '''

    t0 = np.mod(di + 1, 2)
    p = max([s.p[di] for s in Ss])
    Umin = min([s.U[di][ 0] for s in Ss])
    Umax = max([s.U[di][-1] for s in Ss])
    Ss1 = []
    for s in Ss:
        s = s.elevate(p - s.p[di], di)
        knot.remap_knot_vec(s.U[di], Umin, Umax)
        Ss1.append(s)
    U = knot.merge_knot_vecs(*[s.U[di] for s in Ss1])
    Ss2 = []
    for s in Ss1:
        s = s.refine(knot.missing_knot_vec(U, s.U[di]), di)
        Ss2.append(s)
    return Ss2


def find_common_edges(sl, sr):

    ''' Find the common edges of two surfaces based on their corner
    points.  Also determine if the second surface should be flipped
    (swapped) in order to retrieve the orientation of the first one.  An
    error is raised if no or more than one common edge is found.

    Surface topology:          Edge 0
                        Crn0 .________. Crn3
                             | ___ v  |
                             ||       |
                      Edge 2 || u     | Edge 3
                             |        |
                             .________.
                        Crn1            Crn2
                               Edge 1

    '''

    edgemap = {(0, 3): 0, (3, 0): 0, (1, 2): 1, (2, 1): 1,
               (0, 1): 2, (1, 0): 2, (2, 3): 3, (3, 2): 3}

    cl, cr = np.zeros((4, 3)), np.zeros((4, 3))
    for k, ij in enumerate([(0,0), (-1,0), (-1,-1), (0,-1)]):
        cl[k] = sl.cobj.cpts[ij].xyz
        cr[k] = sr.cobj.cpts[ij].xyz
    crns = []
    for l in xrange(4):
        for r in xrange(4):
            l2n = util.norm(cl[l] - cr[r])
            if l2n < 1e-3:
                crns.append((l, r))
    edgs = zip(*crns)
    if ((len(crns) != 2) or
        (not all([edgs[i] in edgemap for i in (0, 1)]))):
        raise NoCommonEdgeCouldBeFound(sl, sr)
    ls = edgs[1]
    swap = True if ls[1] == np.mod(ls[0] + 1, 4) else False
    return edgemap[edgs[0]], edgemap[edgs[1]], swap


def reorient_surfaces(Ss):

    ''' Reorient all Surfaces according to the normal direction of the
    first one***, i.e. all local axes will be aligned likewise:
           _______   _______               _______   _______
          | ___ v | | ___ v |             |       | |       |
          ||      | ||      |      or     | u     | |  u    |
          || u    | || u    |             ||      | ||      |
          |       | |       | ...         ||___ v | ||___ v | ...
          |_______| |_______|             |_______| |_______|

    *** The first Surface may be reoriented, if necessary.


    '''

    reorientmap = {(0, 1, 2, 3): 'copy()',
                   (3, 2, 0, 1): 'reverse(0).swap()',
                   (2, 3, 1, 0): 'reverse(1).swap()',
                   (1, 0, 3, 2): 'reverse(0).reverse(1)'}

    Ss = list(Ss)
    for k in xrange(len(Ss) - 1):
        el, er, swap = find_common_edges(Ss[k], Ss[k+1])
        if k == 0:
            if el == 0:
                fei = (3, 2, 0, 1)
            elif el == 1:
                fei = (2, 3, 1, 0)
            elif el == 2:
                fei = (1, 0, 3, 2)
            elif el == 3:
                fei = (0, 1, 2, 3)
            Ss[0] = eval('Ss[0].' + reorientmap[fei])
            dummy, dummy, swap = find_common_edges(Ss[0], Ss[1])
        if swap:
            Ss[k+1] = Ss[k+1].swap()
        if er == 0:
            newei = (0, 1, 2, 3) if swap else (2, 3, 1, 0)
        elif er == 1:
            newei = (1, 0, 3, 2) if swap else (3, 2, 0, 1)
        elif er == 2:
            newei = (2, 3, 1, 0) if swap else (0, 1, 2, 3)
        elif er == 3:
            newei = (3, 2, 0, 1) if swap else (1, 0, 3, 2)
        Ss[k+1] = eval('Ss[k+1].' + reorientmap[newei])
    return Ss


def get_sweep_orient_func(B0, T, Tp=None, Tw=None, m=10):

    ''' Get the orientation function B(v) used by make_swept_surface.

    It is a vector-valued function satisfying (B(v) dot y(v) = 0) for
    all v, where y(v) = T'(v) / |T'(v)|.  If T(v) is planar, the
    solution is simple: let B(v) be a constant function whose value is a
    vector normal to the plane of T(v).

    The situation is more difficult for arbitrary trajectories.  A
    common solution is the use of the projection normal method: let
    {vi}, i=0,...m, be an increasing sequence of parameter values in the
    domain of the trajectory curve T.  At each vi, a vector, Bi, is
    computed, and then an interpolation method is used to obtain the
    function B(v).  More specifically, set B0 to an arbitrary unit
    length vector orthogonal to T'(v0) (or to its projection onto Tp,
    see code below).  Then, for i=1,...,m, compute Ti = T'(vi) /
    |T'(vi)| and set

                    bi = B(i-1) - (B(i-1) dot Ti) Ti

                             Bi = bi / |bi|

    In words, Bi is obtained by projecting B(i-1) onto the plane defined
    by Ti.  Care must be taken to avoid points where Ti || B(i-1).
    Also, the case where T(v) is closed is not supported.

    Parameters
    ----------
    B0 = the first orientation vector
    T = the trajectory Curve
    Tp = the plane, if any, to project the Tis prior projecting the
         B(i-1)s
    Tw = the B-spline twist function, if any, defined with respect
          to the parameter values v of T
    m = the number of points to construct Bv with

    Returns
    -------
    Bv = the orientation function (a nonrational cubic Curve)

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 481.

    '''

    V, = T.U; vb = np.linspace(V[0], V[-1], m)
    Ti = np.zeros((m, 3))
    for i in xrange(m):
        t = T.eval_derivatives(vb[i], 1)[1]
        if Tp is not None:
            t = util.point_to_plane((0,0,0), Tp, t)
        Ti[i] = util.normalize(t)
    B = np.zeros_like(Ti)
    B[0] = util.normalize(B0)
    for i in xrange(1, m):
        Bi = util.point_to_plane((0,0,0), Ti[i], B[i-1])
        B[i] = util.normalize(Bi)
    if Tw:
        for i in xrange(m):
            b = np.append(B[i], 1.0)
            R = Tw.eval_point(vb[i])[0]
            transform.rotate(b, R, L=Ti[i])
            B[i] = b[:3]
    U, Pw = fit.global_curve_interp(m - 1, B, 3, vb)
    cpol = curve.ControlPolygon(Pw=Pw)
    return curve.Curve(cpol, (3,), (U,))


def get_sweep_local_sys(v, T, Bv):

    ''' Get the local coordinate system O, X, Y, Z on the trajectory
    Curve T at parameter v (see make_swept_surface).  O is simply T
    evaluated at v and Y is the tangent direction at O.  Z is computed
    from the orientation function Bv (see get_sweep_orient_func).
    Finally, X is simply the cross product of Y and Z.

    '''

    O, Y = T.eval_derivatives(v, 1)
    Z = Bv.eval_point(v)
    X = np.cross(Y, Z)
    X, Y, Z = [util.normalize(V) for V in X, Y, Z]
    return O, X, Y, Z


def extract_cross_boundary_deriv(S, ei):

    ''' Extract the cross-boundary derivative field lying on the edge ei
    of the Surface S.  Edges 0, 1, 2, 3 respectively correspond to the
    low and high ends of u and v.  Note: this function only supports
    nonrational Surfaces as weights are simply discarded.

    '''

    n, p, U, m, q, V, Pw = S.var()
    P = Pw[...,:-1]
    if ei == 0:
        r1, r2, s1, s2 = 0, 1, 0, m
    elif ei == 1:
        r1, r2, s1, s2 = n - 1, n, 0, m
    elif ei == 2:
        r1, r2, s1, s2 = 0, n, 0, 1
    elif ei == 3:
        r1, r2, s1, s2 = 0, n, m - 1, m
    P = surface_deriv_cpts(n, p, U, m, q, V, P, 1, r1, r2, s1, s2)
    P = P[1,0,-2,:] if ei in (0, 1) else P[0,1,:,-2]
    if ei in (0, 1):
        p, U = q, V
    Pw = nurbs.obj_mat_to_4D(P)
    return curve.Curve(curve.ControlPolygon(Pw=Pw), (p,), (U,))


def blend_cubic_bezier(n, p, U, Cs, Ds, di=0):

    ''' Create a cubically blended Surface that interpolates the
    boundary Curves Cs, as well as the derivatives Ds.

    Source: The NURBS Book (2nd Ed.), Pg. 501.

    '''

    S = np.zeros((n + 1, 4, 3))
    C0, C1 = [c.cobj.Pw[:,:-1] for c in Cs]
    D0, D1 = [d.cobj.Pw[:,:-1] for d in Ds]
    for i in xrange(n + 1):
        S[i,0] = C0[i]
        S[i,1] = C0[i] + D0[i] / 3.0
        S[i,2] = C1[i] - D1[i] / 3.0
        S[i,3] = C1[i]
    Sw = nurbs.obj_mat_to_4D(S)
    s = Surface(ControlNet(Pw=Sw), (p,3), (U,[0,0,0,0,1,1,1,1]))
    return s if di == 0 else s.swap()


def blend_bicubic_bezier(Ck, Cl, Dk, Dl):

    ''' Create a bicubic tensor-product Surface to the four boundaries
    Curves C_(k=0)(u), C_(k=1)(u), C_(l=0)(v), and C_(l=1)(v) and to the
    four twists vectors T00, T10, T01, and T11.

    Source: The NURBS Book (2nd Ed.), Pg. 502.

    '''

    T = np.zeros((4, 4, 3))
    T00 = Dk[0].eval_derivatives(0, 1)[1]
    T10 = Dk[0].eval_derivatives(1, 1)[1]
    T01 = Dk[1].eval_derivatives(0, 1)[1]
    T11 = Dk[1].eval_derivatives(1, 1)[1]
    T[0,0] = Ck[0].eval_point(0)
    T[1,0] = T[0,0] + Dl[0].eval_point(0) / 3.0
    T[0,1] = T[0,0] + Dk[0].eval_point(0) / 3.0
    T[1,1] = T[1,0] + T[0,1] - T[0,0] + T00 / 9.0
    T[3,0] = Ck[0].eval_point(1)
    T[2,0] = T[3,0] - Dl[1].eval_point(0) / 3.0
    T[3,1] = T[3,0] + Dk[0].eval_point(1) / 3.0
    T[2,1] = T[2,0] + T[3,1] - T[3,0] - T10 / 9.0
    T[0,3] = Ck[1].eval_point(0)
    T[1,3] = T[0,3] + Dl[0].eval_point(1) / 3.0
    T[0,2] = T[0,3] - Dk[1].eval_point(0) / 3.0
    T[1,2] = T[1,3] + T[0,2] - T[0,3] - T01 / 9.0
    T[3,3] = Ck[1].eval_point(1)
    T[2,3] = T[3,3] - Dl[1].eval_point(1) / 3.0
    T[3,2] = T[3,3] - Dk[1].eval_point(1) / 3.0
    T[2,2] = T[2,3] + T[3,2] - T[3,3] + T11 / 9.0
    Tw = nurbs.obj_mat_to_4D(T)
    return Surface(ControlNet(Pw=Tw), (3,3))


def make_boundary_compatible(Ck, Cl, Dk, Dl):

    ''' One of the requirements to construct bicubically blended Coons
    patches is that the magnitudes of the cross-boundary derivatives
    along the opposite boundary curves must agree with those of the
    derivatives of these boundaries, e.g.

                  | D_(k=0)(u=0) | = | C'_(l=0)(v=0) |
                  | D_(k=0)(u=1) | = | C'_(l=1)(v=0) |

    In practice, this is almost never the case.  To bring these
    magnitudes in compliance, the following precedure is applied:

                   | C'_(l=0)(v=0) |        | C'_(l=1)(v=0) |
    1. Get    f0 = -----------------,  f1 = -----------------
                   | D_(k=0)(u=0) |         | D_(k=0)(u=0) |

    2. Using symbolic operators, compute a new cross-derivative

     Dh_(k=0)(u) = f(u) * D_(k=0)(u), f(u) = (1 - u) * f0 + u * f1

    This will adjust the magnitudes at the ends while maintaining the
    cross-derivative vector directions along the boundary C_(k=0)(u).
    The unfortunate price one has to pay for this is the increase of the
    degree by one, due to the multiplication by a linear function.
    Similar results are applicable across the other three boundaries.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    def get_f(D, C0, C1, i):
        C0 = util.norm(C0.eval_derivatives(i, 1)[1])
        C1 = util.norm(C1.eval_derivatives(i, 1)[1])
        D0 = util.norm(D.eval_point(0))
        D1 = util.norm(D.eval_point(1))
        f0, f1 = C0 / D0, C1 / D1
        return curve.make_linear_curve(point.Point(f0), point.Point(f1))

    Dk = [curve.mult_func_curve(D, get_f(D, C0, C1, i))
          for D, C0, C1, i in zip(Dk, 2 * (Cl[0],), 2 * (Cl[1],), (0, 1))]
    Dl = [curve.mult_func_curve(D, get_f(D, C0, C1, i))
          for D, C0, C1, i in zip(Dl, 2 * (Ck[0],), 2 * (Ck[1],), (0, 1))]
    return Dk, Dl


def make_twist_compatible(Dk, Dl, eps):

    ''' A famous flaw of the Coons formulation is that the method works
    only if the cross-derivatives are so-called twist compatible.  That
    is, at each corner of a patch, the derivatives of the corresponding
    two cross-boundary derivatives must be the same

                     D'_(k=0)(u=0) = D'_(l=0)(v=0)
                     D'_(k=0)(u=1) = D'_(l=1)(v=0)
                     D'_(k=1)(u=0) = D'_(l=0)(v=1)
                     D'_(k=1)(u=1) = D'_(l=1)(v=1)

    In pratice this is almost never the case.  In order to bring these
    into compliance, here an approximation method with a manageable
    tolerance is chosen.  It consists of (1) averaging the twists at
    each corner and (2) reapproximating each derivative curve with the
    end derivatives, i.e. the average of twists, specified.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    T00 = np.average((Dk[0].eval_derivatives(0, 1)[1],
                      Dl[0].eval_derivatives(0, 1)[1]), axis=0)
    T10 = np.average((Dk[0].eval_derivatives(1, 1)[1],
                      Dl[1].eval_derivatives(0, 1)[1]), axis=0)
    T01 = np.average((Dk[1].eval_derivatives(0, 1)[1],
                      Dl[0].eval_derivatives(1, 1)[1]), axis=0)
    T11 = np.average((Dk[1].eval_derivatives(1, 1)[1],
                      Dl[1].eval_derivatives(1, 1)[1]), axis=0)
    DTTks = zip(Dk, (T00, T01), (T10, T11))
    DTTls = zip(Dl, (T00, T10), (T01, T11))
    Dk = [reapprox_curve_end_ders(eps, *DTT) for DTT in DTTks]
    Dl = [reapprox_curve_end_ders(eps, *DTT) for DTT in DTTls]
    return Dk, Dl


def reapprox_curve_end_ders(eps, D, Ts=None, Te=None):

    ''' Reapproximate a NURBS Curve D(u) with two end derivatives Ts and
    Te specified, up to the user-specified angular tolerance eps.  This
    reapproximation makes use of knot insertion.  Note that if there is
    compatibility at one or both ends no new knot will be computed.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    n, p, U, Vw = D.var()
    V = nurbs.obj_mat_to_3D(Vw)
    D0, Dp0 = D.eval_derivatives(0, 1)
    D1, Dp1 = D.eval_derivatives(1, 1)
    alfs = alfe = - np.inf
    if Ts is not None:
        Dus = U[p+1]
        Vt1 = D0 + Dus / p * Ts
        alfs = util.angle(V[1], Vt1)
    if Te is not None:
        Due = 1.0 - U[n]
        Vtn1 = D1 - Due / p * Te
        alfe = util.angle(V[n-1], Vtn1)
    if alfs < eps and alfe < eps:
        return D
    epsr = np.deg2rad(eps)
    r = []
    if alfs >= eps:
        num = np.sin(epsr) * p * util.norm(D0)
        den = util.norm(Ts - Dp0) + np.sin(epsr) * util.norm(Dp0)
        r.append(num / den)
    if alfe >= eps:
        num = np.sin(epsr) * p * util.norm(D1)
        den = util.norm(Te - Dp1) + np.sin(epsr) * util.norm(Dp1)
        r.append(1 - num / den)
    D = D.refine(sorted(r)); U, = D.U
    assert len(U) == n + len(r) + p + 2
    n += len(r)
    if alfs >= eps:
        Dus = U[p+1]
        Vt1 = D.eval_point(0) + Dus / p * Ts
        D.cobj.Pw[1,:-1] = Vt1
    if alfe >= eps:
        Due = 1.0 - U[n]
        Vtn1 = D.eval_point(1) - Due / p * Te
        D.cobj.Pw[n-1,:-1] = Vtn1
    return D


def split_nsided_region(Ck, Dk):

    ''' Split (N + 1) compatible boundary Curves and cross-derivatives
    at their parametric midpoints.  Also return the split points Mk as
    well as the derivatives DMk at those points.  A few requirements
    must be satisfied by Ck and Dk in order for this to work, see
    docstring of make_nsided_region for details.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    N = len(Ck) - 1
    Mk = np.zeros((N + 1, 3))
    DMk = np.zeros((N + 1, 3))
    AL = np.zeros(N + 1)
    CLRk, DLRk = [], []
    for k in xrange(N + 1):
        ck, dk = Ck[k], Dk[k]
        U, = ck.U
        AL[k] = curve.param_to_arc_length(ck)
        um = (U[-1] + U[0]) / 2.0
        Mk[k] = ck.eval_point(um)
        DMk[k] = dk.eval_point(um)
        cl, cr = ck.split(um)
        dl, dr = dk.split(um)
        cr, dr = cr.reverse(), dr.reverse()
        for c in (cl, cr, dl, dr):
            U, = c.U; knot.normalize_knot_vec(U)
        CLRk.append((cl, cr))
        DLRk.append((dl, dr))
    for k in xrange(N + 1):
        sf = (AL[np.mod(k - 1, N + 1)] +
              AL[np.mod(k + 1, N + 1)]) / 4.0
        DMk[k] = sf * util.normalize(DMk[k])
    return CLRk, DLRk, Mk, DMk


def find_central_point_nsided_region(Mk, DMk):

    ''' Find an initial central point C based on the midpoints Mk and
    corresponding derivatives DMk.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    N = len(Mk) - 1
    Qk = Mk + DMk / 2.0
    return np.sum(Qk, axis=0) / (N + 1)


def find_normal_vec_nsided_region(C, Mk, DMk, CLRk0=None):

    ''' Find an initial normal vector CN based on the central point C,
    midpoints Mk and corresponding derivatives DMk.  Special measures
    are taken if N = 1.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    N = len(Mk) - 1
    if CLRk0:
        cl0, cr0 = CLRk0
        Mk1, Mk3 = Mk
        Mk = np.zeros((4, 3))
        Mk[0], Mk[2] = cl0.eval_point(0), cr0.eval_point(0)
        Mk[1], Mk[3] = Mk1, Mk3
    v0 = Mk - C
    v1 = np.roll(Mk, -1, axis=0) - C
    Wk = np.cross(v0, v1)
    return np.sum(Wk, axis=0) / (N + 1)


def generate_inner_curves_nsided_region(C, CN, Mk, DMk):

    ''' The shape of the inner curves has a definite effect on the shape
    of the N-sided patch.  Even though there is no magic formula, here
    cubic Bezier curves with start and end tangents Ts and Te are
    utilized.  While Ts is simply DMk at any particular point, Te is
    deduced from the central point, C, the central normal vector, CN,
    and the split midpoints, Mk.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    N = len(Mk) - 1
    P = np.zeros((4, 3))
    CIk = []
    for k in xrange(N + 1):
        P[0], P[3] = Mk[k], C
        Ts = DMk[k]
        Te = C - util.point_to_plane(C, CN, Mk[k])
        Ts, Te = [util.normalize(T) for T in Ts, Te]
        a = 16.0 - util.norm(Ts + Te)**2
        b = 12.0 * (P[3] - P[0]).dot(Ts + Te)
        c = - 36.0 * util.norm(P[3] - P[0])**2
        alf = np.roots((a, b, c))
        dummy, alf = np.sort(alf)
        P[1] = P[0] + alf * Ts / 3.0
        P[2] = P[3] - alf * Te / 3.0
        Pw = nurbs.obj_mat_to_4D(P)
        cpol = curve.ControlPolygon(Pw=Pw)
        CI = curve.Curve(cpol, (3,))
        CIk.append(CI)
    return CIk


def generate_inner_cross_deriv_nsided_region(CN, CIk, CLRk):

    ''' To ensure smooth joining of surface patches along the inner
    Curves CIk, cross-boundary derivatives across these Curves must be
    computed, one to the left, BLk, and one to the right, BRk.  These
    derivatives must define the same tangent plane at any point along
    CIk.

    Source: Piegl & Tiller, Filling n-sided regions with NURBS patches,
            The Visual Computer, 1999.

    '''

    def get_coef(D, T, B):
        a, b = np.column_stack((D, T)), B
        pqs, l2n = lstsq(a, b)[:2]
        if l2n > 1e-6:
            print('nurbs.surface.generate_inner_cross_deriv_nsided_region :: '
                  'overdetermined system has no solution ({})'.format(l2n))
        return pqs

    def generate_cross_deriv(Bs, Be):
        ps, qs = get_coef(Ds, Ts, Bs)
        pe, qe = get_coef(De, Te, Be)

        pt = curve.make_linear_curve(point.Point(ps), point.Point(pe))
        qt = curve.make_linear_curve(point.Point(qs), point.Point(qe))

        n, p, U, Pw = ci.var()
        P = curve.curve_deriv_cpts(n, p, U, Pw[:,:-1], 1, 0, n)[1][:-1]
        Pw = nurbs.obj_mat_to_4D(P)
        DIk = curve.Curve(curve.ControlPolygon(Pw=Pw), (2,))

        pDIk = curve.mult_func_curve(DIk, pt)
        qT = curve.mult_func_curve(T, qt)

        dummy, p, U, cc = curve.make_curves_compatible1([pDIk, qT])
        P = cc[0].cobj.Pw[:,:-1] + cc[1].cobj.Pw[:,:-1]
        Pw = nurbs.obj_mat_to_4D(P)
        return curve.Curve(curve.ControlPolygon(Pw=Pw), (p,), (U,))

    N = len(CIk) - 1
    BLRk = []
    for k in xrange(N + 1):
        cl, cr = CLRk[k]
        ci, cim1, cip1 = (CIk[k],
                          CIk[np.mod(k - 1, N + 1)],
                          CIk[np.mod(k + 1, N + 1)])

        Bsr = cl.eval_derivatives(1, 1)[1]
        Bsl = cr.eval_derivatives(1, 1)[1]

        Ds = ci.eval_derivatives(0, 1)[1]
        De = ci.eval_derivatives(1, 1)[1]

        Ber = cim1.eval_derivatives(1, 1)[1]
        Bel = cip1.eval_derivatives(1, 1)[1]

        Nk = np.cross(Bsr, Ds)

        Ts, Te = np.cross(Nk, Ds), np.cross(CN, De)
        T = curve.make_linear_curve(point.Point(*Ts), point.Point(*Te))

        BLk = generate_cross_deriv(Bsl, Bel)
        BRk = generate_cross_deriv(Bsr, Ber)

        BLRk.append((BLk, BRk))
    return BLRk


def check_coons_compatibility(Ck, Cl, Dk, Dl):

    def check(i, a, b):
        na, nb = util.norm(a), util.norm(b)
        if not np.allclose(na, nb, atol=1e-3):
            l2n = util.norm(na - nb)
            t = 'twist' if i in (10, 11, 12, 13) else 'boundary'
            print('nurbs.surface.check_coons_compatibility :: '
                  '{} incompatible ({})'.format(t, l2n))

    Ck0, Ck1 = Ck; Dk0, Dk1 = Dk
    Cl0, Cl1 = Cl; Dl0, Dl1 = Dl

    # boundary
    check(1, Dk0.eval_point(0), Cl0.eval_derivatives(0, 1)[1])
    check(2, Dk0.eval_point(1), Cl1.eval_derivatives(0, 1)[1])
    check(3, Dk1.eval_point(0), Cl0.eval_derivatives(1, 1)[1])
    check(4, Dk1.eval_point(1), Cl1.eval_derivatives(1, 1)[1])
    check(5, Dl0.eval_point(0), Ck0.eval_derivatives(0, 1)[1])
    check(6, Dl0.eval_point(1), Ck1.eval_derivatives(0, 1)[1])
    check(7, Dl1.eval_point(0), Ck0.eval_derivatives(1, 1)[1])
    check(8, Dl1.eval_point(1), Ck1.eval_derivatives(1, 1)[1])

    # twist
    check(10, Dk0.eval_derivatives(0, 1)[1], Dl0.eval_derivatives(0, 1)[1])
    check(11, Dk0.eval_derivatives(1, 1)[1], Dl1.eval_derivatives(0, 1)[1])
    check(12, Dk1.eval_derivatives(0, 1)[1], Dl0.eval_derivatives(1, 1)[1])
    check(13, Dk1.eval_derivatives(1, 1)[1], Dl1.eval_derivatives(1, 1)[1])


# EXCEPTIONS


class SurfaceException(nurbs.NURBSException):
    pass

class ImproperInput(SurfaceException):
    pass

class NoInitialStepDirectionCouldBeFound(SurfaceException):
    pass

class NoCommonEdgeCouldBeFound(SurfaceException):
    pass

class SurfaceSurfaceLikelyDiverged(SurfaceException):
    pass
