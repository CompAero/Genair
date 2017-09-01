import numpy as np
from scipy.misc import comb

import basis
import curve
import knot
import nurbs
import surface
import transform
import util

import plot.pobject


__all__ = ['Volume', 'ControlVolume',
           'make_composite_volume',
           'make_ruled_volume',
           'make_ruled_volume2',
           'make_trilinear_interp_volume',
           'make_trilinear_volume']


class ControlVolume(nurbs.ControlObject):

    def __init__(self, cpts=None, Pw=None):

        ''' See nurbs.nurbs.ControlObject.

        Parameters
        ----------
        cpts = the list of list of list of (control) Points
        Pw = the object matrix

        '''

        super(ControlVolume, self).__init__(cpts, Pw)


class Volume(nurbs.NURBSObject, plot.pobject.PlotVolume):

    def __init__(self, cvol, p, U=None):

        ''' See nurbs.nurbs.NURBSObject.

        Parameters
        ----------
        cvol = the ControlVolume
        p = the u, v, w degrees of the Volume
        U = the u, v, w knot vectors

        Examples
        --------

             u |
               |
            P4 .___________. P6
              /|          /|
             / |         / |
         P5 .___________.P7|
            |  |        |  |
            |P0.________|__._P1__ w
            |  /        |  /
            | /         | /
            |/          |/
         P2 .___________. P3
           /
        v /

        >>> P0 = Point( 1,-1,-1)
        >>> P1 = Point(-1,-1,-1)
        >>> P2 = Point( 1, 1,-1)
        >>> P3 = Point(-1, 1,-1)
        >>> P4 = Point( 1,-1, 1)
        >>> P5 = Point(-1,-1, 1)
        >>> P6 = Point( 1, 1, 1)
        >>> P7 = Point(-1, 1, 1)
        >>> cvol = ControlVolume([[[P0, P1], [P2, P3]],
        ...                       [[P4, P5], [P6, P7]]])
        >>> v = Volume(cvol, (1,1,1))

        '''

        n, m, l = cvol.n
        p, q, r = p
        if n < p or m < q or l < r:
            raise nurbs.TooFewControlPoints((n, p), (m, q), (l, r))
        self._cobj = cvol.copy()
        self._p = p, q, r
        if not U:
            U = (knot.uni_knot_vec(n, p),
                 knot.uni_knot_vec(m, q),
                 knot.uni_knot_vec(l, r))
        self.U = U
        super(Volume, self).__init__()

    def __setstate__(self, d):
        ''' Unpickling. '''
        self.__dict__.update(d)
        super(Volume, self).__init__()

# EVALUATION OF POINTS AND DERIVATIVES

    def eval_point(self, u, v, w):

        ''' Evaluate a point.

        Parameters
        ----------
        u, v, w = the parameter values of the point

        Returns
        -------
        V = the xyz coordinates of the point

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        return rat_volume_point(n, p, U, m, q, V, l, r, W, Pw, u, v, w)

    def eval_points(self, us, vs, ws):

        ''' Evaluate multiple points.

        Parameters
        ----------
        us, vs, ws = the parameter values of each point

        Returns
        -------
        V = the xyz coordinates of all points

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        return rat_volume_point_v(n, p, U, m, q, V, l, r, W,
                                  Pw, us, vs, ws, len(us))

    def eval_derivatives(self, u, v, w, d):

        ''' Evaluate derivatives at a point.

        Parameters
        ----------
        u, v, w = the parameter values of the point
        d = the number of derivatives to evaluate

        Returns
        -------
        VABC = all derivatives, where VABC[a,b,c,:] is the derivative of
               V(u,v,w) with respect to u a times, v b times and w c
               times (0 <= a + b + c <= d)

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        Aders = volume_derivs_alg1(n, p, U, m, q, V, l, r, W,
                                   Pw[:,:,:,:-1], u, v, w, d)
        if self.isrational:
            wders = volume_derivs_alg1(n, p, U, m, q, V, l, r, W,
                                       Pw[:,:,:,-1], u, v, w, d)
            return rat_volume_derivs(Aders, wders, d)
        return Aders

# KNOT INSERTION

    def split(self, u, di):

        ''' Split the Volume in one direction.

        Parameters
        ----------
        u = the parameter value at which to split the Volume
        di = the parametric direction in which to split the Volume (0, 1
             or 2)

        Returns
        -------
        [Volume, Volume] = the two split Volumes

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        u = knot.clean_knot(u)
        if di == 0:
            if u == U[0] or u == U[-1]:
                return [self.copy()]
            k, s = basis.find_span_mult(n, p, U, u)
            rr = p - s
            if rr > 0:
                U, V, W, Pw = volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                              Pw, u, k, s, rr, 0)
            Ulr = (np.append(U[:k+rr+1], u), np.insert(U[k-s+1:], 0, u))
            Vlr = V, V
            Wlr = W, W
            Pwlr = Pw[:k-s+1,:,:], Pw[k-s:,:,:]
        elif di == 1:
            if u == V[0] or u == V[-1]:
                return [self.copy()]
            k, s = basis.find_span_mult(m, q, V, u)
            rr = q - s
            if rr > 0:
                U, V, W, Pw = volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                              Pw, u, k, s, rr, 1)
            Ulr = U, U
            Vlr = (np.append(V[:k+rr+1], u), np.insert(V[k-s+1:], 0, u))
            Wlr = W, W
            Pwlr = Pw[:,:k-s+1,:], Pw[:,k-s:,:]
        elif di == 2:
            if u == W[0] or u == W[-1]:
                return [self.copy()]
            k, s = basis.find_span_mult(l, r, W, u)
            rr = r - s
            if rr > 0:
                U, V, W, Pw = volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                              Pw, u, k, s, rr, 2)
            Ulr = U, U
            Vlr = V, V
            Wlr = (np.append(W[:k+rr+1], u), np.insert(W[k-s+1:], 0, u))
            Pwlr = Pw[:,:,:k-s+1], Pw[:,:,k-s:]
        return [self.__class__(ControlVolume(Pw=Pw), (p,q,r), (U,V,W))
                for Pw, U, V, W in zip(Pwlr, Ulr, Vlr, Wlr)]

    def extract(self, u, di):

        ''' Extract an isoparametric Surface from the Volume.

        Parameters
        ----------
        u = the parameter value at which to extract the Surface
        di = the parametric direction in which to extract the Surface
             (0, 1 or 2)

        Returns
        -------
        Surface = the extracted Surface

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        u = knot.clean_knot(u)
        if di == 0:
            if u == U[0]:
                Pwc = Pw[0,:,:]
            elif u == U[-1]:
                Pwc = Pw[-1,:,:]
            else:
                k, s = basis.find_span_mult(n, p, U, u)
                rr = p - s
                if rr > 0:
                    U, V, W, Pw = \
                            volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                            Pw, u, k, s, rr, 0)
                Pwc = Pw[k-s,:,:]
            p, q = q, r
            U, V = V, W
        elif di == 1:
            if u == V[0]:
                Pwc = Pw[:,0,:]
            elif u == V[-1]:
                Pwc = Pw[:,-1,:]
            else:
                k, s = basis.find_span_mult(m, q, V, u)
                rr = q - s
                if rr > 0:
                    U, V, W, Pw = \
                            volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                            Pw, u, k, s, rr, 1)
                Pwc = Pw[:,k-s,:]
            Pwc = np.transpose(Pwc, (1, 0, 2))
            p, q = r, p
            U, V = W, U
        elif di == 2:
            if u == W[0]:
                Pwc = Pw[:,:,0]
            elif u == W[-1]:
                Pwc = Pw[:,:,-1]
            else:
                k, s = basis.find_span_mult(l, r, W, u)
                rr = r - s
                if rr > 0:
                    U, V, W, Pw = \
                            volume_knot_ins(n, p, U, m, q, V, l, r, W,
                                            Pw, u, k, s, rr, 2)
                Pwc = Pw[:,:,k-s]
        return surface.Surface(surface.ControlNet(Pw=Pwc), (p,q), (U,V))

    def extend(self, el, di, end=False):

        ''' Extend the Volume.

        Tangent-plane and curvature continuities are only guaranteed if
        the Volume is nonrational (all weights equal 1.0).

        Parameters
        ----------
        el = the (estimated) length of the extension
        di = the parametric direction in which to extend the Volume (0,
             1 or 2)
        end = whether to extend the start or the end part of the Volume

        Returns
        -------
        Volume = the Volume extension

        '''

        if di == 1:
            Vext = self.swap('uv').extend(el, 0, end)
            return Vext.swap('uv') if Vext else None
        elif di == 2:
            Vext = self.swap('uw').extend(el, 0, end)
            return Vext.swap('uw') if Vext else None
        if end:
            Vext = self.reverse(0).extend(el, 0)
            return Vext.reverse(0) if Vext else None
        n, p, U, m, q, V, l, r, W, Pw = self.var()
        S = self.extract(W[0], 2)
        C = S.extract(V[0], 1)
        u = curve.arc_length_to_param(C, el)
        Vs = self.split(u, 0)
        if len(Vs) == 1:
            return None
        Vext, dummy = Vs
        Pw, cpts = Vext.cobj.Pw, Vext.cobj.cpts
        for sep in xrange(l + 1):
            for col in xrange(m + 1):
                Q, N = [cpt.xyz for cpt in cpts[:2,col,sep]]
                transform.mirror(Pw[:,col,sep], N - Q, Q)
        U = Vext.U[0]; U -= U[0]
        return Vext.reverse(0)

# KNOT REFINEMENT

    def refine(self, X, di):

        ''' Refine the knot vector in one direction.

        Parameters
        ----------
        X = a list of the knots to insert
        di = the parametric direction in which to refine the Volume (0,
             1 or 2)

        Returns
        -------
        Volume = the refined Volume

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        if len(X) != 0:
            if X == 'mid':
                if di == 0:
                    X = knot.midpoints_knot_vec(U)
                elif di == 1:
                    X = knot.midpoints_knot_vec(V)
                elif di == 2:
                    X = knot.midpoints_knot_vec(W)
            X = knot.clean_knot(X)
            if di == 0:
                CU = U
            elif di == 1:
                CU = V
            elif di == 2:
                CU = W
            U, V, W, Pw = refine_knot_vect_volume(n, p, U, m, q, V, l, r, W,
                                                  Pw, X, di)
        return self.__class__(ControlVolume(Pw=Pw), (p,q,r), (U,V,W))

# DEGREE ELEVATION

    def elevate(self, t, di):

        ''' Elevate the Volume's degree in one direction.

        Parameters
        ----------
        t = the number of degrees to elevate the Volume with
        di = the parametric direction in which to degree elevate the
             Volume (0, 1 or 2)

        Returns
        -------
        Volume = the degree elevated Volume

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        if t > 0:
            U, V, W, Pw = \
                    degree_elevate_volume(n, p, U, m, q, V, l, r, W, Pw, t, di)
            if di == 0:
                p += t
            elif di == 1:
                q += t
            elif di == 2:
                r += t
        return self.__class__(ControlVolume(Pw=Pw), (p,q,r), (U,V,W))

# MISCELLANEA

    def project(self, xyz, uvwi=None):

        ''' Project a point.

        Parameters
        ----------
        xyz = the xyz coordinates of a point to project
        uvwi = the initial guess for Newton's method

        Returns
        -------
        u, v, w = the parameter values of the projected point

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        return volume_point_projection(n, p, U, m, q, V, l, r, W,
                                       Pw, xyz, uvwi)

    def reverse(self, di):

        ''' Reverse (flip) the Volume's direction.

        Parameters
        ----------
        di = the reversal direction (0, 1 or 2)

        Returns
        -------
        Volume = the reversed Volume

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        U, V, W, Pw = \
                reverse_volume_direction(n, p, U, m, q, V, l, r, W, Pw, di)
        return self.__class__(ControlVolume(Pw=Pw), (p,q,r), (U,V,W))

    def swap(self, uv):

        ''' Swap directions.

        Parameters
        ----------
        uv = the directions to swap ('uv', 'uw' or 'vw')

        Returns
        -------
        Volume = the swapped Volume

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        if uv == 'uv':
            axes, pqr, UVW = (1, 0, 2, 3), (q, p, r), (V, U, W)
        elif uv == 'uw':
            axes, pqr, UVW = (2, 1, 0, 3), (r, q, p), (W, V, U)
        elif uv == 'vw':
            axes, pqr, UVW = (0, 2, 1, 3), (p, r, q), (U, W, V)
        Pw = np.transpose(Pw, axes)
        return self.__class__(ControlVolume(Pw=Pw), pqr, UVW)

    def check_injectivity(self, num=50, show=False):

        ''' Check if the Volume is injective.

        Let F be a spatial deformation function and J be the Jacobian
        matrix of F.  F is a homeomorphism (injective, onto, and
        invertible) if and only if:

        - F has continuous first partial derivatives, and;
        - det(J) > 0.

        Parameters
        ----------
        num**3 = the number of points where to conduct the injectivity
                 test
        show = whether or not to show the points where the test failed

        Returns
        -------
        npjacs = the number of non-positive Jacobians found

        Source
        ------
        Gain and Dodgson, Preventing Self-Intersection under Free-Form
        Deformation, IEEE Transactions on Visualization and Computer
        Graphics, 2001.

        '''

        n, p, U, m, q, V, l, r, W, Pw = self.var()
        u, v, w = util.construct_flat_grid((U, V, W), 3 * (num,))
        ders1 = volume_derivs_alg1_v(n, p, U, m, q, V, l, r, W,
                                     Pw[...,:-1], u, v, w, 1, num**3)
        ws = Pw[...,-1]
        if (ws != 1.0).any():
            wders1 = volume_derivs_alg1_v(n, p, U, m, q, V, l, r, W,
                                          ws, u, v, w, 1, num**3)
            ders1 = rat_volume_derivs_v(ders1, wders1, 1, num**3)
        a, d, g = ders1[1,0,0,:]
        b, e, h = ders1[0,1,0,:]
        c, f, i = ders1[0,0,1,:]
        jacs = ((a * e * i) + (b * f * g) + (c * d * h) -
                (c * e * g) - (b * d * i) - (a * f * h))
        npjac = (jacs <= 0.0)
        npjacs = npjac.tolist().count(True)
        if npjacs and show:
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x, y, z = u[npjac], v[npjac], w[npjac]
            c = jacs[npjac]
            ax.scatter(x, y, z, c=c)
            ax.set_xlabel('u'); ax.set_xlim3d(U[0], U[-1])
            ax.set_ylabel('v'); ax.set_ylim3d(V[0], V[-1])
            ax.set_zlabel('w'); ax.set_zlim3d(W[0], W[-1])
            plt.show()
        return npjacs


# HEAVY LIFTING FUNCTIONS


def volume_derivs_alg1(n, p, U, m, q, V, l, r, W, P, u, v, w, d):

    ''' Compute a point in a B-spline volume and all partial derivatives
    up to and including order d (0 <= a + b + c <= d), (d > p, q, r) is
    allowed, although the derivatives are 0 in this case (for
    nonrational Volumes); these derivatives are necessary for rational
    Volumes.  Output is the array VABC, where VABC[a,b,c,:] is the
    derivative of V(u,v,w) with respect to u a times, v b times, and w c
    times.

    '''

    VABC = np.zeros((d + 1, d + 1, d + 1, 3))
    tmp1 = np.zeros((q + 1, 3))
    tmp2 = np.zeros((r + 1, 3))
    du, dv, dw = min(d, p), min(d, q), min(d, r)
    uspan = basis.find_span(n, p, U, u)
    Nu = basis.ders_basis_funs(uspan, u, p, du, U)
    vspan = basis.find_span(m, q, V, v)
    Nv = basis.ders_basis_funs(vspan, v, q, dv, V)
    wspan = basis.find_span(l, r, W, w)
    Nw = basis.ders_basis_funs(wspan, w, r, dw, W)
    for a in xrange(du + 1):
        ddb = min(d - a, dv)
        for b in xrange(ddb + 1):
            tmp2[:] = 0.0
            for z in xrange(r + 1):
                tmp1[:] = 0.0
                for y in xrange(q + 1):
                    for x in xrange(p + 1):
                        tmp1[y] += (Nu[a,x] *
                                     P[uspan-p+x,vspan-q+y,wspan-r+z])
                for y in xrange(q + 1):
                    tmp2[z] += Nv[b,y] * tmp1[y]
            ddc = min(d - a - b, dw)
            for c in xrange(ddc + 1):
                for z in xrange(r + 1):
                    VABC[a,b,c] += Nw[c,z] * tmp2[z]
    return VABC


def volume_derivs_alg1_v(n, p, U, m, q, V, l, r, W, P, u, v, w, d, num):

    '''  Idem volume_derivs_alg1, vectorized in u, v, w.

    '''

    VABC = np.zeros((d + 1, d + 1, d + 1, 3, num))
    tmp1 = np.zeros((q + 1, 3, num))
    tmp2 = np.zeros((r + 1, 3, num))
    du, dv, dw = min(d, p), min(d, q), min(d, r)
    uspan = basis.find_span_v(n, p, U, u, num)
    Nu = basis.ders_basis_funs_v(uspan, u, p, du, U, num)
    vspan = basis.find_span_v(m, q, V, v, num)
    Nv = basis.ders_basis_funs_v(vspan, v, q, dv, V, num)
    wspan = basis.find_span_v(l, r, W, w, num)
    Nw = basis.ders_basis_funs_v(wspan, w, r, dw, W, num)
    for a in xrange(du + 1):
        ddb = min(d - a, dv)
        for b in xrange(ddb + 1):
            tmp2[:] = 0.0
            for z in xrange(r + 1):
                tmp1[:] = 0.0
                for y in xrange(q + 1):
                    for x in xrange(p + 1):
                        tmp1[y] += (Nu[a,x] *
                                     P[uspan-p+x,vspan-q+y,wspan-r+z].T)
                for y in xrange(q + 1):
                    tmp2[z] += Nv[b,y] * tmp1[y]
            ddc = min(d - a - b, dw)
            for c in xrange(ddc + 1):
                for z in xrange(r + 1):
                    VABC[a,b,c] += Nw[c,z] * tmp2[z]
    return VABC


def rat_volume_point(n, p, U, m, q, V, l, r, W, Pw, u, v, w):

    ''' Compute a point on a rational B-spline volume at fixed u, v and
    w parameter values.

    '''

    Sw = np.zeros(4)
    tmp1 = np.zeros((q + 1, 4))
    tmp2 = np.zeros((r + 1, 4))
    uspan = basis.find_span(n, p, U, u)
    Nu = basis.basis_funs(uspan, u, p, U)
    vspan = basis.find_span(m, q, V, v)
    Nv = basis.basis_funs(vspan, v, q, V)
    wspan = basis.find_span(l, r, W, w)
    Nw = basis.basis_funs(wspan, w, r, W)
    for j in xrange(r + 1):
        tmp1[:] = 0.0
        for l in xrange(q + 1):
            for k in xrange(p + 1):
                tmp1[l] += (Nu[k] *
                             Pw[uspan-p+k,vspan-q+l,wspan-r+j])
        for l in xrange(q + 1):
            tmp2[j] += Nv[l] * tmp1[l]
    for j in xrange(r + 1):
        Sw += Nw[j] * tmp2[j]
    return Sw[:3] / Sw[-1]


def rat_volume_point_v(n, p, U, m, q, V, l, r, W, Pw, u, v, w, num):

    ''' Idem rat_volume_point, vectorized in u, v, w.

    '''

    u, v, w = [np.asfarray(u) for u in u, v, w]
    Sw = np.zeros((4, num))
    tmp1 = np.zeros((q + 1, 4, num))
    tmp2 = np.zeros((r + 1, 4, num))
    uspan = basis.find_span_v(n, p, U, u, num)
    Nu = basis.basis_funs_v(uspan, u, p, U, num)
    vspan = basis.find_span_v(m, q, V, v, num)
    Nv = basis.basis_funs_v(vspan, v, q, V, num)
    wspan = basis.find_span_v(l, r, W, w, num)
    Nw = basis.basis_funs_v(wspan, w, r, W, num)
    for j in xrange(r + 1):
        tmp1[:] = 0.0
        for l in xrange(q + 1):
            for k in xrange(p + 1):
                tmp1[l] += Nu[k] * Pw[uspan-p+k,vspan-q+l,wspan-r+j].T
        for l in xrange(q + 1):
            tmp2[j] += Nv[l] * tmp1[l]
    for j in xrange(r + 1):
        Sw += Nw[j] * tmp2[j]
    return Sw[:3] / Sw[-1]


def rat_volume_derivs(Aders, wders, d):

    ''' Given that (u,v,w) is fixed, and that all derivatives A^(a,b,c),
    w^(a,b,c) for (a,b,c >= 0) and (0 <= a + b + c <= d) have been
    computed and loaded into the arrays Aders and wders, respectively,
    this algorithm computes the point, V(u,v,w) and the derivatives,
    V^(a,b,c)(u,v,w), (0 <= a + b + c <= d).  The volume point is
    returned in VABC[0,0,0,:] and the a,b,cth derivative is returned in
    VABC[a,b,c,:].

    '''

    VABC = np.zeros((d + 1, d + 1, d + 1, 3))
    for a in xrange(d + 1):
        for b in xrange(d - a + 1):
            for c in xrange(d - a - b + 1):
                v = Aders[a,b,c]
                for s in xrange(1, c + 1):
                    v -= comb(c,s) * wders[0,0,s] * VABC[a,b,c-s]
                for j in xrange(1, b + 1):
                    for s in xrange(c + 1):
                        v -= (comb(b,j) * comb(c,s) *
                              wders[0,j,s] * VABC[a,b-j,c-s])
                for i in xrange(1, a + 1):
                    for j in xrange(b + 1):
                        for s in xrange(c + 1):
                            v -= (comb(a,i) * comb(b,j) * comb(c,s) *
                                  wders[i,j,s] * VABC[a-i,b-j,c-s])
                VABC[a,b,c] = v / wders[0,0,0]
    return VABC


def rat_volume_derivs_v(Aders, wders, d, num):

    ''' Idem rat_volume_derivs, vectorized in u, v, w.

    '''

    VABC = np.zeros((d + 1, d + 1, d + 1, 3, num))
    for a in xrange(d + 1):
        for b in xrange(d - a + 1):
            for c in xrange(d - a - b + 1):
                v = Aders[a,b,c]
                for s in xrange(1, c + 1):
                    v -= comb(c,s) * wders[0,0,s] * VABC[a,b,c-s]
                for j in xrange(1, b + 1):
                    for s in xrange(c + 1):
                        v -= (comb(b,j) * comb(c,s) *
                              wders[0,j,s] * VABC[a,b-j,c-s])
                for i in xrange(1, a + 1):
                    for j in xrange(b + 1):
                        for s in xrange(c + 1):
                            v -= (comb(a,i) * comb(b,j) * comb(c,s) *
                                  wders[i,j,s] * VABC[a-i,b-j,c-s])
                VABC[a,b,c] = v / wders[0,0,0]
    return VABC


# FUNDAMENTAL GEOMETRIC ALGORITHMS


def volume_knot_ins(n, p, U, m, q, V, l, r, W, Pw, u, k, s, rr, di):

    if di == 0:
        VQ, WQ = V.copy(), W.copy()
        Qw = np.zeros((n + rr + 1, m + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for col in xrange(m + 1):
                UQ, Qw[:,col,sep] = \
                        curve.curve_knot_ins(n, p, U, Pw[:,col,sep],
                                             u, k, s, rr)
    elif di == 1:
        UQ, WQ = U.copy(), W.copy()
        Qw = np.zeros((n + 1, m + rr + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for row in xrange(n + 1):
                VQ, Qw[row,:,sep] = \
                        curve.curve_knot_ins(m, q, V, Pw[row,:,sep],
                                             u, k, s, rr)
    elif di == 2:
        UQ, VQ = U.copy(), V.copy()
        Qw = np.zeros((n + 1, m + 1, l + rr + 1, 4))
        for col in xrange(m + 1):
            for row in xrange(n + 1):
                WQ, Qw[row,col,:] = \
                        curve.curve_knot_ins(l, r, W, Pw[row,col,:],
                                             u, k, s, rr)
    return UQ, VQ, WQ, Qw


def refine_knot_vect_volume(n, p, U, m, q, V, l, r, W, Pw, X, di):

    rr = len(X) - 1
    if di == 0:
        VQ, WQ = V.copy(), W.copy()
        Qw = np.zeros((n + rr + 2, m + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for col in xrange(m + 1):
                UQ, Qw[:,col,sep] = \
                        curve.refine_knot_vect_curve(n, p, U, Pw[:,col,sep], X)
    elif di == 1:
        UQ, WQ = U.copy(), W.copy()
        Qw = np.zeros((n + 1, m + rr + 2, l + 1, 4))
        for sep in xrange(l + 1):
            for row in xrange(n + 1):
                VQ, Qw[row,:,sep] = \
                        curve.refine_knot_vect_curve(m, q, V, Pw[row,:,sep], X)
    elif di == 2:
        UQ, VQ = U.copy(), V.copy()
        Qw = np.zeros((n + 1, m + 1, l + rr + 2, 4))
        for col in xrange(m + 1):
            for row in xrange(n + 1):
                WQ, Qw[row,col,:] = \
                        curve.refine_knot_vect_curve(l, r, W, Pw[row,col,:], X)
    return UQ, VQ, WQ, Qw


def degree_elevate_volume(n, p, U, m, q, V, l, r, W, Pw, t, di):

    if di == 0:
        VQ, WQ = V.copy(), W.copy()
        nh = curve.mult_degree_elevate(n, p, U, t)[0]
        Qw = np.zeros((nh + 1, m + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for col in xrange(m + 1):
                UQ, Qw[:,col,sep] = \
                        curve.degree_elevate_curve(n, p, U, Pw[:,col,sep], t)
    elif di == 1:
        UQ, WQ = U.copy(), W.copy()
        mh = curve.mult_degree_elevate(m, q, V, t)[0]
        Qw = np.zeros((n + 1, mh + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for row in xrange(n + 1):
                VQ, Qw[row,:,sep] = \
                        curve.degree_elevate_curve(m, q, V, Pw[row,:,sep], t)
    elif di == 2:
        UQ, VQ = U.copy(), V.copy()
        lh = curve.mult_degree_elevate(l, r, W, t)[0]
        Qw = np.zeros((n + 1, m + 1, lh + 1, 4))
        for col in xrange(m + 1):
            for row in xrange(n + 1):
                WQ, Qw[row,col,:] = \
                        curve.degree_elevate_curve(l, r, W, Pw[row,col,:], t)
    return UQ, VQ, WQ, Qw


# ADVANCED GEOMETRIC ALGORITHMS


def volume_point_projection(n, p, U, m, q, V, l, r, W, Pw, Pi,
                            uvwi=None, eps=1e-15):

    ''' Find the parameter values ui, vi, wi for which V(ui,vi,wi) is
    closest to Pi.  This is achieved by solving the three following
    functions simultaneously: f(u,v,w) = Vu(u,v,w) * (V(u,v,w) - Pi) =
    0, g(u,v,w) = Vv(u,v,w) * (V(u,v,w) - Pi) = 0 and h(u,v,w) =
    Vw(u,v,w) * (V(u,v,w) - Pi) = 0.  A measure of Euclidean distance,
    eps, is used to indicate convergence.

    '''

    if uvwi is None:
        num = 50 # knob
        us, vs, ws = util.construct_flat_grid((U, V, W), 3 * (num,))
        Vs = rat_volume_point_v(n, p, U, m, q, V, l, r, W, Pw, us, vs, ws,
                                num**3)
        i = np.argmin(util.distance_v(Vs, Pi))
        ui, vi, wi = us[i], vs[i], ws[i]
    else:
        ui, vi, wi = uvwi
    P, w = Pw[:,:,:,:-1], Pw[:,:,:,-1]
    rat = (w != 1.0).any()
    J = np.zeros((3, 3))
    K = np.zeros(3)
    for ni in xrange(20): # knob
        VABC = volume_derivs_alg1(n, p, U, m, q, V, l, r, W, P, ui, vi, wi, 2)
        if rat:
            wders = volume_derivs_alg1(n, p, U, m, q, V, l, r, W, w,
                                       ui, vi, wi, 2)
            VABC = rat_volume_derivs(VABC, wders, 2)
        PP = VABC[0,0,0]
        VU, VUU = VABC[1,0,0], VABC[2,0,0]
        VV, VVV = VABC[0,1,0], VABC[0,2,0]
        VW, VWW = VABC[0,0,1], VABC[0,0,2]
        VUV = VVU = VABC[1,1,0]
        VUW = VWU = VABC[1,0,1]
        VVW = VWV = VABC[0,1,1]
        R = PP - Pi
        if util.norm(R) <= eps:
            return ui, vi, wi
        VUVR = np.dot(VUV, R)
        VUWR = np.dot(VUW, R)
        VVWR = np.dot(VVW, R)
        VUV = np.dot(VU, VV)
        VUW = np.dot(VU, VW)
        VVW = np.dot(VV, VW)
        J[:,:] = [[util.norm(VU)**2 + np.dot(R, VUU), VUV + VUVR, VUW + VUWR],
                  [VUV + VUVR, util.norm(VV)**2 + np.dot(R, VVV), VVW + VVWR],
                  [VUW + VUWR, VVW + VVWR, util.norm(VW)**2 + np.dot(R, VWW)]]
        K[:] = [np.dot(VU, R), np.dot(VV, R), np.dot(VW, R)]
        d0, d1, d2 = np.linalg.solve(J, - K)
        uii, vii, wii = d0 + ui, d1 + vi, d2 + wi
        if uii < U[0]:
            uii = U[0]
        elif uii > U[-1]:
            uii = U[-1]
        if vii < V[0]:
            vii = V[0]
        elif vii > V[-1]:
            vii = V[-1]
        if wii < W[0]:
            wii = W[0]
        elif wii > W[-1]:
            wii = W[-1]
        if util.norm((uii - ui) * VU +
                     (vii - vi) * VV +
                     (wii - wi) * VW) <= eps:
            return uii, vii, wii
        ui, vi, wi = uii, vii, wii
    raise nurbs.NewtonLikelyDiverged(ui, vi, wi)


def reverse_volume_direction(n, p, U, m, q, V, l, r, W, Pw, di):

    if di == 0:
        S, T = V.copy(), W.copy()
        Qw = np.zeros((n + 1, m + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for col in xrange(m + 1):
                R, Qw[:,col,sep] = \
                        curve.reverse_curve_direction(n, p, U, Pw[:,col,sep])
    elif di == 1:
        R, T = U.copy(), W.copy()
        Qw = np.zeros((n + 1, m + 1, l + 1, 4))
        for sep in xrange(l + 1):
            for row in xrange(n + 1):
                S, Qw[row,:,sep] = \
                        curve.reverse_curve_direction(m, q, V, Pw[row,:,sep])
    elif di == 2:
        R, S = U.copy(), V.copy()
        Qw = np.zeros((n + 1, m + 1, l + 1, 4))
        for col in xrange(m + 1):
            for row in xrange(n + 1):
                T, Qw[row,col,:] = \
                        curve.reverse_curve_direction(l, r, W, Pw[row,col,:])
    return R, S, T, Qw


# TOOLBOX


def make_trilinear_volume(P000, P001, P010, P011, P100, P101, P110, P111):

    ''' Construct a trilinear Volume.

    Parameters
    ----------
    P000, P001, P010, P011, P100, P101, P110, P111 = the eight corner
                                                     Points

    Returns
    -------
    Volume = the trilinear Volume

    '''

    cvol = ControlVolume([[[P000, P001], [P010, P011]],
                          [[P100, P101], [P110, P111]]])
    return Volume(cvol, (1,1,1))


def make_ruled_volume(s1, s2):

    ''' Construct a ruled Volume.

    Parameters
    ----------
    s1, s2 = the two Surfaces to rule

    Returns
    -------
    Volume = the ruled Volume

    Source
    ------
    Handbook of Grid Generation, Pg. 30-19.

    '''

    n, m, p, q, U, V, Ss = surface.make_surfaces_compatible1([s1, s2])
    s1, s2 = Ss
    Pw = np.zeros((n + 1, m + 1, 2, 4))
    Pw[:,:,0], Pw[:,:,1] = s1.cobj.Pw, s2.cobj.Pw
    return Volume(ControlVolume(Pw=Pw), (p,q,1), (U,V,[0,0,1,1]))


def make_ruled_volume2(s1, s2):

    ''' Construct a ruled Volume (Type 2).

    Parameters
    ----------
    s1, s2 = the two compatible Surfaces to rule

    Returns
    -------
    Volume = the (Type 2) ruled Volume

    '''

    m = s1.cobj.n[1]
    p = s1.p[1]
    V = s1.U[1]
    Pw = np.zeros((2, m + 1, 2, 4))
    Pw[0,:,0] = s1.cobj.Pw[0]
    Pw[1,:,0] = s1.cobj.Pw[-1]
    Pw[0,:,1] = s2.cobj.Pw[0]
    Pw[1,:,1] = s2.cobj.Pw[-1]
    return Volume(ControlVolume(Pw=Pw), (1,p,1), ([0,0,1,1],V,[0,0,1,1]))


def make_trilinear_interp_volume(Sk, Sl, Sm):

    ''' Linearly interpolate a tridirectional Surface network.

    Parameters
    ----------
    Sk, Sl, Sm = the input Surfaces (three 2-tuples)

    Returns
    -------
    Volume = the transfinite trilinear interpolation Volume

    Source
    ------
    Handbook of Grid Generation, Pg. 30-23.

    '''

    if len(Sk) != 2 or len(Sk) != 2 or len(Sm) != 2:
        raise ImproperInput()
    Sk = surface.make_surfaces_compatible1(Sk); Sk = Sk[-1]
    Sl = surface.make_surfaces_compatible1(Sl); Sl = Sl[-1]
    Sm = surface.make_surfaces_compatible1(Sm); Sm = Sm[-1]
    L1 = make_ruled_volume(*Sk); L1 = L1.swap('uw').swap('vw')
    L2 = make_ruled_volume(*Sl); L2 = L2.swap('uv').swap('vw')
    L3 = make_ruled_volume(*Sm)
    LT1 = make_ruled_volume2(*Sk); LT1 = LT1.swap('uw').swap('vw')
    LT2 = make_ruled_volume2(*Sl); LT2 = LT2.swap('uv').swap('vw')
    LT3 = make_ruled_volume2(*Sm)
    PT = np.ones((2, 2, 2, 4))
    uk, vl, wm = (0, -1), (0, -1), (0, -1)
    for m in xrange(2):
        sm, w = Sm[m], wm[m]
        for l in xrange(2):
            sl, v = Sl[l], vl[l]
            for k in xrange(2):
                sk, u = Sk[k], uk[k]
                Q1, Q2, Q3  = (sk.cobj.cpts[v,w].xyz,
                               sl.cobj.cpts[w,u].xyz,
                               sm.cobj.cpts[u,v].xyz)
                l2n = (util.distance(Q1, Q2) +
                       util.distance(Q1, Q3) +
                       util.distance(Q2, Q3))
                if l2n > 1e-3:
                    print('nurbs.volume.make_trilinear_interp_volume :: '
                          'point inconsistency ({})'.format(l2n))
                PT[k,l,m,:3] = Q1
    T = Volume(ControlVolume(Pw=PT), (1,1,1))
    d, d, d, p, q, r, U, V, W, Vs = \
            make_volumes_compatible1([L1, L2, L3, LT1, LT2, LT3, T])
    Pijk = (Vs[0].cobj.Pw + Vs[1].cobj.Pw + Vs[2].cobj.Pw -
            Vs[3].cobj.Pw - Vs[4].cobj.Pw - Vs[5].cobj.Pw +
            Vs[6].cobj.Pw)
    return Volume(ControlVolume(Pw=Pijk), (p,q,r), (U,V,W))


def make_composite_volume(Vs, di=2):

    ''' Link two or more di-aligned Volumes to form one composite
    Volume.  Unlike make_composite_curve and make_composite_surface, a
    reorientation of the Volumes prior to composition is not performed.

    Parameters
    ----------
    Vs = the connected Volumes to unite
    di = the parametric direction in which to compose the Volumes (0, 1
         or 2)

    Returns
    -------
    Volume = the composite Volume

    '''

    t0 = np.mod(di + 1, 3)
    t1 = np.mod(di + 2, 3)
    Vs = make_volumes_compatible2(Vs, di=di)
    v = Vs[0]
    n, U, Pw = v.cobj.n[di], v.U[di], v.cobj.Pw
    UQ = U[:n+1]
    if di == 0:
        Qw = Pw[:-1,:,:]
    elif di == 1:
        Qw = Pw[:,:-1,:]
    elif di == 2:
        Qw = Pw[:,:,:-1]
    for v in Vs[1:]:
        n, U, Pw = v.cobj.n[di], v.U[di], v.cobj.Pw
        UQ = np.append(UQ, U[1:n+1])
        if di == 0:
            Qw = np.append(Qw, Pw[:-1,:,:], axis=0)
        elif di == 1:
            Qw = np.append(Qw, Pw[:,:-1,:], axis=1)
        elif di == 2:
            Qw = np.append(Qw, Pw[:,:,:-1], axis=2)
    v = Vs[-1]
    n, p, U, Pw = v.cobj.n[di], v.p[di], v.U[di], v.cobj.Pw
    UQ = np.append(UQ, U[-p-1:])
    if di == 0:
        Qw = np.append(Qw, Pw[-1:,:,:], axis=0)
    elif di == 1:
        Qw = np.append(Qw, Pw[:,-1:,:], axis=1)
    elif di == 2:
        Qw = np.append(Qw, Pw[:,:,-1:], axis=2)
    UVW = [0, 0, 0]
    UVW[di], UVW[t0], UVW[t1] = UQ, v.U[t0], v.U[t1]
    return Vs[0].__class__(ControlVolume(Pw=Qw), v.p, UVW)


# UTILITIES


def make_volumes_compatible1(Vs):

    ''' Ensure that the Volumes are defined on the same parameter
    ranges, be of common degrees and share the same knot vectors.

    '''

    p = max([v.p[0] for v in Vs])
    q = max([v.p[1] for v in Vs])
    r = max([v.p[2] for v in Vs])
    Umin = min([v.U[0][ 0] for v in Vs])
    Umax = max([v.U[0][-1] for v in Vs])
    Vmin = min([v.U[1][ 0] for v in Vs])
    Vmax = max([v.U[1][-1] for v in Vs])
    Wmin = min([v.U[2][ 0] for v in Vs])
    Wmax = max([v.U[2][-1] for v in Vs])
    Vs1 = []
    for v in Vs1:
        dp, dq, dr = p - v.p[0], q - v.p[1], r - v.p[2]
        v = v.elevate(dp, 0).elevate(dq, 1).elevate(dr, 2)
        knot.remap_knot_vec(v.U[0], Umin, Umax)
        knot.remap_knot_vec(v.U[1], Vmin, Vmax)
        knot.remap_knot_vec(v.U[2], Wmin, Wmax)
        Vs1.append(v)
    U = knot.merge_knot_vecs(*[v.U[0] for v in Vs1])
    V = knot.merge_knot_vecs(*[v.U[1] for v in Vs1])
    W = knot.merge_knot_vecs(*[v.U[2] for v in Vs1])
    Vs2 = []
    for v in Vs1:
        v = v.refine(knot.missing_knot_vec(U, v.U[0]), 0)
        v = v.refine(knot.missing_knot_vec(V, v.U[1]), 1)
        v = v.refine(knot.missing_knot_vec(W, v.U[2]), 2)
        Vs2.append(v)
    n, m, l = Vs2[0].cobj.n
    return n, m, l, p, q, r, U, V, W, Vs2


def make_volumes_compatible2(Vs, di=2):

    ''' Make Volumes compatible (in the di-direction only), and force
    the end parameter value of the ith Volume to be equal to the start
    parameter of the (i + 1)th Volume.

    Parameters
    ----------
    di = the parametric direction in which to make the Volumes
         compatible (0, 1 or 2)

    '''

    t0 = np.mod(di + 1, 3)
    t1 = np.mod(di + 2, 3)
    p = max([v.p[0] for v in Vs])
    q = max([v.p[1] for v in Vs])
    r = max([v.p[2] for v in Vs])
    Umin = min([v.U[t0][ 0] for v in Vs])
    Umax = max([v.U[t0][-1] for v in Vs])
    Vmin = min([v.U[t1][ 0] for v in Vs])
    Vmax = max([v.U[t1][-1] for v in Vs])
    Vs1 = []
    for v in Vs:
        dp, dq, dr = p - v.p[0], q - v.p[1], r - v.p[2]
        v = v.elevate(dp, 0).elevate(dq, 1).elevate(dr, 2)
        knot.remap_knot_vec(v.U[t0], Umin, Umax)
        knot.remap_knot_vec(v.U[t1], Vmin, Vmax)
        Vs1.append(v)
    U = knot.merge_knot_vecs(*[v.U[t0] for v in Vs1])
    V = knot.merge_knot_vecs(*[v.U[t1] for v in Vs1])
    Vs2 = []
    for v in Vs1:
        v = v.refine(knot.missing_knot_vec(U, v.U[t0]), t0)
        v = v.refine(knot.missing_knot_vec(V, v.U[t1]), t1)
        Vs2.append(v)
    vl = Vs2[0]
    U = vl.U[di]; U -= U[0]
    knot.clean_knot_vec(U)
    for vr in Vs2[1:]:
        U = vr.U[di]; U += vl.U[di][-1] - U[0]
        knot.clean_knot_vec(U)
        vl = vr
    return Vs2
