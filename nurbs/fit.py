import numpy as np
import scipy

import basis
import curve
import knot
import nurbs
import surface
import util


__all__ = ['refit_curve',
           'refit_surface']


# INTERPOLATION


def global_curve_interp(n, Q, p, uk=None, U=None):

    ''' Suppose a pth-degree nonrational B-spline curve interpolant to
    the set of points {Qk}, k = 0,...,n, is seeked.  If a parameter
    value ubk is assigned to each Qk, and an appropriate knot vector U =
    {u0,...,um} is selected, it is possible to set up a (n + 1) x (n +
    1) system of linear equations

               Qk = C(ubk) = sum_(i=0)^n (Nip(ubk) * Pi)

    The control points, Pi, are the (n + 1) unknowns.  Let r be the
    number of coordinates in the Qk (r < 4).  Note that this method is
    independent of r; there is only one coefficient matrix, with r
    right-hand sides and, correspondingly, r solution sets for the r
    coordinates.

    Parameters
    ----------
    n + 1 = the number of data points to interpolate
    Q = the point set in object matrix form
    p = the degree of the interpolant
    uk = the parameter values associated to each point (if available)
    U = the knot vector to use for the interpolation (if available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the interpolant

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 369.

    '''

    if p < 1 or n < p:
        raise ImproperInput(p, n)
    Q = nurbs.obj_mat_to_3D(Q)
    P = np.zeros((n + 1, 3))
    if uk is None:
        uk = knot.chord_length_param(n, Q)
    if U is None:
        U = knot.averaging_knot_vec(n, p, uk)
    A = scipy.sparse.lil_matrix((n + 1, n + 1))
    for i in xrange(n + 1):
        span = basis.find_span(n, p, U, uk[i])
        A[i,span-p:span+1] = basis.basis_funs(span, uk[i], p, U)
    lu = scipy.sparse.linalg.splu(A.tocsc())
    for i in xrange(3):
        rhs = Q[:,i]
        P[:,i] = lu.solve(rhs)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def global_curve_interp_ders(n, Q, p, k, Ds, l, De, uk=None):

    ''' Idem to global_curve_interp, but, in addition to the data
    points, end derivatives are also given

               Ds^1,...,Ds^k    De^1,...,De^l    k,l < p

    where Ds^i denotes the ith derivative at the start point and De^j
    the jth derivative at the end.

    Parameters
    ----------
    n + 1 = the number of data points to interpolate
    Q = the point set in object matrix form
    p = the degree of the interpolant
    k, l = the number of start and end point derivatives
    Ds, De = the start and end point derivatives
    uk = the parameter values associated to each point (if available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the interpolant

    Source
    ------
    Piegl & Tiller, Curve Interpolation with Arbitrary End Derivatives,
    Engineering with Computers, 2000.

    '''

    if p < 1 or not k < p or not l < p or n + k + l < p:
        raise ImproperInput(p, n, k, l)
    Q = nurbs.obj_mat_to_3D(Q)
    Ds, De = [np.asfarray(V) for V in Ds, De]
    nk = n + k; nkl = nk + l
    P = np.zeros((nkl + 1, 3))
    if uk is None:
        uk = knot.chord_length_param(n, Q)
    U = knot.end_derivs_knot_vec(n, p, k, l, uk)
    A = scipy.sparse.lil_matrix((nkl + 1, nkl + 1))
    A[:k+1,:p+1] = \
            basis.ders_basis_funs(p, uk[0], p, k, U)
    for i in xrange(1, n):
        span = basis.find_span(nkl, p, U, uk[i])
        A[i+k,span-p:span+1] = basis.basis_funs(span, uk[i], p, U)
    A[nkl+1:nk-1:-1,nkl-p:nkl+1] = \
            basis.ders_basis_funs(nkl, uk[-1], p, l, U)
    lu = scipy.sparse.linalg.splu(A.tocsc())
    for i in xrange(3):
        rhs = Q[:,i]
        ind = k * [1] + l * [n]
        der = np.hstack((Ds[:,i]    if k else [],
                         De[::-1,i] if l else []))
        rhs = np.insert(rhs, ind, der)
        P[:,i] = lu.solve(rhs)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def global_surf_interp(n, m, Q, p, q, uk=None, vl=None):

    ''' A set of (n + 1) x (m + 1) data poins {Qkl}, k=0,...,n and
    l=0,...,m, is given, and it is desired to construct a nonrational
    (p,q)th-degree B-spline surface interpolating the points, i.e.

                          Qkl = S(ubk, vbk) =
          sum_(i=0)^n sum_(j=0)^m (Nip(ubk) * Njq(vbk) * Pij)

    The interpolation is conducted in two steps:

    1.  using U and the ub_k, do (m + 1) curve interpolations through
        Q0l,...,Qnl (for l=0,...,m); this yields the points {Ril};

    2.  using V and the vb_l, do (n + 1) curve interpolations through
        Ri0,...,Rim (for i=0,...,n); this yields the {Pij}.

    Parameters
    ----------
    n + 1, m + 1 = the number of data points to interpolate in the u and
                   v directions, respectively
    Q = the point set in object matrix form
    p, q = the degrees of the interpolant in the and v directions,
           respectively
    uk, vl = the parameter values associated to each points in the u and
             v directions, respectively (if available)

    Returns
    -------
    U, V, Pw = the knot vectors and the object matrix of the interpolant

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 380.

    '''

    Q = nurbs.obj_mat_to_3D(Q)
    Pw = np.zeros((n + 1, m + 1, 4))
    if uk is None or vl is None:
        uk2, vl2 = surf_mesh_params(n, m, Q)
        if uk is None: uk = uk2
        if vl is None: vl = vl2
    U = knot.averaging_knot_vec(n, p, uk)
    V = knot.averaging_knot_vec(m, q, vl)
    for l in xrange(m + 1):
        dummy, Pw[:,l] = global_curve_interp(n, Q[:,l], p, uk, U)
    for i in xrange(n + 1):
        dummy, Pw[i,:] = global_curve_interp(m, Pw[i,:], q, vl, V)
    return U, V, Pw


def local_curve_interp(n, Q, Ts=None, Te=None):

    ''' Let {Qk}, k=0,...,n, be a set of three-dimensional data points.
    This algorithm constructs a C1 cubic (p = 3) Bezier curve segment,
    Ck(u), between each pair of points Q_k, Q_(k+1).

    Parameters
    ----------
    n + 1 = the number of data points to interpolate
    Q = the point set in object matrix form
    Ts, Te = the tangent vectors at the start and end data points (if
             available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the interpolant

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 395.

    '''

    if n < 1:
        raise ImproperInput(n)
    Q = nurbs.obj_mat_to_3D(Q)
    T = compute_unit_tangents(n, Q)
    if Ts is not None: T[ 0] = util.normalize(Ts)
    if Te is not None: T[-1] = util.normalize(Te)
    uk = np.zeros(n + 1)
    P = np.zeros((2 * n + 2, 3))
    P[0] = Q[0]
    for k in xrange(n):
        a = 16.0 - util.norm(T[k] + T[k+1])**2
        b = 12.0 * np.dot(Q[k+1] - Q[k], T[k] + T[k+1])
        c = - 36.0 * util.norm(Q[k+1] - Q[k])**2
        alfs = np.roots((a, b, c))
        dummy, alf = np.sort(alfs)
        P[2*k+1] = Q[k] + alf * T[k] / 3.0
        P[2*k+2] = Q[k+1] - alf * T[k+1] / 3.0
        uk[k+1] = uk[k] + 3.0 * util.norm(P[2*k+1] - Q[k])
    P[-1] = Q[-1]
    Pw = nurbs.obj_mat_to_4D(P)
    uk = uk[1:n].repeat(2) / uk[-1]
    U = np.zeros(2 * n + 6); U[-4:] = 1.0
    U[4:-4] = uk
    return U, Pw


def local_surf_interp(n, m, Q):

    ''' A C11 (C1 continuous in both the u and v directions), bicubic,
    local surface interpolation scheme.  Let {Qkl}, k=0,...,n and
    l=0,...,m, be a set of data points, and let {(ubk, vbl)} be the
    corresponding parameter pairs, computed by chord length averaging.
    This algorithm produces a bicubic surface, S(u,v), satisfying

                              S(ubk,vbl) =
     sum_(i=0)^(2n+1) sum_(j=0)^(2m+1) (Ni3(ubk) * Nj3(vbl) * Pij)

    This surface is obtained by constructing nm bicubic Bezier patches,
    {Bkl(u,v)}, k=0,...,n-1, l=0,...,m-1, where Q_(k,l), Q_(k+1,l),
    Q_(k,l+1), Q_(k+1,l+1) are the corner points of the patch, and the
    patches join with C11 continuity across their boundaries.

    Parameters
    ----------
    n + 1, m + 1 = the number of data points to interpolate in the u and
                   v directions, respectively
    Q = the point set in object matrix form

    Returns
    -------
    U, V, Pw = the knot vectors and the object matrix of the interpolant

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 404.

    '''

    if n < 1 or m < 1:
        raise ImproperInput(n, m)
    Q = nurbs.obj_mat_to_3D(Q)
    uk, vl = np.zeros(n + 1), np.zeros(m + 1)
    r, s = np.zeros(m + 1), np.zeros(n + 1)
    Td = np.zeros((n + 1, m + 1, 3, 3))
    U, V = np.zeros(2 * n + 6), np.zeros(2 * m + 6)
    P = np.zeros((3 * n + 1, 3 * m + 1, 3))
    total = 0.0
    for l in xrange(m + 1):
        Td[:,l,0] = compute_unit_tangents(n, Q[:,l])
        r[l] = 0.0
        for k in xrange(1, n + 1):
            d = util.distance(Q[k,l], Q[k-1,l])
            uk[k] += d
            r[l] += d
        total += r[l]
    for k in xrange(1, n):
        uk[k] = uk[k-1] + uk[k] / total
    uk[n] = 1.0
    total = 0.0
    for k in xrange(n + 1):
        Td[k,:,1] = compute_unit_tangents(m, Q[k,:])
        s[k] = 0.0
        for l in xrange(1, m + 1):
            d = util.distance(Q[k,l], Q[k,l-1])
            vl[l] += d
            s[k] += d
        total += s[k]
    for l in xrange(1, m):
        vl[l] = vl[l-1] + vl[l] / total
    vl[m] = 1.0
    U[4:-4], V[4:-4] = uk[1:n].repeat(2), vl[1:m].repeat(2)
    U[-4:], V[-4:] = 1.0, 1.0
    for l in xrange(m + 1):
        for k in xrange(n + 1):
            P[3*k,3*l] = Q[k,l]
    for l in xrange(m + 1):
        for k in xrange(n):
            a = r[l] * (uk[k+1] - uk[k]) / 3.0
            P[3*k+1,3*l] = Q[k,l] + a * Td[k,l,0]
            P[3*k+2,3*l] = Q[k+1,l] - a * Td[k+1,l,0]
    for k in xrange(n + 1):
        for l in xrange(m):
            a = s[k] * (vl[l+1] - vl[l]) / 3.0
            P[3*k,3*l+1] = Q[k,l] + a * Td[k,l,1]
            P[3*k,3*l+2] = Q[k,l+1] - a * Td[k,l+1,1]
    Dkl = np.zeros((n + 1, m + 1, 2, 3))
    for l in xrange(m + 1):
        for k in xrange(n + 1):
            Dkl[k,l,0] = r[l] * Td[k,l,0]
            Dkl[k,l,1] = s[k] * Td[k,l,1]
    ak = np.ones(n + 1)
    for k in xrange(1, n):
        duk, duk1 = uk[k] - uk[k-1], uk[k+1] - uk[k]
        ak[k] = duk / (duk + duk1)
    bl = np.ones(m + 1)
    for l in xrange(1, m):
        dvl, dvl1 = vl[l] - vl[l-1], vl[l+1] - vl[l]
        bl[l] = dvl / (dvl + dvl1)
    dkl = np.zeros((n + 1, m + 1, 2, 3))
    for k in xrange(1, n):
        for l in xrange(m + 1):
            duk, duk1 = uk[k] - uk[k-1], uk[k+1] - uk[k]
            dkl[k,l,0] = ((1.0 - ak[k]) * (Dkl[k,l,1] - Dkl[k-1,l,1]) / duk +
                           ak[k] * (Dkl[k+1,l,1] - Dkl[k,l,1]) / duk1)
    for l in xrange(1, m):
        for k in xrange(n + 1):
            dvl, dvl1 = vl[l] - vl[l-1], vl[l+1] - vl[l]
            dkl[k,l,1] = ((1.0 - bl[l]) * (Dkl[k,l,0] - Dkl[k,l-1,0]) / dvl +
                           bl[l] * (Dkl[k,l+1,0] - Dkl[k,l,0]) / dvl1)
    for k in xrange(n + 1):
        dvl1, dvln = vl[1] - vl[0], vl[-1] - vl[-2]
        dkl[k,0,1] = 2.0 * (Dkl[k,1,0] - Dkl[k,0,0]) / dvl1 - dkl[k,1,0]
        dkl[k,-1,1] = 2.0 * (Dkl[k,-1,0] - Dkl[k,-2,0]) / dvln - dkl[k,-2,0]
    for l in xrange(m + 1):
        duk1, dukn = uk[1] - uk[0], uk[-1] - uk[-2]
        dkl[0,l,0] = 2.0 * (Dkl[1,l,1] - Dkl[0,l,1]) / duk1 - dkl[1,l,1]
        dkl[-1,l,0] = 2.0 * (Dkl[-1,l,1] - Dkl[-2,l,1]) / dukn - dkl[-2,l,1]
    for l in xrange(m + 1):
        for k in xrange(n + 1):
            Td[k,l,2] = ((ak[k] * dkl[k,l,1] + bl[l] * dkl[k,l,0]) /
                         (ak[k] + bl[l]))
    for l in xrange(m):
        for k in xrange(n):
            gamma = (uk[k+1] - uk[k]) * (vl[l+1] - vl[l]) / 9.0
            P[3*k+1,3*l+1] = (gamma * Td[k,l,2] +
                              P[3*k,3*l+1] + P[3*k+1,3*l] - P[3*k,3*l])
            P[3*k+2,3*l+1] = (- gamma * Td[k+1,l,2] +
                              P[3*k+3,3*l+1] - P[3*k+3,3*l] + P[3*k+2,3*l])
            P[3*k+1,3*l+2] = (- gamma * Td[k,l+1,2] +
                              P[3*k+1,3*l+3] - P[3*k,3*l+3] + P[3*k,3*l+2])
            P[3*k+2,3*l+2] = (gamma * Td[k+1,l+1,2] +
                              P[3*k+2,3*l+3] + P[3*k+3,3*l+2] - P[3*k+3,3*l+3])
    P = np.delete(P, np.s_[3:-3:3], axis=0)
    P = np.delete(P, np.s_[3:-3:3], axis=1)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, V, Pw


# APPROXIMATION


def global_curve_approx_fixedn(r, Q, p, n, uk=None, U=None, NTN=None):

    ''' Assume the degree, number of control points (minus one) and data
    points are given.  A pth-degree nonrational curve

            C(u) = sum_(i=0)^n (Nip(u) * Pi)    u in [0, 1]

    is seeked, satisfying that:

    - Q0 = C(0) and Qm = C(1);
    - the remaining Qk are approximated in the least-squares sense, i.e.

                    sum_(k=1)^(m-1) |Qk - C(ubk)|^2

      is a minimum with respect to the (n + 1) variables, Pi; the {ubk}
      are the precomputed parameter values.

    The resulting curve generally does not pass precisely through Qk,
    and C(ubk) is not the closest points on C(u) to Qk.

    Parameters
    ----------
    r + 1 = the number of data points to fit
    Q = the point set in object matrix form
    p = the degree of the fit
    n + 1 = the number of control points to use in the fit
    uk = the parameter values associated to each data point (if available)
    U = the knot vector to use in the fit (if available)
    NTN = the decomposed matrix of scalars (if available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the fitted curve

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 410.

    '''

    if p < 1 or n < p:
        raise ImproperInput(p, n)
    if not r > n:
        raise NotEnoughDataPoints(r, n)
    Q = nurbs.obj_mat_to_3D(Q)
    if uk is None:
        uk = knot.chord_length_param(r, Q)
    if U is None:
        U = knot.approximating_knot_vec(n, p, r, uk)
    if NTN is None:
        lu, NT, Os, Oe = build_decompose_NTN(r, p, n, uk, U)
    else:
        lu, NT, Os, Oe = NTN
    R = Q - (Os[0] * Q[0][:,np.newaxis]).T - \
            (Oe[0] * Q[r][:,np.newaxis]).T
    P = np.zeros((n + 1, 3))
    P[0], P[n] = Q[0], Q[r]
    for i in xrange(3):
        rhs = np.dot(NT, R[1:-1,i])
        P[1:-1,i] = lu.solve(rhs)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def global_curve_approx_fixedn_ders(r, Q, p, n, k, Ds, l, De, uk=None):

    ''' Idem to global_curve_approx_fixedn, but, in addition to the data
    points, end derivatives are also given

               Ds^1,...,Ds^k    De^1,...,De^l    k,l < p + 1

    where Ds^i denotes the ith derivative at the start point and De^j
    the jth derivative at the end.

    Parameters
    ----------
    r + 1 = the number of data points to fit
    Q = the point set in object matrix form
    p = the degree of the fit
    n + 1 = the number of control points to use in the fit
    k, l = the number of start and end point derivatives
    Ds, De = the start and end point derivatives
    uk = the parameter values associated to each data point (if available)

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the fitted curve

    Source
    ------
    Piegl & Tiller, Least-Squares B-spline Curve Approximation with
    Arbitrary End Derivatives, Engineering with Computers, 2000.

    '''

    if p < 1 or n < p or p < k or p < l or n < k + l + 1:
        raise ImproperInput(p, n, k, l)
    if not r > n:
        raise NotEnoughDataPoints(r, n)
    Q = nurbs.obj_mat_to_3D(Q)
    Ds, De = [np.asfarray(V) for V in Ds, De]
    if uk is None:
        uk = knot.chord_length_param(r, Q)
    U = knot.approximating_knot_vec_end(n, p, r, k, l, uk)
    P = np.zeros((n + 1, 3))
    P[0], P[n] = Q[0], Q[r]
    if k > 0:
        ders = basis.ders_basis_funs(p, uk[0], p, k, U)
        for i in xrange(1, k + 1):
            Pi = Ds[i-1]
            for h in xrange(i):
                Pi -= ders[i,h] * P[h]
            P[i] = Pi / ders[i,i]
    if l > 0:
        ders = basis.ders_basis_funs(n, uk[-1], p, l, U)
        for j in xrange(1, l + 1):
            Pj = De[j-1]
            for h in xrange(j):
                Pj -= ders[j,-h-1] * P[-h-1]
            P[-j-1] = Pj / ders[j,-j-1]
    if n > k + l + 1:
        Ps = [P[   i][:,np.newaxis] for i in xrange(k + 1)]
        Pe = [P[-j-1][:,np.newaxis] for j in xrange(l + 1)]
        lu, NT, Os, Oe = build_decompose_NTN(r, p, n, uk, U, k, l)
        R = Q.copy()
        for i in xrange(k + 1):
            R -= (Os[i] * Ps[i]).T
        for j in xrange(l + 1):
            R -= (Oe[j] * Pe[j]).T
        for i in xrange(3):
            rhs = np.dot(NT, R[1:-1,i])
            P[k+1:-l-1,i] = lu.solve(rhs)
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def global_curve_approx_fixedn_fair(r, Q, p, n, B=60):

    ''' Idem to global_curve_approx_fixedn, but, in addition to fitting
    the data points, the curve energy functional

                    int_u ( B * || C''(u) ||^2 ) du

    is also minimized.  B is some user-defined coefficient; if set to 0
    then the problem reverts back to the normal least-squares fit.  Note
    that given the polynomial nature of B-splines the above integral is
    carried out exactly using Gauss-Legendre quadrature.

    Parameters
    ----------
    r + 1 = the number of data points to fit
    Q = the point set in object matrix form
    p = the degree of the fit
    n + 1 = the number of control points to use in the fit
    B = the bending coefficient

    Returns
    -------
    U, Pw = the knot vector and the object matrix of the fitted curve

    Source
    ------
    Park et al., A method for approximate NURBS curve compatibility
    based on multiple curve refitting, Computer-aided design, 2000.

    '''

    if p < 1 or n < p:
        raise ImproperInput(p, n)
    if not r > n:
        raise NotEnoughDataPoints(r, n)
    Q = nurbs.obj_mat_to_3D(Q)
    uk = knot.chord_length_param(r, Q)
    U = knot.approximating_knot_vec(n, p, r, uk)
    Uu = np.unique(U); ni = len(Uu) - 1
    # get the Gauss-Legendre roots and weights
    gn = (p + 2) // 2
    gx, gw = scipy.special.orthogonal.p_roots(gn)
    # precompute the N^(2) vector for each knot span
    u = []
    for i in xrange(ni):
        a, b = Uu[i], Uu[i+1]
        uu = (b - a) * (gx + 1) / 2.0 + a
        u.append(uu)
    u = np.hstack(u)
    N2 = np.zeros((n + 1, gn * ni))
    for i in xrange(n + 1):
        N2[i] = basis.ders_one_basis_fun_v(p, U, i, u, 2, gn * ni)[2]
    # build the (normalized) stiffness matrix K
    K = np.zeros((n + 1, n + 1))
    for i in xrange(n + 1):
        for j in xrange(i, min(i + p + 1, n + 1)):
            kij = 0.0
            for k in xrange(ni):
                a, b = Uu[k], Uu[k+1]
                kgn = k * gn
                NN = N2[i,kgn:kgn+gn] * N2[j,kgn:kgn+gn]
                kij += (b - a) / 2.0 * np.dot(gw, NN)
            K[i,j] = kij
    K += K.T; K[np.diag_indices_from(K)] /= 2
    K /= np.max(K)
    # build the system matrix NTN
    N = np.zeros((r + 1, n + 1))
    spans = basis.find_span_v(n, p, U, uk, r + 1)
    bfuns = basis.basis_funs_v(spans, uk, p, U, r + 1)
    spans0, spans1 = spans - p, spans + 1
    for s in xrange(r + 1):
        N[s,spans0[s]:spans1[s]] = bfuns[:,s]
    NTN = np.dot(N.T, N)
    # build the matrix of constraints C
    C = np.zeros((2, n + 1))
    for i in (0, -1):
        span = basis.find_span(n, p, U, uk[i])
        bfun = basis.basis_funs(span, uk[i], p, U)
        C[i,span-p:span+1] = bfun
    # build the rhs vector D
    D = np.zeros((2, 3))
    D[0], D[1] = Q[0], Q[r]
    # build and solve Eq. (12)
    A = scipy.sparse.lil_matrix((n + 3, n + 3))
    A[:n+1,:n+1] = B * K + NTN
    A[:n+1,n+1:] = C.T
    A[n+1:,:n+1] = C
    lu = scipy.sparse.linalg.splu(A.tocsc())
    P = np.zeros((n + 1, 3))
    rhs = np.zeros(n + 3)
    for i in xrange(3):
        rhs[:n+1] = np.dot(N.T, Q[:,i])
        rhs[n+1:] = D[:,i]
        sol = lu.solve(rhs)
        P[:,i] = sol[:n+1]
    Pw = nurbs.obj_mat_to_4D(P)
    return U, Pw


def global_curve_approx_errbnd(r, Q, p, E, ps=1):

    ''' Approximate (fit) a set of points {Qk}, k=0,...,r, to within a
    specified tolerance E.  The algorithm is a Type 2 approximation
    method, i.e. it starts with many control points, fit, check
    deviation, and discard control points if possible.

    While the least-squares based algorithm global_curve_approx_fixedn
    is appropriate for the fitting step, it can sometimes fail, either
    because degree elevation requires more control points than there are
    data points, or because the many multiple knots make the system of
    equations singular.  A simple fix is to restart the process with a
    higher degree curve.  The deviation checking is measured by the max
    norm deviation and, finally, control points are discarded by knot
    removal.

    Parameters
    ----------
    r + 1 = the number of data points to fit
    Q = the point set in object matrix form
    p = the degree of the fit
    E = the max norm deviation of the fit
    ps = the starting degree for the interpolating curve

    Returns
    -------
    Uh, Pwh = the knot vector and the object matrix of the fitted curve

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 431.

    '''

    if p < ps:
        raise ImproperInput(p, ps)
    Q = nurbs.obj_mat_to_3D(Q)
    uk = knot.chord_length_param(r, Q)
    U, Pw = global_curve_interp(r, Q, ps, uk)
    n = Pw.shape[0] - 1
    ek = np.zeros(r + 1)
    try:
        for deg in xrange(ps, p + 1):
            nh, Uh, dummy = remove_knots_bounds_curve(n, deg, U, Pw, uk, ek, E)
            if deg == p:
                break
            n, U = curve.mult_degree_elevate(nh, deg, Uh, 1)
            U, Pw = global_curve_approx_fixedn(r, Q, deg + 1, n, uk, U)
            update_errors(n, deg + 1, U, Pw, r, Q, uk, ek)
        if n == nh:
            return U, Pw
        U, n = Uh, nh
        U, Pw = global_curve_approx_fixedn(r, Q, p, n, uk, U)
        update_errors(n, p, U, Pw, r, Q, uk, ek)
        dummy, Uh, Pwh = remove_knots_bounds_curve(n, p, U, Pw, uk, ek, E)
        return Uh, Pwh
    except (RuntimeError, NotEnoughDataPoints):
        return global_curve_approx_errbnd(r, Q, p, E, ps + 1)


def global_surf_approx_fixednm(r, s, Q, p, q, n, m, uk=None, vl=None):

    ''' Let {Qkl}, k=0,...,r and l=0,...,s, be the (r + 1) x (s + 1) set
    of points to be approximated by a (p,q)th degree nonrational
    surface, with (n + 1) x (m + 1) control points.  The algorithm
    interpolates the corner points Q00, Qr0, Q0s and Qrs precisely, and
    approximates the remaining {Qkl}.  It fits the (s + 1) rows of data
    first, then fits across the resulting control points to produce the
    (n + 1) x (m + 1) surface control points.

    Parameters
    ----------
    r + 1, s + 1 = the number of data points to fit in the u and v
                   directions, respectively
    Q = the point set in object matrix form
    p, q = the degrees of the fit in the u and v directions,
           respectively
    n + 1, m + 1 = the number of control points in the u and v
                   directions, respectively, to use in the fit
    uk, vk = the parameter values associated to each points in the u and
             v directions, respectively (if available)

    Returns
    -------
    U, V, Pw = the knot vectors and the object matrix of the fitted
               surface

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 422.

    '''

    Q = nurbs.obj_mat_to_3D(Q)
    if uk is None or vl is None:
        uk2, vl2 = surf_mesh_params(r, s, Q)
        if uk is None: uk = uk2
        if vl is None: vl = vl2
    U = knot.approximating_knot_vec(n, p, r, uk)
    V = knot.approximating_knot_vec(m, q, s, vl)
    NTNu = build_decompose_NTN(r, p, n, uk, U)
    NTNv = build_decompose_NTN(s, q, m, vl, V)
    tmp = np.zeros((n + 1, s + 1, 4))
    Pw = np.zeros((n + 1, m + 1, 4))
    for j in xrange(s + 1):
        dummy, tmp[:,j] = \
                global_curve_approx_fixedn(r, Q[:,j], p, n, uk, U, NTNu)
    for i in xrange(n + 1):
        dummy, Pw[i,:] = \
                global_curve_approx_fixedn(s, tmp[i,:], q, m, vl, V, NTNv)
    return U, V, Pw


# TOOLBOX


def refit_curve(C, ncp, p=3, num=1000):

    ''' Refit an arbitrary Curve with another Curve of arbitrary degree
    by sampling it at equally spaced intervals.  If possible the
    original end derivatives are kept intact.

    Parameters
    ----------
    C = the Curve to refit
    ncp = the number of control points to use in the fit
    p = the degree to use in the fit
    num = the number of points to sample C with

    Returns
    -------
    Curve = the refitted Curve

    '''

    U, = C.U
    us, = util.construct_flat_grid((U,), (num,))
    Q = C.eval_points(us).T
    Ds, De = [], []
    if ncp > 3:
        Ds = C.eval_derivatives(U[ 0], 1)[1:]
        De = C.eval_derivatives(U[-1], 1)[1:]
    U, Pw = global_curve_approx_fixedn_ders(num - 1, Q, p, ncp - 1,
                                            len(Ds), Ds, len(De), De, us)
    return curve.Curve(curve.ControlPolygon(Pw=Pw), (p,), (U,))


def refit_surface(S, ncp, mcp, numu=200, numv=200, eps=None):

    ''' Refit an arbitrary Surface with another Surface of degree (3,3)
    by sampling it at equally spaced intervals.  The fit is performed in
    two steps.  First, the four boundary Curves of S as well as their
    corresponding cross-boundary derivatives are fitted to form a Coons
    patch (thus yielding a gross approximation of S).  Second, the
    internal control points of that patch are corrected in the
    least-squares sense.

    Parameters
    ----------
    S = the Surface to refit
    ncp, mcp = the number of control points in the u and v directions,
               respectively, to use in the fit
    numu, numv = the number of points to sample S with in the u and v
                 directions, respectively
    eps = see surface.make_coons_surface

    Returns
    -------
    Surface = the bicubic, refitted Surface

    '''

    S = S.copy(); U, V = S.U
    knot.normalize_knot_vec(U)
    knot.normalize_knot_vec(V)
    Ck = [S.extract(0, 1), S.extract(1, 1)]
    Cl = [S.extract(0, 0), S.extract(1, 0)]
    Dk = [surface.extract_cross_boundary_deriv(S, 2),
          surface.extract_cross_boundary_deriv(S, 3)]
    Dl = [surface.extract_cross_boundary_deriv(S, 0),
          surface.extract_cross_boundary_deriv(S, 1)]
    Ck = [refit_curve(C, ncp) for C in Ck]
    Cl = [refit_curve(C, mcp) for C in Cl]
    Dk = [refit_curve(D, ncp) for D in Dk]
    Dl = [refit_curve(D, mcp) for D in Dl]
    Sr = surface.make_coons_surface(Ck, Cl, Dk, Dl, eps)
    us, vs = (np.linspace(0, 1, numu),
              np.linspace(0, 1, numv))
    usf, vsf = util.construct_flat_grid((us, vs))
    Q = S.eval_points(usf, vsf).T
    Q = Q.reshape((numu, numv, 3))
    global_surf_approx_fixednm_cntr(numu - 1, numv - 1, Q, us, vs, 2,
                                    Sr.cobj.n[0], Sr.cobj.n[1],
                                    Sr.p[0], Sr.p[1],
                                    Sr.U[0], Sr.U[1], Sr.cobj.Pw)
    return Sr


# UTILITIES


def surf_mesh_params(n, m, Q):

    ''' Compute surface mesh parameters (ub_k, vb_l).  In u, it computes
    parameters ub_0^l,...,ub_n^l for each l, and then obtains each ub_k
    by averaging across all ub_k^l, l=0,...,m, that is

         ub_k = 1 / (m + 1)  sum_(l=0)^m * ub_k^l    k=0,...,n

    where for each fixed l, ub_k^l, k=0,...,n, was computed by chord
    length parameterization.  The vb_l are analogous.

    Source: The NURBS Book (2nd Ed.), Pg. 377.

    '''

    uk, vl = np.zeros(n + 1), np.zeros(m + 1)
    cds = np.zeros(max(n + 1, m + 1))
    uk[n] = 1.0
    for l in xrange(m + 1):
        total = 0.0
        for k in xrange(1, n + 1):
            cds[k] = util.distance(Q[k,l], Q[k-1,l])
            total += cds[k]
        d = 0.0
        for k in xrange(1, n):
            d += cds[k]
            uk[k] += d / total
    for k in xrange(1, n):
        uk[k] /= m + 1
    vl[m] = 1.0
    for k in xrange(n + 1):
        total = 0.0
        for l in xrange(1, m + 1):
            cds[l] = util.distance(Q[k,l], Q[k,l-1])
            total += cds[l]
        d = 0.0
        for l in xrange(1, m):
            d += cds[l]
            vl[l] += d / total
    for l in xrange(1, m):
        vl[l] /= n + 1
    return uk, vl


def compute_qks(n, Q):

    ''' Compute the vectors q_k = Q_k - Q_(k-1), k=1,...,n, with special
    treatment at the ends.

    '''

    qk = np.zeros((n + 4, 3))
    for k in xrange(2, n + 2):
        qk[k] = Q[k-1] - Q[k-2]
    qk[1] = 2 * qk[2] - qk[3]
    qk[0] = 2 * qk[1] - qk[2]
    qk[n+2] = 2 * qk[n+1] - qk[n]
    qk[n+3] = 2 * qk[n+2] - qk[n+1]
    return qk


def compute_alfs(n, qk):

    ''' Compute the interpolation parameters.

    '''

    alf = np.zeros(n + 1)
    for k in xrange(1, n + 2):
        num = util.norm(np.cross(qk[k-1], qk[k]))
        den = num + util.norm(np.cross(qk[k+1], qk[k+2]))
        if np.allclose(den, 0.0):
            alf[k-1] = 0.5
            continue
        alf[k-1] = num / den
    return alf


def compute_unit_tangents(n, Q):

    ''' Compute the unit length tangent vectors at each point Qk.

    '''

    q = compute_qks(n, Q)
    alf = compute_alfs(n, q)
    V = np.zeros((n + 1, 3))
    T = np.zeros((n + 1, 3))
    for k in xrange(n + 1):
        V[k] = (1.0 - alf[k]) * q[k+1] + alf[k] * q[k+2]
    for k in xrange(n + 1):
        if np.allclose(V[k], 0.0):
            raise NullTangentVectorDetected(k)
        T[k] = util.normalize(V[k])
    return T


def get_removal_bnd_curve(p, U, Pw, u, r, s):

    ''' Let ur be an interior knot of a pth degree nonrational curve,
    C(u), u_r != u_(r+1), and denote the multiplicity of ur by s, i.e.,
    u_(r-s+j) = u_r for j=1,...,s.  Let Ch(u) denote the curve obtained
    by removing one occurence of ur.  After removal, Ch(u) differs from
    C(u) by Br.

    Source: The NURBS Book (2nd Ed.), Pg. 428.

    '''

    o = p + 1
    first, last = r - p, r - s
    off = first - 1
    tmp = np.zeros((last - off + 2, 4))
    tmp[0], tmp[last-off+1] = Pw[off], Pw[last+1]
    i, j = first, last
    ii, jj = 1, last - off
    while j - i > 0:
        alfi = (u - U[i]) / (U[i+o] - U[i])
        alfj = (u - U[j]) / (U[j+o] - U[j])
        tmp[ii] = (Pw[i] - (1.0 - alfi) * tmp[ii-1]) / alfi
        tmp[jj] = (Pw[j] - alfj * tmp[jj+1]) / (1.0 - alfj)
        i += 1; ii += 1
        j -= 1; jj -= 1
    if j - i < 0:
        Br = util.distance(tmp[ii-1], tmp[jj+1])
    else:
        alfi = (u - U[i]) / (U[i+o] - U[i])
        Br = util.distance(Pw[i], alfi * tmp[ii+1] + (1.0 - alfi) * tmp[ii-1])
    return Br


def get_all_removal_bnd_curve(n, p, U, Pw, dk, num):

    ''' Get the maximum bounds Br for all distinct knots dk.

    '''

    r, s = basis.find_span_mult_v(n, p, U, dk, num)
    Br = dk.copy()
    for k, u in enumerate(dk):
        Br[k] = get_removal_bnd_curve(p, U, Pw, u, r[k], s[k])
    return Br


def remove_knots_bounds_curve(n, p, U, Pw, ub, ek, E):

    ''' Let {Q} be a set of points and {ub} the associated set of
    parameters.  Denote by {ek} a set of errors associated with the
    points, and let E be a maximum error bound.  Given a curve C(u) with
    the property that the max norm deviation of each Qk from C(u) is
    less than or equal to ek, this algorithm removes (roughly) as many
    knots as possible from C(u) while maintaining ek <= E for all k.  It
    also updates the {ek}, i.e., it accumulates the error.  Output is
    the new curve, represented by nh, Uh, and Pwh.

    Source: The NURBS Book (2nd Ed.), Pg. 429.

    '''

    dk = np.unique(U[p+1:-p-1])
    if dk.size == 0:
        return n, U, Pw
    nb = ub.size - 1
    Br = get_all_removal_bnd_curve(n, p, U, Pw, dk, len(dk))
    nh, Uh, Pwh = n, U, Pw
    while True:
        Brmk, Brm = Br.argmin(), Br.min()
        ur = dk[Brmk]
        r, s = basis.find_span_mult(nh, p, Uh, ur)
        if Brm == np.inf:
            break
        NewError = np.zeros(nb + 1)
        for k in xrange(nb + 1):
            if np.mod(p + s, 2) == 0:
                kk = (p + s) / 2
                N = basis.one_basis_fun(p, nh + p + 1, Uh, r - kk, ub[k])
                NewError[k] = N * Brm
            elif np.mod(p + s, 2) == 1:
                kk = (p + s + 1) / 2
                alf = (ur - Uh[r-kk+1]) / (Uh[r-kk+p+2] - Uh[r-kk+1])
                N = basis.one_basis_fun(p, nh + p + 1, Uh, r - kk + 1, ub[k])
                NewError[k] = (1.0 - alf) * N * Brm
        tmp = ek + NewError
        if (tmp <= E).all():
            ek = tmp
            nr, Uh, Pwh = curve.remove_curve_knot(nh, p, Uh, Pwh, ur,
                                                  r, s, 1, np.inf)
            if np.mod(nh + 1, 100) == 0:
                print('nurbs.curve.remove_knots_bounds_curve :: '
                      '{} number of points left'.format(nh + 1))
            nh -= nr
            if Uh.size == 2 * p + 2:
                break
            dk = np.unique(Uh[p+1:-p-1])
            Br = get_all_removal_bnd_curve(nh, p, Uh, Pwh, dk, len(dk))
        else:
            Br[Brmk] = np.inf
    return nh, Uh, Pwh


def update_errors(n, p, U, Pw, r, Qw, uk, ek):

    ''' Update the error vector ek by computing the distances between
    the Qk and the curve C(u).  Also update the parameter values uk.

    '''

    for k in xrange(r + 1):
        try:
            u, = curve.curve_point_projection(n, p, U, Pw, Qw[k,:3], uk[k])
        except nurbs.NewtonLikelyDiverged as e:
            u, = e.args
        uk[k] = u
        ek[k] = util.distance(Qw[k,:3], curve.rat_curve_point(n, p, U, Pw, u))


def build_decompose_NTN(r, p, n, uk, U, k=0, l=0):

    ''' Build the matrix of scalars NTN necessary to solve the
    least-squares problem for curve and surface approximations.  The
    matrix is also decomposed using sparse LU.

    '''

    m = n + p + 1
    Os = [basis.one_basis_fun_v(p, m, U, i, uk, r + 1)
          for i in xrange(k + 1)]
    Oe = [basis.one_basis_fun_v(p, m, U, n - j, uk, r + 1)
          for j in xrange(l + 1)]
    if k + l + 1 == n:
        return None, None, Os, Oe
    N = np.zeros((r + 1, n + 1))
    spans = basis.find_span_v(n, p, U, uk, r + 1)
    bfuns = basis.basis_funs_v(spans, uk, p, U, r + 1)
    spans0, spans1 = spans - p, spans + 1
    for s in xrange(r + 1):
        N[s,spans0[s]:spans1[s]] = bfuns[:,s]
    N = N[1:-1,k+1:-l-1]
    NT = N.transpose()
    NTN = np.dot(NT, N)
    NTN = scipy.sparse.csc_matrix(NTN)
    lu = scipy.sparse.linalg.splu(NTN)
    return lu, NT, Os, Oe


def global_surf_approx_fixednm_cntr(r, s, Q, uk, vl, nc, n, m, p, q, U, V, Pw):

    ''' Idem global_surf_approx_fixednm, but here only the interior
    control points Pw[nc:-nc,nc:-nc] are fitted in the least-squares
    sense; the remaining edge components are constrained and will thus
    remain unchanged (IN-PLACE).

    Source
    ------
    Milroy et al., G1 continuity of B-spline surface patches in reverse
    engineering, Computer-Aided Design, 1995.

    '''

    if (n - nc < 2) or (m - nc < 2):
        return
    spansu = basis.find_span_v(n, p, U, uk, r + 1)
    spansv = basis.find_span_v(m, q, V, vl, s + 1)
    bfunsu = basis.basis_funs_v(spansu, uk, p, U, r + 1)
    bfunsv = basis.basis_funs_v(spansv, vl, q, V, s + 1)
    spansu0, spansu1 = spansu - p, spansu + 1
    spansv0, spansv1 = spansv - q, spansv + 1
    k, drc = 0, []
    for i in xrange(r + 1):
        N = np.zeros(n + 1)
        N[spansu0[i]:spansu1[i]] = bfunsu[:,i]
        N = np.repeat(N, m + 1)
        for j in xrange(s + 1):
            M = np.zeros(m + 1)
            M[spansv0[j]:spansv1[j]] = bfunsv[:,j]
            M = np.tile(M, n + 1)
            data = N * M
            col, = np.nonzero(data)
            data = data[col]
            row = np.zeros(col.size); row.fill(k)
            k += 1
            drc.append((data, row, col))
    data, row, col = [np.hstack(V) for V in zip(*drc)]
    C = scipy.sparse.csc_matrix((data, (row, col)))
    print('nurbs.fit.global_surf_approx_fixednm_cntr :: '
          'number of nonzero elements = {}'.format(C.nnz))
    nrow, ncol = (r + 1) * (s + 1), (n + 1) * (m + 1)
    inds = []
    for i in xrange(nc, n + 1 - nc):
        ind = range(i * (m + 1) + nc, (i + 1) * (m + 1) - nc)
        inds += ind
    cols = np.arange(ncol)
    indsi = cols[~np.in1d(cols, inds)]
    P = Pw[...,:-1]
    B = P.reshape((-1, 3))
    Q = Q.reshape((-1, 3))
    Bi = B[inds]
    Ci = C[:,inds]
    Bedge = B[indsi]; del B
    Cedge = C[:,indsi]; del C
    Q = Q - Cedge.dot(Bedge)
    CiT = Ci.transpose()
    CiTCi = CiT.dot(Ci).tocsc(); del Ci
    try:
        lu = scipy.sparse.linalg.splu(CiTCi); del CiTCi
    except RuntimeError:
        print('nurbs.fit.global_surf_approx_fixednm_cntr :: '
              'factor is exactly singular, aborting.')
        return
    for i in xrange(3):
        rhs = CiT.dot(Q[:,i])
        Bi[:,i] = lu.solve(rhs)
    Pw[nc:-nc,nc:-nc,:-1] = \
            Bi.reshape((n + 1 - 2 * nc, m + 1 - 2 * nc, 3))


# EXCEPTIONS


class FitException(Exception):
    pass

class ImproperInput(FitException):
    pass

class NotEnoughDataPoints(FitException):
    pass

class NullTangentVectorDetected(FitException):
    pass
