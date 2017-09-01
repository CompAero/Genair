import numpy as np

import knot


def find_span(n, p, U, u):

    ''' Determine the knot span index.

    Source: The NURBS Book (2nd Ed.), Pg. 68.

    '''

    u = knot.check_knot(U, u)
    if u == U[n+1]:
        return n
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def find_span_v(n, p, U, u, num):

    ''' Idem find_span, vectorized in u..

    '''

    u = knot.check_knot_v(U, u)
    low = np.array(p).repeat(num)
    high = np.array(n + 1).repeat(num)
    mid = (low + high) // 2
    mid[u == U[n+1]] = n
    c1, c2 = u < U[mid], u >= U[mid+1]
    c3 = u != U[n+1]
    while c1.any() or (c2 & c3).any():
        high = np.where(c1, mid, high)
        low = np.where(c2, mid, low)
        mid = np.where(c1 | c2, (low + high) // 2, mid)
        c1, c2 = u < U[mid], u >= U[mid+1]
    return mid


def find_span_mult(n, p, U, u):

    ''' Determine the knot span index and multiplicity.

    '''

    return find_span(n, p, U, u), U.tolist().count(u)


def find_span_mult_v(n, p, U, u, num):

    ''' Idem find_span_mult, vectorized in u.

    '''

    u = np.asfarray(u)
    s = find_span_v(n, p, U, u, num)
    U = U[np.newaxis,:].repeat(num, axis=0)
    u = u[:,np.newaxis].repeat(n + p + 2, axis=1)
    return s, (U == u).sum(axis=1)


# The following functions are based on the property that, in any given
# knot span, [ u_i, u_(i+1) ), at most p + 1 of the B-spline basis
# functions are nonzero, namely the functions (N_(i-p,p)(u),...,
# N_(i,p)(u)).
# NOTE: i is the knot span index of u


def basis_funs(i, u, p, U):

    ''' Compute all nonvanishing basis functions and store them in the
    array (N[0],...,N[p]).

    Source: The NURBS Book (2nd Ed.), Pg. 70.

    '''

    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0
    for j in xrange(1, p + 1):
        left[j], right[j] = u - U[i+1-j], U[i+j] - u
        saved = 0.0
        for r in xrange(j):
            tmp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        N[j] = saved
    return N


def basis_funs_v(i, u, p, U, num):

    ''' Idem basis_funs, vectorized in u.

    '''

    N = np.zeros((p + 1, num))
    left = np.zeros((p + 1, num))
    right = np.zeros((p + 1, num))
    N[0] = 1.0
    for j in xrange(1, p + 1):
        left[j], right[j] = u - U[i+1-j], U[i+j] - u
        saved = 0.0
        for r in xrange(j):
            tmp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        N[j] = saved
    return N


def all_basis_funs(i, u, p, U):

    ''' Idem basis_funs but for all degrees from 0 up to p.  In
    particular, N[j,n] is the value of the nth-degree basis function,
    N_(i-n+j,n)(u), where (0 <= n <= p) and (0 <= j <= n).

    '''

    N = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0,:] = 1.0
    for p in xrange(p + 1):
        for j in xrange(1, p + 1):
            left[j] = u - U[i+1-j]
            right[j] = U[i+j] - u
            saved = 0.0
            for r in xrange(j):
                tmp = N[r,p] / (right[r+1] + left[j-r])
                N[r,p] = saved + right[r+1] * tmp
                saved = left[j-r] * tmp
            N[j,p] = saved
    return N


def ders_basis_funs(i, u, p, n, U):

    ''' Compute the nonzero basis functions and their derivatives, up to
    and including the nth derivative (n <= p).  Output is in the
    two-dimensional array, ders.  ders[k,j] is the kth derivative of the
    function N_(i-p+j,p)(u) where (0 <= k <= n) and (0 <= j <= p).

    Source: The NURBS Book (2nd Ed.), Pg. 72.

    '''

    ders = np.zeros((n + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    a = np.zeros((2, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu[0,0] = 1.0
    for j in xrange(1, p + 1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in xrange(j):
            ndu[j,r] = right[r+1] + left[j-r]
            tmp = ndu[r,j-1] / ndu[j,r]
            ndu[r,j] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        ndu[j,j] = saved
    for j in xrange(p + 1):
        ders[0,j] = ndu[j,p]
    for r in xrange(p + 1):
        s1, s2 = 0, 1
        a[0,0] = 1.0
        for k in xrange(1, n + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2,0] = a[s1,0] / ndu[pk+1,rk]
                d = a[s2,0] * ndu[rk,pk]
            if rk >= - 1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in xrange(j1, j2 + 1):
                a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1,rk+j]
                d += a[s2,j] * ndu[rk+j,pk]
            if r <= pk:
                a[s2,k] = - a[s1,k-1] / ndu[pk+1,r]
                d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j = s1; s1 = s2; s2 = j
    r = p
    for k in xrange(1, n + 1):
        for j in xrange(p + 1):
            ders[k,j] *= r
        r *= p - k
    return ders


def ders_basis_funs_v(i, u, p, n, U, num):

    ''' Idem ders_basis_funs, vectorized in u.

    '''

    ders = np.zeros((n + 1, p + 1, num))
    ndu = np.zeros((p + 1, p + 1, num))
    a = np.zeros((2, p + 1, num))
    left = np.zeros((p + 1, num))
    right = np.zeros((p + 1, num))
    ndu[0,0] = 1.0
    for j in xrange(1, p + 1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in xrange(j):
            ndu[j,r] = right[r+1] + left[j-r]
            tmp = ndu[r,j-1] / ndu[j,r]
            ndu[r,j] = saved + right[r+1] * tmp
            saved = left[j-r] * tmp
        ndu[j,j] = saved
    for j in xrange(p + 1):
        ders[0,j] = ndu[j,p]
    for r in xrange(p + 1):
        s1, s2 = 0, 1
        a[0,0] = 1.0
        for k in xrange(1, n + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2,0] = a[s1,0] / ndu[pk+1,rk]
                d = a[s2,0] * ndu[rk,pk]
            if rk >= - 1:
                j1 = 1
            else:
                j1 = -rk
            if r - 1 <= pk:
                j2 = k - 1
            else:
                j2 = p - r
            for j in xrange(j1, j2 + 1):
                a[s2,j] = (a[s1,j] - a[s1,j-1]) / ndu[pk+1,rk+j]
                d += a[s2,j] * ndu[rk+j,pk]
            if r <= pk:
                a[s2,k] = -a[s1,k-1] / ndu[pk+1,r]
                d += a[s2,k] * ndu[r,pk]
            ders[k,r] = d
            j = s1; s1 = s2; s2 = j
    r = p
    for k in xrange(1, n + 1):
        for j in xrange(p + 1):
            ders[k,j] *= r
        r *= p - k
    return ders


# By contrast, the following two functions are based on the property
# that N_(i,p)(u) = 0 if u is outside the interval [ u_i, u_(i+p+1) )
# (local support property).
# NOTE: here, i refers to the basis function index


def one_basis_fun(p, m, U, i, u):

    ''' Compute the basis function N_(i,p)(u).

    Source: The NURBS Book (2nd Ed.), Pg. 74.

    '''

    N = np.zeros(p + 1)
    if ((i == 0 and u == U[0]) or (i == m - p - 1 and u == U[m])):
        return 1.0
    if u < U[i] or u >= U[i+p+1]:
        return 0.0
    for j in xrange(p + 1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j] = 1.0
    for k in xrange(1, p + 1):
        if N[0] == 0.0:
            saved = 0.0
        else:
            saved = ((u - U[i]) * N[0]) / (U[i+k] - U[i])
        for j in xrange(p - k + 1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            if N[j+1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                tmp = N[j+1] / (Uright - Uleft)
                N[j] = saved + (Uright - u) * tmp
                saved = (u - Uleft) * tmp
    return N[0]


def one_basis_fun_v(p, m, U, i, u, num):


    ''' Idem one_basis_fun, vectorized in u.

    '''

    u = np.asfarray(u)
    N = np.zeros((p + 1, num))
    cs = np.ones(num, dtype='bool')
    if i == 0:
        c = u == U[0]
        N[0,c] = 1.0; cs[c] = False
    if i == m - p - 1:
        c = u == U[m]
        N[0,c] = 1.0; cs[c] = False
    c = u < U[i]
    N[0,cs & c] = 0.0; cs[c] = False
    c = u >= U[i+p+1]
    N[0,cs & c] = 0.0; cs[c] = False
    NN = N[:,cs]
    u = u[cs]
    for j in xrange(p + 1):
        NN[j,(u >= U[i+j]) & (u < U[i+j+1])] = 1.0
    for k in xrange(1, p + 1):
        if (U[i+k] - U[i]) != 0.0:
            divs = ((u - U[i]) * NN[0]) / (U[i+k] - U[i])
        else:
            divs = None
        saved = np.where(NN[0] == 0.0, 0.0, divs)
        for j in xrange(p - k + 1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            zs = NN[j+1] == 0.0; nzs = ~ zs
            NN[j] = np.where(zs, saved, NN[j])
            saved = np.where(zs, 0.0, saved)
            if Uright - Uleft != 0.0:
                divs = NN[j+1] / (Uright - Uleft)
            else:
                divs = None
            tmp = np.where(nzs, divs, 0.0)
            NN[j] = np.where(nzs, saved + (Uright - u) * tmp, NN[j])
            saved = np.where(nzs, (u - Uleft) * tmp, saved)
    N[:,cs] = NN
    return N[0]


def ders_one_basis_fun(p, U, i, u, n):

    ''' Compute derivatives of basis functions N_(i,p)^(k)(u) for (k =
    0,...,n), (n <= p).  The kth derivative is returned in ders[k].

    Source: The NURBS Book (2nd Ed.), Pg. 76.

    '''

    ders = np.zeros(n + 1)
    ND = np.zeros(n + 1)
    N = np.zeros((p + 1, p + 1))
    if u < U[i] or u >= U[i+p+1]:
        return ders
    for j in xrange(p + 1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j,0] = 1.0
    for k in xrange(1, p + 1):
        if N[0,k-1] == 0.0:
            saved = 0.0
        else:
            saved = ((u - U[i]) * N[0,k-1]) / (U[i+k] - U[i])
        for j in xrange(p - k + 1):
            Uleft, Uright = U[i+j+1], U[i+j+k+1]
            if N[j+1,k-1] == 0.0:
                N[j,k] = saved
                saved = 0.0
            else:
                tmp = N[j+1,k-1] / (Uright - Uleft)
                N[j,k] = saved + (Uright - u) * tmp
                saved = (u - Uleft) * tmp
    ders[0] = N[0,p]
    for k in xrange(1, n + 1):
        for j in xrange(k + 1):
            ND[j] = N[j,p-k]
        for jj in xrange(1, k + 1):
            if ND[0] == 0.0:
                saved = 0.0
            else:
                saved = ND[0] / (U[i+p-k+jj] - U[i])
            for j in xrange(k - jj + 1):
                Uleft = U[i+j+1]
                Uright = U[i+j+p-k+jj+1]
                if ND[j+1] == 0.0:
                    ND[j] = (p - k + jj) * saved
                    saved = 0.0
                else:
                    tmp = ND[j+1] / (Uright - Uleft)
                    ND[j] = (p - k + jj) * (saved - tmp)
                    saved = tmp
        ders[k] = ND[0]
    return ders


def ders_one_basis_fun_v(p, U, i, u, n, num):

    ''' Idem ders_one_basis_fun, vectorized in u.

    '''

    u = np.asfarray(u)
    ders = np.zeros((n + 1, num))
    ND = np.zeros((n + 1, num))
    N = np.zeros((p + 1, p + 1, num))
    cs = np.ones(num, dtype='bool')
    c = u < U[i]; cs[c] = False
    c = u >= U[i+p+1]; cs[c] = False
    NN = N[:,:,cs]
    u = u[cs]
    for j in xrange(p + 1):
        NN[j,0,(u >= U[i+j]) & (u < U[i+j+1])] = 1.0
    for k in xrange(1, p + 1):
        if (U[i+k] - U[i]) != 0.0:
            divs = ((u - U[i]) * NN[0,k-1]) / (U[i+k] - U[i])
        else:
            divs = None
        saved = np.where(NN[0,k-1] == 0.0, 0.0, divs)
        for j in xrange(p - k + 1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            zs = NN[j+1,k-1] == 0.0; nzs = ~ zs
            NN[j,k] = np.where(zs, saved, NN[j,k])
            saved = np.where(zs, 0.0, saved)
            if Uright - Uleft != 0.0:
                divs = NN[j+1,k-1] / (Uright - Uleft)
            else:
                divs = None
            tmp = np.where(nzs, divs, 0.0)
            NN[j,k] = np.where(nzs, saved + (Uright - u) * tmp, NN[j,k])
            saved = np.where(nzs, (u - Uleft) * tmp, saved)
    ders[0,cs] = NN[0,p]
    NDD = ND[:,cs]
    for k in xrange(1, n + 1):
        for j in xrange(k + 1):
            NDD[j] = NN[j,p-k]
        for jj in xrange(1, k + 1):
            if (U[i+p-k+jj] - U[i]) != 0.0:
                divs = NDD[0] / (U[i+p-k+jj] - U[i])
            else:
                divs = None
            saved = np.where(NDD[0] == 0.0, 0.0, divs)
            for j in xrange(k - jj + 1):
                Uleft = U[i+j+1]
                Uright = U[i+j+p-k+jj+1]
                zs = NDD[j+1] == 0.0; nzs = ~ zs
                NDD[j] = np.where(zs, (p - k + jj) * saved, NDD[j])
                saved = np.where(zs, 0.0, saved)
                if Uright - Uleft != 0.0:
                    divs = NDD[j+1] / (Uright - Uleft)
                else:
                    divs = None
                tmp = np.where(nzs, divs, 0.0)
                NDD[j] = np.where(nzs, (p - k + jj) * (saved - tmp), NDD[j])
                saved = np.where(nzs, tmp, saved)
        ders[k,cs] = NDD[0]
    return ders
