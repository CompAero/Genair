import numpy as np

import util


# Many functions in this module rely on the fact the given knot vectors
# are "clean", i.e. any given knot is at least +/- a certain tolerance
# from any other knot, except for end knots and knots with
# multiplicities greater than 1.  This is achieved by making sure all
# knot vectors are rounded to NDEC decimal places.
NDEC = 5


# CLEANING KNOTS


def clean_knot(u):
    ''' Clean the single knot u, i.e. simply round it to NDEC decimal
    places.  Ideally, this should be called every time a knot is to be
    inserted, removed, etc., from a knot vector. '''
    return np.round(u, decimals=NDEC)

def clean_knot_vec(U):
    ''' Clean the entire knot vector U, i.e. ensure there are no close
    knots.  For example, this is called automatically by a NURBSObject
    upon instantiation (IN-PLACE). '''
    Ui, ind = np.unique(U.round(decimals=NDEC), return_inverse=True)
    U[:] = Ui[ind]


# CHECKING KNOTS


def check_knot(U, u):
    ''' Check if u is within the bounds of U, assuming U is clean.
    Here, the returned knot must not be rounded. '''
    ur = np.round(u, decimals=NDEC)
    if U[0] == ur:
        return U[0]
    if U[-1] == ur:
        return U[-1]
    if not U[0] < u < U[-1]:
        raise KnotOutsideKnotVectorRange(U, u)
    return u

def check_knot_v(U, u):
    ''' Idem check_knot, vectorized in u. '''
    u = np.asfarray(u)
    ur = np.round(u, decimals=NDEC)
    u[U[0] == ur] = U[0]; u[U[-1] == ur] = U[-1]
    if (u < U[0]).any() or (u > U[-1]).any():
        raise KnotOutsideKnotVectorRange(U, u)
    return u

def check_knot_vec(n, p, U):
    ''' Perform some consistency checks on the knot vector U, assuming U
    is clean. '''
    m = U.size - 1
    if m != n + p + 1:
        raise NonMatchingKnotVectorLength(m, n, p)

    Ul = U.tolist()
    nl, nr = [Ul.count(U[i]) for i in (0, -1)]
    if nl != p + 1 or nr != p + 1:
        raise UnclampedKnotVector(nl, nr, p, U)

    if (U != np.sort(U)).any():
        raise NonStrictlyIncreasingKnotVector(U)

    mult = find_int_mult_knot_vec(p, U)
    multv = np.array(mult.values())
    if (multv > p).any():
        raise InteriorKnotMultiplicityGreaterThanOrder(p, mult)


# COMPUTING PARAMETERS


def chord_length_param(n, Q):
    ''' Compute parameter values based on chord length parameterization.
    Assumes that the parameters lie in the range u in [0, 1]. '''
    Ub = np.zeros(n + 1)
    clk = np.zeros(n + 1)
    for k in xrange(1, n + 1):
        clk[k] = util.norm(Q[k] - Q[k-1])
    d = np.sum(clk)
    for k in xrange(1, n):
        Ub[k] = Ub[k-1] + clk[k] / d
    Ub[n] = 1.0
    return Ub

def centripetal_param(n, Q):
    ''' Compute parameter values based on the centripetal method.
    Assumes that the parameters lie in the range u in [0, 1]. '''
    Ub = np.zeros(n + 1)
    clk = np.zeros(n + 1)
    for k in xrange(1, n + 1):
        clk[k] = np.sqrt(util.norm(Q[k] - Q[k-1]))
    d = np.sum(clk)
    for k in xrange(1, n):
        Ub[k] = Ub[k-1] + clk[k] / d
    Ub[n] = 1.0
    return Ub

def skinned_param(n, m, Q):
    ''' Compute parameter values suitable for skinning. '''
    Ub = np.zeros(m + 1)
    clk = np.zeros((n + 1, m + 1))
    di = np.zeros(n + 1)
    Pki = Q[0]
    for k in xrange(1, m + 1):
        Pk = Q[k]
        for i in xrange(n + 1):
            clk[i,k] = util.norm(Pk[i] - Pki[i-1])
        Pki = Pk
    for i in xrange(n + 1):
        di[i] = np.sum(clk[i])
    for k in xrange(1, m):
        sumi = 0.0
        for i in xrange(n + 1):
            sumi += clk[i,k] / di[i]
        Ub[k] = Ub[k-1] + 1.0 / (n + 1) * sumi
    Ub[m] = 1.0
    return Ub


# BUILDING KNOT VECTORS


def uni_knot_vec(n, p):
    ''' Construct a uniform and normalized knot vector, i.e. all
    interior knots are equally spaced and lie in [0, 1]. '''
    U = np.zeros(n + p + 2)
    for j in xrange(1, n - p + 1):
        U[j+p] = float(j) / (n - p + 1)
    U[-p-1:] = 1.0
    clean_knot_vec(U)
    return U

def averaging_knot_vec(n, p, Ub):
    ''' Construct a knot vector based on averaging, a method that
    reflects the distribution of the parameter values Ub. '''
    U = np.zeros(n + p + 2)
    for j in xrange(1, n - p + 1):
        U[j+p] = np.sum(Ub[j:j+p]) / p
    U[:p+1], U[-p-1:] = Ub[0], Ub[-1]
    clean_knot_vec(U)
    return U

def end_derivs_knot_vec(n, p, k, l, Ub):
    ''' Construct a knot vector suitable for an interpolation problem
    with end derivatives specified.  First, a knot vector for
    interpolation without end condition is computed, then knots are
    added into the first and last spans until the required number of
    knots are obtained. '''
    U = np.zeros(n + k + l + p + 2)
    m = n + k + l
    for i in xrange(p + 1):
        U[i] = Ub[0]
        U[m+i+1] = Ub[n]
    istt, iend = 1 - k, n - p + l
    r = p
    for i in xrange(istt, iend + 1):
        js, je = max(0, i), min(n, i + p - 1)
        r += 1
        sm = 0
        for j in xrange(js, je + 1):
            sm += Ub[j]
        U[r] = sm / (je - js + 1)
    clean_knot_vec(U)
    return U

def approximating_knot_vec(n, p, r, Ub):
    ''' Construct a knot vector suitable for approximation problems.
    Each knot span is guaranteed to contain at least one ubk. '''
    U = np.zeros(n + p + 2)
    d = float(r + 1) / (n - p + 1)
    for j in xrange(1, n - p + 1):
        i = int(j * d)
        alf = j * d - i
        U[p+j] = (1.0 - alf) * Ub[i-1] + alf * Ub[i]
    U[:p+1], U[-p-1:] = Ub[0], Ub[-1]
    clean_knot_vec(U)
    return U

def approximating_knot_vec_end(n, p, r, k, l, Ub):
    ''' Construct a knot vector especially for approximation problems
    with end derivatives specified. '''
    U = np.zeros(n + p + 2)
    for i in xrange(p + 1):
        U[i], U[n+i+1] = Ub[0], Ub[r]
    nc = n - k - l
    w = np.zeros(nc + 1)
    inc = float(r + 1) / float(nc + 1)
    low = high = 0
    d = -1
    for i in xrange(nc + 1):
        d += inc
        high = int(np.floor(d + 0.5))
        sum = 0.0
        for j in xrange(low, high + 1):
            sum += Ub[j]
        w[i] = sum / (high - low + 1)
        low = high + 1
    it = 1 - k
    ie = nc - p + l
    r = p
    for i in xrange(it, ie + 1):
        js = max(0, i)
        je = min(nc, i + p - 1)
        r += 1
        sum = 0
        for j in xrange(js, je + 1):
            sum += w[j]
        U[r] = sum / (je - js + 1)
    clean_knot_vec(U)
    return U


# MANIPULATING KNOT VECTORS


def normalize_knot_vec(U):
    ''' Normalize all knots (or parameter values) to [0, 1] (IN-PLACE).
    '''
    u0, um = U[0], U[-1]
    U[:] = (np.asfarray(U) - u0) / (um - u0)
    clean_knot_vec(U)

def remap_knot_vec(U, u0, um):
    ''' Remap all knots (or parameter values) to [u0, um] (IN-PLACE).
    '''
    normalize_knot_vec(U)
    U[:] = u0 + np.asfarray(U) * (um - u0)
    clean_knot_vec(U)


# MULTIPLICITIES


def find_mult_knot_vec(U):
    ''' Find the multiplicities of each knot in U, including the end
    knots. '''
    mult = {}.fromkeys(np.unique(U), 0)
    for u in U:
        mult[u] += 1
    return mult

def find_int_mult_knot_vec(p, U):
    ''' Idem to find_mult_knot_vec, but count the multiplicities of the
    interior knots only. '''
    U = U[p+1:-p-1]
    return find_mult_knot_vec(U)


# MIDPOINTS


def midpoints_knot_vec(U):
    ''' Return all knot spans' midpoint. '''
    U = np.unique(U)
    return (U[:-1] + U[1:]) / 2.0

def midpoint_longest_knot_vec(U):
    ''' Return the midpoint of the longest knot span. '''
    spans = U[1:] - U[:-1]
    m = np.argmax(spans)
    return U[m] + spans[m] / 2.0

def midpoints_longest_knot_vec(U, n):
    ''' Return n knots that shall refine U at its longest n mid spans.
    '''
    X = []
    while len(X) < n:
        x = midpoint_longest_knot_vec(U)
        X.append(x)
        U = np.append(U, x); U.sort()
    return np.sort(X)

def midpoint_longest_knot_vec_ins(U, n):
    ''' Return U with the midpoint of its longest n knot spans inserted.
    '''
    X = midpoints_longest_knot_vec(U, n)
    U = np.append(U, X)
    return np.sort(U)


# SEGMENTING KNOT VECTORS


def segment_knot_vec_int(p, S):
    ''' Refine the given interval S until it contains at least (p + 2)
    knots. '''
    r = len(S)
    if r < p - 2:
        S = midpoint_longest_knot_vec_ins(S, p + 2 - r)
    return S

def segment_knot_vec(n, p, U, u1, u2):
    ''' Find the missing knots that would ensure there are at least (p +
    2) knots between u1 and u2, inclusively.  No matter what, u1 and/or
    u2 are included if they are not already in U. '''
    Ui = U.copy()
    for u in (u1, u2):
        if u not in U:
            U = np.append(U, u)
            n += 1
    U.sort()
    k1 = np.where(U==u1)[0] if U[ 0] != u1 else p
    k2 = np.where(U==u2)[0] if U[-1] != u2 else n + 1
    S = segment_knot_vec_int(p, U[k1:k2+1])
    return np.setdiff1d(S, Ui)


# MISSING KNOTS


def missing_knot_vec(V, U):
    ''' Return all knots that are in V but not in U, assuming U and V
    are cleaned. '''
    multV = find_mult_knot_vec(V)
    multU = find_mult_knot_vec(U)
    U = np.zeros(0)
    for v, mv in sorted(multV.items()):
        mu = multU.get(v, 0)
        U = np.append(U, np.array((mv - mu) * [v]))
    return U


# MERGING KNOT VECTORS


def merge_knot_vecs(*Us):
    ''' Merge all knot vectors Us in a new knot vector U, e.g. given U1
    and U2, uj is in U if it is in either U1 or U2.  The maximum
    multiplicity of uj in U1 or U2 is also carried over to U.  Assumes
    all knot vectors are cleaned. '''
    mults = []
    for U in Us:
        mults.append(find_mult_knot_vec(U))
    uk = np.hstack(Us)
    mult = {}.fromkeys(np.unique(uk), 0)
    for u in mult:
        mult[u] = np.max([m.get(u, 0) for m in mults])
    U = np.zeros(0)
    for v, m in sorted(mult.items()):
        U = np.append(U, np.array(m * [v]))
    return U


# EXCEPTIONS


class KnotVectorException(Exception):
    pass

class NonMatchingKnotVectorLength(KnotVectorException):
    pass

class NonStrictlyIncreasingKnotVector(KnotVectorException):
    pass

class UnclampedKnotVector(KnotVectorException):
    pass

class InteriorKnotMultiplicityGreaterThanOrder(KnotVectorException):
    pass

class KnotOutsideKnotVectorRange(KnotVectorException):
    pass
