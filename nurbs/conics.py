import numpy as np

import curve
import fit
import knot
import transform
import util


__all__ = ['make_circle_nrat',
           'make_circle_rat',
           'make_ellipse',
           'make_hyperbola',
           'make_parabola']


def make_circle_rat(O, X, Y, r, ths, the):

    ''' Construct, using double knots, a rational quadratic circular arc
    in three-dimensional space of arbitrary sweep angle theta (0 deg <
    theta <= 360 deg).

    In particular, the circular arc is represented exactly by

     C(u) = O + r * cos(u) * X + r * sin(u) * Y    ths <= u <= the

    and is oriented counter-clockwise in the local coordinate system
    defined by O, X and Y.

    Parameters
    ----------
    O = the center of the circular arc
    X = a vector lying in the plane of definition of the circular arc
    Y = a vector in the plane of definition of the circular arc, and
        orthogonal to X
    r = the radius
    ths, the = the start and end angles (in degrees), measured with
               respect to X

    Returns
    -------
    Curve = the circular arc

    Examples
    --------
    >>> O, X, Y = ([0, 1, 0], [1, 0, 0], [0, 0, 1])
    >>> arc = nurbs.tb.make_circle_rat(O, X, Y, 3, 75, 340)

    or

    >>> full = nurbs.tb.make_circle_rat(O, X, Y, 3, 75, 75 + 360)

    Source
    ------
    The NURBS Book (2nd Ed.), Pg. 308.

    '''

    X, Y = [util.normalize(V) for V in (X, Y)]
    if not np.allclose(np.dot(X, Y), 0.0):
        raise NonOrthogonalAxes(X, Y)
    if not r > 0:
        raise ImproperInput(r)
    if the < ths:
        the += 360.0
    theta = the - ths
    if theta <= 90.0:
        narcs = 1
    elif theta <= 180.0:
        narcs = 2
    elif theta <= 270.0:
        narcs = 3
    else:
        narcs = 4
    n = 2 * narcs
    Pw, U = np.zeros((n + 1, 4)), np.zeros(n + 4)
    ths, the, theta = [np.deg2rad(th) for th in (ths, the, theta)]
    dtheta = theta / narcs
    w1 = np.cos(dtheta / 2.0)
    P0 = O + r * np.cos(ths) * X + r * np.sin(ths) * Y
    T0 = - np.sin(ths) * X + np.cos(ths) * Y
    Pw[0] = np.hstack((P0, 1.0))
    index = 0; angle = ths
    for i in xrange(1, narcs + 1):
        angle += dtheta
        P2 = O + r * np.cos(angle) * X + r * np.sin(angle) * Y
        Pw[index+2] = np.hstack((P2, 1.0))
        T2 = - np.sin(angle) * X + np.cos(angle) * Y
        P1 = util.intersect_3D_lines(P0, T0, P2, T2)
        Pw[index+1] = np.hstack((w1 * P1, w1))
        index += 2
        if i < narcs:
            P0, T0 = P2, T2
    j = 2 * narcs + 1
    U[j:] = 1.0
    if narcs == 2:
        U[3] = U[4] = 0.5
    elif narcs == 3:
        U[3] = U[4] = 1.0 / 3.0
        U[5] = U[6] = 2.0 / 3.0
    elif narcs == 4:
        U[3] = U[4] = 0.25
        U[5] = U[6] = 0.5
        U[7] = U[8] = 0.75
    return curve.Curve(curve.ControlPolygon(Pw=Pw), (2,), (U,))


def make_circle_nrat(O, X, Y, r, ths, the, p=3, TOL=1e-5):

    ''' Construct a nonrational approximation to a circular arc in
    three-dimensional space of arbitrary sweep angle theta (0 deg <
    theta <= 360 deg).

    In particular, the circular arc approximates

     C(u) = O + r * cos(u) * X + r * sin(u) * Y    ths <= u <= the

    and is oriented counter-clockwise in the local coordinate system
    defined by O, X and Y.

    Parameters
    ----------
    O = the center of the circular arc
    X = a vector lying in the plane of definition of the circular arc
    Y = a vector in the plane of definition of the circular arc, and
        orthogonal to X
    r = the radius
    ths, the = the start and end angles (in degrees), measured with
               respect to X
    p = the degree of the circular arc
    TOL = the maximum deviation from an exact circular arc

    Returns
    -------
    Curve = the circular arc

    Source
    ------
    Piegl & Tiller, Circle Approximation Using Integral B-Spline Curves,
    Computer-Aided Design, 2003.

    '''

    X, Y = [util.normalize(V) for V in (X, Y)]
    if not np.allclose(np.dot(X, Y), 0.0):
        raise NonOrthogonalAxes(X, Y)
    if not r > 0 or not p > 1 or not ths < the:
        raise ImproperInput(r, ths, the)
    k = p - 1; n = k
    if p == 2:
        n += 1
    ths, the = [np.deg2rad(th) for th in (ths, the)]
    the0 = the
    dth = the - ths
    Q = np.zeros((n + 2, 3))
    ns = 0
    while True:
        ns += 1
        Ds, De = (util.eval_ders_trigo(ths, k),
                  util.eval_ders_trigo(the, k))
        for i in xrange(1, k + 1):
            Ds[i] *= dth**i
            De[i] *= dth**i
        us = np.linspace(ths, the, n + 2)
        for i, u in enumerate(us):
            Q[i] = util.eval_ders_trigo(u, 0)
        U, Pw = fit.global_curve_interp_ders(n + 1, Q, p, k, Ds[1:], k, De[1:])
        nf = Pw.shape[0] - 1
        mU = knot.midpoints_knot_vec(U)
        errs = []
        for m in mU:
            um, = curve.curve_point_projection(nf, p, U, Pw, [0,0,0], ui=m)
            P = curve.rat_curve_point(nf, p, U, Pw, um)
            d = abs(1.0 - util.distance(P, [0,0,0]))
            errs.append(d)
        if max(errs) <= TOL:
            break
        dth = (the0 - ths) / (ns + 1)
        the = ths + dth
    dtheta = np.rad2deg(the0 - ths) / ns
    cpol = curve.ControlPolygon(Pw=Pw)
    arcs = []
    for n in xrange(ns):
        arc = curve.Curve(cpol, (p,), (U,))
        arc.rotate(n * dtheta, L=[0,0,1], Q=[0,0,0])
        arcs.append(arc)
    arc = curve.make_composite_curve(arcs, remove=False)
    arc.scale(r)
    u = 1.0
    for n in xrange(1, ns):
        arc = arc.remove(u, k)[1]
        u += 1.0
    Z = np.cross(X, Y)
    transform.custom(arc.cobj.Pw, R=np.column_stack((X, Y, Z)), T=O)
    return arc


def make_ellipse(O, X, Y, a, b, u0, u2):

    ''' Construct a quadratic elliptical arc in three-dimensional space
    of arbitrary sweep u (0 <= u <= 2 * pi).

    In particular, the elliptical arc is represented by

      C(u) = O + a * cos(u) * X + b * sin(u) * Y    u0 <= u <= u2

    and is oriented counter-clockwise in the local coordinate system
    defined by O, X and Y.

    Parameters
    ----------
    O = the center of the elliptical arc
    X, Y = the major and minor axes
    a, b = the major and minor radii
    u0, u2 = the parameter values of the end points, measured with
             respect to X

    Returns
    -------
    Curve = the elliptical arc

    Examples
    --------
    >>> pi = np.pi
    >>> O, X, Y = ([0, 1, 0], [1, 0, 0], [0, 0, 1])
    >>> arc = nurbs.tb.make_ellipse(O, X, Y, 3, 1, pi / 2, 2 * pi)

    or

    >>> full = nurbs.tb.make_ellipse(O, X, Y, 3, 1, pi / 2, pi / 2 + 2 * pi)

    '''

    def point(u):
        return O + a * np.cos(u) * X + b * np.sin(u) * Y

    def tangent(u):
        return - a * np.sin(u) * X + b * np.cos(u) * Y

    X, Y = [util.normalize(V) for V in (X, Y)]
    if not np.allclose(np.dot(X, Y), 0.0):
        raise NonOrthogonalAxes(X, Y)
    if not u0 < u2:
        raise ImproperInput(u0, u2)
    if not a > b > 0:
        raise ImproperInput(a, b)
    if np.mod(u2 - u0, 2 * np.pi) != 0.0:
        P0, T0, P2, T2, P = calc_input_conic(u0, u2, point, tangent)
        return make_open_conic(P0, T0, P2, T2, P)
    else:
        u2 = u0 + np.pi / 2.0
        P0, T0, P2, T2, P = calc_input_conic(u0, u2, point, tangent)
        return make_full_ellipse(P0, T0, P2, T2, P)


def make_hyperbola(O, X, Y, a, b, u0, u2):

    ''' Construct the right branch of a quadratic hyperbola in
    three-dimensional space of arbitrary sweep u (-infty < u < infty).

    In particular, the hyperbola is represented by

     C(u) = O - a * cosh(u) * X - b * sinh(u) * Y    u0 <= u <= u2

    and is oriented according to the local coordinate system defined by
    O, X and Y.

    Parameters
    ----------
    O = the center of the hyperbola
    X, Y = the transverse and imaginary axes
    a, b = the major and minor radii
    u0, u2 = the parameter values of the end points

    Returns
    -------
    Curve = the hyperbola

    Examples
    --------
    >>> O, X, Y = ([0, 1, 0], [-1, 0, 0], [0, 0, 1])
    >>> arc = nurbs.tb.make_hyperbola(O, X, Y, 1, 2, -2, 2)

    '''

    def point(u):
        return O + a * np.cosh(u) * X + b * np.sinh(u) * Y

    def tangent(u):
        return a * np.sinh(u) * X + b * np.cosh(u) * Y

    X, Y = [util.normalize(V) for V in (X, Y)]
    if not np.allclose(np.dot(X, Y), 0.0):
        raise NonOrthogonalAxes(X, Y)
    if not u0 < u2:
        raise ImproperInput(u0, u2)
    if not a > 0 or not b > 0:
        raise ImproperInput(a, b)
    P0, T0, P2, T2, P = calc_input_conic(u0, u2, point, tangent)
    return make_open_conic(P0, T0, P2, T2, P)


def make_parabola(O, X, Y, a, u0, u2):

    ''' Construct a quadratic parabola in three-dimensional space of
    arbitrary sweep u (-infty < u < infty).

    In particular, the parabola is represented by

        C(u) = O + a * u^2 * X + 2 * a * u * Y    u0 <= u <= u2

    and is oriented according to the local coordinate system defined by
    O, X and Y.

    Parameters
    ----------
    O = the vertex of the parabola
    X, Y = the parabola's axis and its tangent direction at O
    a = the focal distance
    u0, u2 = the parameter values of the end points

    Returns
    -------
    Curve = the parabola

    Examples
    --------
    >>> O, X, Y = ([0, 1, 0], [-1, 0, 0], [0, 0, 1])
    >>> arc = nurbs.tb.make_parabola(O, X, Y, 2, -2, 2)

    '''

    def point(u):
        return O + a * u**2 * X + 2 * a * u * Y

    def tangent(u):
        return 2 * a * u * X + 2 * a * Y

    X, Y = [util.normalize(V) for V in (X, Y)]
    if not np.allclose(np.dot(X, Y), 0.0):
        raise NonOrthogonalAxes(X, Y)
    if not u0 < u2:
        raise ImproperInput(u0, u2)
    if not a > 0:
        raise ImproperInput(a)
    P0, T0, P2, T2, P = calc_input_conic(u0, u2, point, tangent)
    return make_open_conic(P0, T0, P2, T2, P)


# HEAVY LIFTING FUNCTIONS


def make_open_conic(P0, T0, P2, T2, P):

    ''' Construct an arbitrary open conic arc in three-dimensional
    space.  The resulting NURBS curve consists of either one, two, or
    four segments connected with C1 continuity.

    Source: The NURBS Book (2nd Ed.), Pg. 317.

    '''

    P0, T0 = [np.asfarray(V) for V in P0, T0]
    P2, T2 = [np.asfarray(V) for V in P2, T2]
    P = np.asfarray(P)
    P1, w1 = make_one_arc(P0, T0, P2, T2, P)
    if w1 <= -1.0:
        raise ParabolaOrHyperbolaOutsideConvexHull(w1)
    if w1 >= 1.0: # hyperbola or parabola
        nsegs = 1
    else: # ellipse
        if w1 > 0.0 and util.angle(P0, P1, P2) > 60.0:
            nsegs = 1
        elif w1 < 0.0 and util.angle(P0, P1, P2) > 90.0:
            nsegs = 4
        else:
            nsegs = 2
    n = 2 * nsegs
    Pw = np.zeros((n + 1, 4))
    U = np.zeros(n + 4)
    j = 2 * nsegs + 1
    U[j:] = 1.0
    Pw[0] = np.hstack((P0, 1.0))
    Pw[n] = np.hstack((P2, 1.0))
    if nsegs == 1:
        Pw[1] = np.hstack((w1 * P1, w1))
        cpol = curve.ControlPolygon(Pw=Pw)
        return curve.Curve(cpol, (2,), (U,))
    Q1, S, R1, wqr = split_arc(P0, P1, w1, P2)
    if nsegs == 2:
        Pw[2] = np.hstack((S, 1.0))
        Pw[1] = np.hstack((wqr * Q1, wqr))
        Pw[3] = np.hstack((wqr * R1, wqr))
        U[3] = U[4] = 0.5
        cpol = curve.ControlPolygon(Pw=Pw)
        return curve.Curve(cpol, (2,), (U,))
    Pw[4] = np.hstack((S, 1.0))
    w1 = wqr
    HQ1, HS, HR1, wqr = split_arc(P0, Q1, w1, S)
    Pw[2] = np.hstack((HS, 1.0))
    Pw[1] = np.hstack((wqr * HQ1, wqr))
    Pw[3] = np.hstack((wqr * HR1, wqr))
    HQ1, HS, HR1, wqr = split_arc(S, R1, w1, P2)
    Pw[6] = np.hstack((HS, 1.0))
    Pw[5] = np.hstack((wqr * HQ1, wqr))
    Pw[7] = np.hstack((wqr * HR1, wqr))
    for i in xrange(2):
        U[i+3] = 0.25
        U[i+5] = 0.5
        U[i+7] = 0.75
    cpol = curve.ControlPolygon(Pw=Pw)
    return curve.Curve(cpol, (2,), (U,))


def make_full_ellipse(P0, T0, P2, T2, P):

    ''' Construct a full ellipse in three-dimensional space based on
    information given in the form of Group 2: start and end points,
    together with their tangent directions and an additional point.

    Source: The NURBS Book (2nd Ed.), Pg. 318-320.

    '''

    P0, T0 = [np.asfarray(V) for V in P0, T0]
    P2, T2 = [np.asfarray(V) for V in P2, T2]
    P = np.asfarray(P)
    P1, w1 = make_one_arc(P0, T0, P2, T2, P)
    S, T = P0 - P1, P2 - P1
    k = 1.0 / w1**2
    e = k / (k - 1.0) / 2
    a = util.norm(S)**2
    b = np.dot(S, T)
    g = util.norm(T)**2
    d = a * g - b**2
    E = a + g - 2 * b
    ls = np.roots((2 * d, - (k * E + 4 * b), 2 * (k - 1)))
    l1, l2 = np.sort(ls)
    r1, r2 = np.sqrt(e / l1), np.sqrt(e / l2)
    if abs(k / 2 - g * l1) > abs(k / 2 - a * l1):
        xb = k / 2 - g * l1
        yb = b * l1 - k / 2 + 1
    else:
        xb = b * l1 - k / 2 + 1
        yb = k / 2 - a * l1
    r = a * xb**2 + 2 * b * xb * yb + g * yb**2
    x0, y0 = xb / r, yb / r
    Q1 = P1 + (e + r1 * x0) * S + (e + r1 * y0) * T
    Q2 = P1 + (e - r1 * x0) * S + (e - r1 * y0) * T
    U = util.normalize(Q2 - Q1)
    V = np.cross(U, util.normalize(np.cross(S, T)))
    Pw, Uq = np.zeros((8 + 1, 4)), np.zeros(8 + 4)
    C = P1 + e * (S + T)
    Pw[0] = Pw[8] = np.hstack((C + r1 * U, 1.0))
    Pw[1] = np.hstack((w1 * (C + r1 * U + r2 * V), w1))
    Pw[2] = np.hstack((C + r2 * V, 1.0))
    Pw[3] = np.hstack((w1 * (C + r2 * V - r1 * U), w1))
    Pw[4] = np.hstack((C - r1 * U, 1.0))
    Pw[5] = np.hstack((w1 * (C - r1 * U - r2 * V), w1))
    Pw[6] = np.hstack((C - r2 * V, 1.0))
    Pw[7] = np.hstack((w1 * (C - r2 * V + r1 * U), w1))
    for i in xrange(3):
        Uq[i] = 0.0
        Uq[i+9] = 1.0
    for i in xrange(2):
        Uq[i+3] = 0.25
        Uq[i+5] = 0.5
        Uq[i+7] = 0.75
    cpol = curve.ControlPolygon(Pw=Pw)
    return curve.Curve(cpol, (2,), (Uq,))


# UTILITIES


def make_one_arc(P0, T0, P2, T2, P):

    ''' Construct one rational Bezier conic arc.  Since w1 can be
    negative or P1 infinite, this algorithm handles any conic arc except
    a full ellipse.  It is thus adequate for parabolic and hyperbolic
    arcs, and for elliptical arcs for which w1 > 0 and whose sweep angle
    is not too large.

    Source: The NURBS Book (2nd Ed.), Pg. 314.

    '''

    V02 = P2 - P0
    try:
        P1 = util.intersect_3D_lines(P0, T0, P2, T2)
        V1P = P - P1
        Q = util.intersect_3D_lines(P1, V1P, P0, V02)
        a = np.sqrt(util.distance(P0, Q) / util.distance(Q, P2))
        u = a / (1.0 + a)
        num = ((1.0 - u)**2 * np.dot(P - P0, P1 - P) +
                u**2 * np.dot(P - P2, P1 - P))
        den = 2.0 * u * (1.0 - u) * np.dot(P1 - P, P1 - P)
        w1 = num / den
    except util.ParallelLines: # infinite control point
        Q = util.intersect_3D_lines(P, T0, P0, V02)
        a = np.sqrt(util.distance(P0, Q) / util.distance(Q, P2))
        u = a / (1.0 + a)
        b = 2.0 * u * (1 - u)
        b = util.distance(P, Q) * (1.0 - b) / b
        P1 = b * util.normalize(T0)
        w1 = 0.0
    return P1, w1


def split_arc(P0, P1, w1, P2):

    ''' Split the arc P0P1P2 at u = 1/2 (using the deCasteljau
    algorithm), then reparameterize so that the end weights are 1 for
    both of the two new segments.

    Source: The NURBS Book (2nd Ed.), Pg. 315-317.

    '''

    if w1 == 0.0:
        Q1, R1 = P0 + P1, P2 + P1
        S = (Q1 + R1) / 2.0
        wqr = np.sqrt(2.0) / 2.0
        return Q1, S, R1, wqr
    else:
        Q1 = (P0 + w1 * P1) / (1.0 + w1)
        R1 = (w1 * P1 + P2) / (1.0 + w1)
        S = (Q1 + R1) / 2.0
        wqr = np.sqrt((1.0 + w1) / 2.0)
        return Q1, S, R1, wqr


def calc_input_conic(u0, u2, point, tangent):

    ''' Calculate the start and end points (P0, P2) of a conic section,
    together with the tangent directions at those two points (T0, T2),
    plus one additional point on the arc P.

    '''

    P0, T0 = point(u0), tangent(u0)
    P2, T2 = point(u2), tangent(u2)
    up = (u0 + u2) / 2.0
    P = point(up)
    return P0, T0, P2, T2, P


# EXCEPTIONS


class ConicsException(Exception):
    pass

class ImproperInput(ConicsException):
    pass

class NonOrthogonalAxes(ConicsException):
    pass

class ParabolaOrHyperbolaOutsideConvexHull(ConicsException):
    pass
