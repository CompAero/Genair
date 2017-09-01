import numpy as np
from scipy.integrate import quad

from part            import Part

from nurbs.knot      import normalize_knot_vec
from nurbs.surface   import (make_ruled_surface,
                             make_revolved_surface_nrat,
                             make_revolved_surface_rat)
from nurbs.transform import scale
from nurbs.util      import intersect_line_plane

from plot.figure     import Figure as draw


__all__ = ['WingStructure', 'Nacelle', 'Cabin',
           'split_wing', 'get_Lq_from_design_Cls']


class WingStructure(Part):

    def __init__(self, wing):

        self.wing = wing

        self.spars = []
        self.ribs = []

    def generate_spars(self, spar0=0.05, spar1=0.6, subspar=None):

        wi = self.wing
        h0, h1 = wi.halves

        c0_fore = h0.extract(spar0, 0)
        c1_fore = h1.extract(spar0, 0)

        c0_aft = h0.extract(spar1, 0)
        c1_aft = h1.extract(spar1, 0)

        spar_fore = make_ruled_surface(c0_fore, c1_fore)
        spar_aft = make_ruled_surface(c0_aft, c1_aft)

        self.spars = [spar_fore, spar_aft]

        if subspar is not None:
            c0_sub = h0.extract(subspar, 0)
            c1_sub = h1.extract(subspar, 0)
            spar_sub = make_ruled_surface(c0_sub, c1_sub)
            self.spars.append(spar_sub)

    def generate_ribs(self, spar0=0.05, spar2=0.9, nribs=10):

        wi = self.wing
        h0, h1 = wi.halves

        vs = np.linspace(0, 1, nribs)
        ribs = []
        for v in vs:
            c0 = h0.extract(v, 1)
            c1 = h1.extract(v, 1)

            dummy, c0 = c0.split(spar0)
            dummy, c1 = c1.split(spar0)

            c0, dummy = c0.split(spar2)
            c1, dummy = c1.split(spar2)

            rib = make_ruled_surface(c0, c1)
            ribs.append(rib)

        self.ribs = ribs

    def _glue(self, parent=None):
        super(WingStructure, self)._glue(parent)
        g = self.spars + self.ribs
        return g

    def _draw(self):
        return self.spars + self.ribs


class Nacelle(Part):

    def __init__(self, airfoil):

        self.airfoil = airfoil.copy()

        self.nurbs = None
        self.halves = []

        self.D = 0.0

    def revolve(self, D, ang=0.0, show=True):

        n = self.airfoil.nurbs.reverse()
        n.mirror(N=[0,0,1])
        n.translate([0,0,-D/2.0])

        S, T = [0,0,0], [-1,0,0]
        nurbs = make_revolved_surface_rat(n, S, T, 360)
        #nurbs = make_revolved_surface_nrat(n, S, T, 360)
        #if ang != 0.0:
        #    rad = np.deg2rad(ang)
        #    N = [np.cos(rad), 0, np.sin(rad)]
        #    n, dummy = nurbs.cobj.n
        #    for i in xrange(n + 1):
        #        Pw = nurbs.cobj.Pw[i]
        #        L0 = Pw[0,:3]
        #        P = intersect_line_plane(L0, [1,0,0], [0,0,0], N)
        #        sf = (L0[0] - P[0]) / L0[0]
        #        scale(Pw, sf, L0, (1,0,0))
        hs = nurbs.split(0.5, 1)

        self.nurbs = nurbs
        self.halves = hs
        self.D = D
        if show:
            return draw(self)

    def _glue(self, parent=None):
        super(Nacelle, self)._glue(parent)
        g = []
        if self.nurbs:
            g += [self.nurbs] + self.halves
        return g

    def _draw(self):
        if self.nurbs:
            return self.halves
        return []


class Cabin(Part):

    def __init__(self, ss):
        self.surfaces = list(ss)

    def _glue(self, parent=None):
        super(Cabin, self)._glue(parent)
        g = []
        if self.surfaces:
            g += self.surfaces
        return g

    def _draw(self):
        if self.surfaces:
            return self.surfaces
        return []


def split_wing(wing, u):

    w0 = wing.copy()
    w1 = wing.copy()

    (T0, T1), (Bv0, Bv1) = wing.T.split(u), wing.Bv.split(u)
    n0, n1 = wing.nurbs.split(u, 1)

    for c in T0, T1, Bv0, Bv1:
        normalize_knot_vec(c.U[0])
    w0.T, w0.Bv = T0, Bv0
    w1.T, w1.Bv = T1, Bv1

    for s in n0, n1:
        normalize_knot_vec(s.U[1])
    w0._halve(n0)
    w1._halve(n1)

    Tw = wing.Tw
    scx, _, scz = wing.scs
    if Tw:
        Tw0, Tw1 = Tw.split(u)
    if scx:
        scx0, scx1 = scx.split(u)
    if scz:
        scz0, scz1 = scz.split(u)

    if Tw:
        for c in Tw0, Tw1:
            normalize_knot_vec(c.U[0])
        w0.Tw = Tw0
        w1.Tw = Tw1
    if scx:
        for c in scx0, scx1:
            normalize_knot_vec(c.U[0])
        w0.scs[0] = scx0
        w1.scs[0] = scx1
    if scz:
        for c in scz0, scz1:
            normalize_knot_vec(c.U[0])
        w0.scs[2] = scz0
        w1.scs[2] = scz1

    # afternoon hack
    af0 = w0.airfoils[0]
    af1 = af0.copy()
    af1._halve(n0.extract(1, 1))
    af1.transform(show=False)
    af1.glue(); af1.translate([0.25,0,0])
    w0.airfoils[1] = af1
    w1.airfoils[0] = af1

    w0.tip = None
    return w0, w1


def get_Lq_from_design_Cls(wing, Cl0, Cl1):

    wing = wing.copy()

    b = np.abs(wing.T.cobj.cpts[-1].xyz[1] -
               wing.T.cobj.cpts[ 0].xyz[1])
    #wing.glue(); wing.sweep = 0
    #b = param_to_arc_length(wing.T)
    #b = np.abs(b * np.cos(np.deg2rad(wing.dihedral)))
    print b

    # L/q = int_0^b c(y) * Cl(y) dy
    cCl = lambda u: wing.scs[0].eval_point(u)[0] * \
                    ((Cl1 - Cl0) * u + Cl0)
    return b * quad(cCl, 0, 1)[0]


# TODO


#class SplitWingTransition(Part):
#
#    def __init__(self, wing, winglet_fore, winglet_aft):
#
#        self.wing = wing
#        self.winglet_fore = winglet_fore
#        self.winglet_aft = winglet_aft
#
#        self.nurbs = None
#        self.quaters = []
#
#        self._bound()
#
#    def _bound(self):
#        wi = self.wing
#        wl_fore = self.winglet_fore
#        wl_aft = self.winglet_aft
#
#        c0 = wi.nurbs.extract(1, 1)
#        d0 = extract_cross_boundary_deriv(wi.nurbs, 3)
#
#        c1_fore = wl_fore.nurbs.extract(0, 1)
#        d1_fore = extract_cross_boundary_deriv(wl_fore.nurbs, 2)
#
#        c1_aft = wl_aft.nurbs.extract(0, 1)
#        d1_aft = extract_cross_boundary_deriv(wl_aft.nurbs, 2)
#
#        c11, c12 = c1_fore.split(0.5)
#        d11, d12 = d1_fore.split(0.5)
#
#        c10, c13 = c1_aft.split(0.5)
#        d10, d13 = d1_aft.split(0.5)
#
#        for c, d in zip((c10, c11, c12, c13), (d10, d11, d12, d13)):
#            al = param_to_arc_length(c)
#            remap_knot_vec(c.U[0], 0, al)
#            remap_knot_vec(d.U[0], 0, al)
#
#        #for d in d10, d11, d12, d13:
#        #    d.cobj.Pw[:,0] = 0
#
#        x, y, z = (d10.cobj.cpts[-1].xyz + d11.cobj.cpts[0].xyz) / 2.0
#        d10.cobj.cpts[-1].xyzw = x, y, z, 1.0
#        d11.cobj.cpts[ 0].xyzw = x, y, z, 1.0
#
#        x, y, z = (d12.cobj.cpts[-1].xyz + d13.cobj.cpts[0].xyz) / 2.0
#        d12.cobj.cpts[-1].xyzw = x, y, z, 1.0
#        d13.cobj.cpts[ 0].xyzw = x, y, z, 1.0
#
#        c1 = make_composite_curve(c10, c11, c12, c13)
#        d1 = make_composite_curve(d10, d11, d12, d13)
#
#        U, = c1.U; normalize_knot_vec(U)
#        U, = d1.U; normalize_knot_vec(U)
#
#        self.Ck = [c0, c1]
#        self.Dk = [d0, d1]
#
#    def blend(self, sf=0.1):
#        Ck, Dk = self.Ck, self.Dk
#        n, p, U, CDk = make_curves_compatible1(Ck + Dk)
#        Ck, Dk = CDk[:2], CDk[2:]
#        for d in Dk:
#            d.scale(sf)
#        self.nurbs = blend_cubic_bezier(n, p, U, Ck, Dk, 0)
#        return draw(self)
#
#    def decompose(self):
#        n = self.nurbs
#        u1 = self.winglet_fore.airfoil._ule
#        u2 = u1 + self.winglet_aft.airfoil._ule
#        u3 = u1 + 1.0
#
#        s1, s2 = n.split(u1, 0)
#        s2, s3 = s2.split(u2, 0)
#        s3, s4 = s3.split(u3, 0)
#
#        for s in s1, s2, s3, s4:
#            s.colorize()
#            self.quaters.append(s)
#
#        draw(self)
#
#    def _glue(self, parent=None):
#        super(SplitWingTransition, self)._glue(parent)
#        return [self.nurbs] + self.quaters
#
#    def _draw(self):
#        return self.quaters or [self.nurbs]
#
#
#def estimate_side_lengths(ps, S):
#
#    def map_curve(us, vs):
#        uv = ps.eval_points(us, vs)
#        uv = uv[0,:], uv[1,:]
#        Q = S.eval_points(*uv).T
#        args = r - 1, Q, 3, n
#        U, Pw = global_curve_approx_fixedn(*args)
#        return Curve(ControlPolygon(Pw=Pw), (3,), (U,))
#
#    r, n = 100, 10
#
#    U, V = ps.U
#    us, vs = (np.linspace(U[0], U[-1], r),
#              np.linspace(V[0], V[-1], r))
#
#    Cs = [map_curve(us, np.repeat(V[ 0], r)),
#          map_curve(us, np.repeat(V[-1], r)),
#          map_curve(np.repeat(U[ 0], r), vs),
#          map_curve(np.repeat(U[-1], r), vs)]
#
#    ls = [param_to_arc_length(c) for c in Cs]
#    lu, lv = ls[:2], ls[2:]
#    return np.average(lu), np.average(lv)
#
#
#def append_wings(w0, w1):
#
#    w0, w1 = w0.copy(), w1.copy()
#
#    for c in w0.T, w1.T:
#        reparam_arc_length_curve(c)
#
#    for wi in w0, w1:
#        U, = wi.T.U
#        scx, _, scz = wi.scs
#        for c in scx, scz:
#            remap_knot_vec(c.U[0], U[0], U[-1])
#        remap_knot_vec(wi.nurbs.U[1], U[0], U[-1])
#
#    #T = make_composite_curve([w0.T, w1.T], remove=False)
#    #normalize_knot_vec(T.U[0])
#
#    #Bv = make_composite_curve([w0.Bv, w1.Bv], remove=False)
#    #normalize_knot_vec(Bv.U[0])
#
#    scx = make_composite_curve([w0.scs[0], w1.scs[0]], remove=False)
#    normalize_knot_vec(scx.U[0])
#
#    scz = make_composite_curve([w0.scs[2], w1.scs[2]], remove=False)
#    normalize_knot_vec(scz.U[0])
#
#    n = make_composite_surface([w0.nurbs, w1.nurbs], di=1,
#                               reorient=False, remove=False)
#    normalize_knot_vec(n.U[1])
#
#    af0 = w0.airfoils[0]
#    af1 = w1.airfoils[-1] if w1.airfoils[-1] else w1.airfoils[0]
#
#    af0, af1 = af0.copy(), af1.copy()
#    for af in af0, af1:
#        af.transform(show=False)
#
#    wing = w0.copy()
#    wing.airfoils = [af0, af1]
#    #wing.T = T
#    #wing.Bv = Bv
#    wing.scs = (scx, None, scz)
#    wing.nurbs = n
#    wing._halve(n)
#    wing.T = wing.QC
#
#    if w1.tip: wing.tip = w1.tip
#    return wing
