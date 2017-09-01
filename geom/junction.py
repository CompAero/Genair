import numpy as np
from scipy.optimize import brute, fmin

from part           import Part

from nurbs.curve    import (ControlPolygon,
                            Curve,
                            make_linear_curve,
                            param_to_arc_length,
                            make_composite_curve,
                            arc_length_to_param)
from nurbs.fit      import (local_curve_interp,
                            global_curve_approx_fixedn,
                            global_surf_approx_fixednm)
from nurbs.knot     import (normalize_knot_vec,
                            KnotOutsideKnotVectorRange)
from nurbs.nurbs    import obj_mat_to_4D
from nurbs.point    import (Point,
                            obj_mat_to_points)
from nurbs.surface  import (ControlNet,
                            Surface,
                            make_composite_surface,
                            surface_surface_intersect,
                            make_ruled_surface)
from nurbs.util     import (distance,
                            construct_flat_grid)

from plot.figure    import Figure as draw


__all__ = ['TrimmedJunction', 'WatertightJunction']


class TrimmedJunction(Part):

    '''


    '''

    def __init__(self, wing, S, half, tip):

        '''

        half = the half of the TrimmedJunction we are interested in (0: lower,
               1: upper, 2: both)

        '''

        self.wing = wing # the Wing object (no copy!)
        self._nurbs = wing.nurbs.copy()

        self.S = S.copy() # the Surface to intersect the Wing with

        self.half = half # the half of interest, normally Wing.half
        self.tip = tip # the end of interest

        self.ICs = [None, None] # the intersection (trim) Curves

    def attach(self):

        '''

        '''

        wing, S = self.wing, self.S
        U, V = S.U
        b0 = S.extract(U[ 0], 0)
        b1 = S.extract(U[-1], 0)
        b2 = S.extract(V[ 0], 1)
        b3 = S.extract(V[-1], 1)
        fig = draw(b0, b1, b2, b3, wing)
        fig.m.toggle_mode('_AttachWingJunctionMode')
        fig.m.mode.junction = self
        return fig

    def extend(self, l=0.0, show=True):

        '''

        '''

        wing, nurbs = self.wing, self._nurbs
        if l != 0.0:
            u = arc_length_to_param(wing.QC, l)
            ne = nurbs.extend(u, 1, end=self.tip)
            nn = (ne, nurbs) if not self.tip else (nurbs, ne)
            nc = make_composite_surface(nn, reorient=False)
            normalize_knot_vec(nc.U[1])
            wing._halve(nc)
        if show:
            return draw(self.S, self.wing)

    def intersect(self, CRT=5e-3, ANT=1e-2, show=True):

        '''

        '''

        P0 = find_LE_starting_point(self.wing.LE, self.S)
        print('geom.junction.TrimmedJunction.intersect :: '
              'LE starting point found: {}'.format(P0))

        IP = []
        for half, h in zip((0, 1), self.wing.halves):
            if self.half in (2, half):

                args = h, self.S, P0, CRT, ANT
                Q, stuv = surface_surface_intersect(*args)

                if show:
                    Qw = obj_mat_to_4D(Q)
                    IP += obj_mat_to_points(Qw).tolist()

                n = stuv.shape[0] - 1
                z = np.zeros((n + 1, 1))
                st = stuv[:,:2]
                Q = np.hstack((st, z))
                U, Pw = local_curve_interp(n, Q)
                IC = Curve(ControlPolygon(Pw=Pw), (3,), (U,))
                self.ICs[half] = IC

        if show:
            return draw(self.S, self.wing, *IP)

    def trim(self, show=True):

        '''

        '''

        if not any(self.ICs):
            raise Exception

        for half, h, IC in zip((0, 1), self.wing.halves, self.ICs):
            if not self.half in (2, half):
                continue
            U, V = h.U
            if not self.tip:
                jnc1 = self.wing.junctions[1]
                if (not jnc1) or (jnc1 and not jnc1.ICs[half]):
                    P0 = IC.cobj.cpts[ 0]
                    P1 = IC.cobj.cpts[-1]
                    P2 = Point(U[-1], V[-1])
                    P3 = Point(U[ 0], V[-1])
                    C0 = IC
                    C1 = make_linear_curve(P1, P2)
                    C2 = make_linear_curve(P2, P3)
                    C3 = make_linear_curve(P3, P0)
                else:
                    IC1 = jnc1.ICs[half]
                    P0 = IC.cobj.cpts[ 0]
                    P1 = IC.cobj.cpts[-1]
                    P2 = IC1.cobj.cpts[-1]
                    P3 = IC1.cobj.cpts[ 0]
                    C0 = IC
                    C1 = make_linear_curve(P1, P2)
                    C2 = IC1.reverse()
                    C3 = make_linear_curve(P3, P0)
            else:
                jnc0 = self.wing.junctions[0]
                if (not jnc0) or (jnc0 and not jnc0.ICs[half]):
                    P0 = Point(U[ 0], V[0])
                    P1 = Point(U[-1], V[0])
                    P2 = IC.cobj.cpts[-1]
                    P3 = IC.cobj.cpts[ 0]
                    C0 = make_linear_curve(P0, P1)
                    C1 = make_linear_curve(P1, P2)
                    C2 = IC
                    C3 = make_linear_curve(P3, P0)
                else:
                    IC0 = jnc0.ICs[half]
                    P0 = IC0.cobj.cpts[ 0]
                    P1 = IC0.cobj.cpts[-1]
                    P2 = IC.cobj.cpts[-1]
                    P3 = IC.cobj.cpts[ 0]
                    C0 = IC0
                    C1 = make_linear_curve(P1, P2)
                    C2 = IC.reverse()
                    C3 = make_linear_curve(P3, P0)
            IC = make_composite_curve([C0, C1, C2, C3], remove=False)
            h.trim(IC)

        if show:
            return draw(self.S, self.wing)

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        pass

    def _draw(self):
        ''' See Part._draw. '''
        pass


class WatertightJunction(Part):
    pass


# UTILITIES


def find_LE_starting_point(LE, S):
    ''' '''

    def dist(tuv):
        try:
            t, u, v = tuv
            return distance(LE.eval_point(t),
                            S.eval_point(u, v))
        except KnotOutsideKnotVectorRange:
            return np.inf

    U, V = S.U
    bnds = [(0, 1), (U[0], U[-1]), (V[0], V[-1])]
    tuv = brute(dist, bnds, finish=None)
    t, u, v = fmin(dist, tuv, xtol=0.0, ftol=0.0, disp=False)
    P0 = LE.eval_point(t)
    S0 = S.eval_point(u, v)
    if not np.allclose(P0 - S0, 0.0):
        raise Exception
    return P0


# TODO


#class WatertightJunction(Part):
#
#    '''
#
#    Intended usage
#    --------------
#    >>> jnc = WatertightJunction(wing, S)
#    >>> jnc.attach()
#    >>> jnc.extend()
#    >>> jnc.intersect()
#    >>> jnc.design()
#    >>> jnc.approximate()
#
#    '''
#
#    def __init__(self, wing, S, half, tip):
#
#        '''
#
#        '''
#
#        self.wing = wing
#        self._nurbs = wing.nurbs.copy()
#
#        self.S = S.copy(); U, V = self.S.U
#        normalize_knot_vec(U)
#        normalize_knot_vec(V)
#
#        self.half = half
#        self.tip = tip
#
#        self.stuv = [None, None]
#
#        self.parametric_surfaces = []
#        self.surfaces = []
#        self.halves = []
#
#    def attach(self):
#
#        '''
#
#        '''
#
#        wing, S = self.wing, self.S
#        U, V = S.U
#        b0 = S.extract(U[ 0], 0)
#        b1 = S.extract(U[-1], 0)
#        b2 = S.extract(V[ 0], 1)
#        b3 = S.extract(V[-1], 1)
#        fig = draw(b0, b1, b2, b3, wing)
#        fig.m.toggle_mode('_AttachWingJunctionMode')
#        fig.m.mode.junction = self
#        return fig
#
#    def extend(self, l=0.0, show=True):
#
#        '''
#
#        '''
#
#        wing, nurbs = self.wing, self._nurbs
#        if l != 0.0:
#            u = arc_length_to_param(wing.QC, l)
#            ne = nurbs.extend(u, 1, end=self.tip)
#            nn = (ne, nurbs) if not self.tip else (nurbs, ne)
#            nc = make_composite_surface(nn, reorient=False)
#            normalize_knot_vec(nc.U[1])
#            wing._halve(nc)
#        if show:
#            return draw(self.S, self.wing)
#
#    def intersect(self, CRT=5e-3, ANT=1e-2, show=True):
#
#        '''
#
#        '''
#
#        P0 = find_LE_starting_point(self.wing.LE, self.S)
#        print('geom.junction.TrimmedJunction.intersect :: '
#              'LE starting point found: {}'.format(P0))
#
#        IP = []
#        for half, h in zip((0, 1), self.wing.halves):
#            if self.half in (2, half):
#
#                args = h, self.S, P0, CRT, ANT
#                Q, stuv = surface_surface_intersect(*args)
#
#                if show:
#                    Qw = obj_mat_to_4D(Q)
#                    IP += obj_mat_to_points(Qw).tolist()
#
#                self.stuv[half] = stuv
#
#        if show:
#            return draw(self.S, self.wing, *IP)
#
#    def design(self, ncp=100, show=True):
#
#        '''
#
#        '''
#
#        self.parametric_surfaces = []
#        self.surfaces = []
#
#        ICst = []
#        for stuv in self.stuv:
#            n = stuv.shape[0] - 1
#            z = np.zeros((n + 1, 1))
#            st = stuv[:,:2]
#            Q = np.hstack((st, z))
#            U, Pw = local_curve_interp(n, Q)
#            IC = Curve(ControlPolygon(Pw=Pw), (3,), (U,))
#            ICst.append(IC)
#
#        Vs = []
#        for IC, stuv in zip(ICst, self.stuv):
#            st = stuv[:,:2]
#            vs = [0.0]
#            for s, t in st[1:-1]:
#                v, = IC.project((s, t, 0), vs[-1])
#                vs.append(v)
#            vs.append(1.0)
#            Vs.append(np.array(vs))
#
#        ICst, ICuv = [], []
#        for stuv, vs in zip(self.stuv, Vs):
#
#            n = stuv.shape[0] - 1
#            z = np.zeros((n + 1, 1))
#
#            st = stuv[:,:2]
#            Q = np.hstack((st, z))
#            U, Pw = global_curve_approx_fixedn(n, Q, 2, ncp, vs)
#            IC = Curve(ControlPolygon(Pw=Pw), (2,), (U,))
#            ICst.append(IC)
#
#            uv = stuv[:,2:]
#            Q = np.hstack((uv, z))
#            U, Pw = global_curve_approx_fixedn(n, Q, 2, ncp, vs)
#            IC = Curve(ControlPolygon(Pw=Pw), (2,), (U,))
#            ICuv.append(IC)
#
#        V = self.wing.halves[0].U[1]
#        v = V[0] if self.tip else V[-1]
#        p0, p1 = Point(0, v, 0), Point(1, v, 0)
#        C = make_linear_curve(p0, p1)
#        self._parametric_halves = []
#        for IC in ICst:
#            cc = (C, IC) if self.tip else (IC, C)
#            ph = make_ruled_surface(*cc)
#            self._parametric_halves.append(ph)
#
#        p0, p1 = Point(0,0), Point(0,1)
#        p2, p3 = Point(1,0), Point(1,1)
#
#        c0 = make_linear_curve(p0, p2)
#        c1 = make_linear_curve(p1, p3)
#        c2 = make_linear_curve(p0, p1)
#        c3 = make_linear_curve(p2, p3)
#
#        if show:
#            LE, TE = self.stuv[0][0,2:], self.stuv[0][-1,2:]
#            LE, TE = [Point(E[0], E[1]) for E in LE, TE]
#            return draw(c0, c1, c2, c3, LE, TE, *ICuv)
#
#    def approximate(self, ppu=200, ppc=10, show=True):
#
#        '''
#
#        '''
#
#        if not self.parametric_surfaces:
#            raise Exception
#
#        def map_surface(ps, S, i=0):
#            #lu, lv = estimate_side_lengths(ps, S)
#
#            #r, s = int(ppu * lu), int(ppu * lv)
#            #n, m = r / ppc, s / ppc
#            # tmp
#            if i == 0:
#                r, s = 1200, 150
#                n, m = 50, 8
#            elif i == 1:
#                r, s = 150, 150
#                n, m = 8, 8
#            elif i == 2:
#                r, s = 1200, 150
#                n, m = 50, 8
#
#            print('geom.junction.WatertightJunction.approximate :: '
#                    'eval: {}, fit: {}'.format((r, s), (n, m)))
#
#            U, V = ps.U
#            uk, vl = (np.linspace(U[0], U[-1], r),
#                      np.linspace(V[0], V[-1], s))
#            st = construct_flat_grid((uk, vl))
#
#            uv = ps.eval_points(*st)
#            uv = uv[0,:], uv[1,:]
#
#            Q = S.eval_points(*uv).T
#            Q = Q.reshape((r, s, 3))
#
#            args = r - 1, s - 1, Q, 3, 3, n, m, uk, vl
#            U, V, Pw = global_surf_approx_fixednm(*args)
#            S = Surface(ControlNet(Pw=Pw), (3,3), (U,V))
#            S.colorize()
#            return S
#
#        # tmp
#        ss = []
#        for ps in self.parametric_surfaces[:2]:
#            s = map_surface(ps, self.S, 0)
#            ss.append(s)
#        for ps in self.parametric_surfaces[2:]:
#            s = map_surface(ps, self.S, 1)
#            ss.append(s)
#
#        hs = []
#        for ps, h in zip(self._parametric_halves, self.wing.halves):
#            h = map_surface(ps, h, 2)
#            hs.append(h)
#
#        self.surfaces = ss
#        self.halves = hs
#        if show:
#            return draw(self)
#
#    def _glue(self, parent=None):
#        ''' See Part._glue. '''
#        pass
#
#    def _draw(self):
#        ''' See Part._draw. '''
#        d = self.surfaces or [self.S]
#        d += self.halves or [self.wing]
#        return d
#
#
#def estimate_side_lengths(ps, S):
#    ''' '''
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
#    us, vs = (np.linspace(0, 1, r),
#              np.linspace(0, 1, r))
#
#    Cs = [map_curve(us, np.repeat(0, r)),
#          map_curve(us, np.repeat(1, r)),
#          map_curve(np.repeat(0, r), vs),
#          map_curve(np.repeat(1, r), vs)]
#
#    ls = [param_to_arc_length(c) for c in Cs]
#    lu, lv = ls[:2], ls[2:]
#    return np.average(lu), np.average(lv)
