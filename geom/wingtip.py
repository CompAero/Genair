import numpy as np

from part          import Part

from nurbs.conics  import make_one_arc
from nurbs.curve   import (ControlPolygon,
                           Curve,
                           make_curves_compatible1)
from nurbs.fit     import (global_curve_interp,
                           refit_curve,
                           refit_surface)
from nurbs.knot    import remap_knot_vec
from nurbs.nurbs   import obj_mat_to_4D
from nurbs.surface import (make_gordon_surface,
                           make_composite_surface)
from nurbs.util    import (normalize,
                           point_to_plane)

from plot.figure   import Figure as draw


__all__ = ['Wingtip']


class Wingtip(Part):

    ''' Create a Wingtip to fill the gap located at the tip of a Wing.

    '''

    def __init__(self, wing, half=2):

        ''' Instantiate a Wingtip.

        Parameters
        ----------
        wing = the Wing whose tip will be filled
        half = the half of the Wingtip we are interested in (0: lower,
               1: upper, 2: both)

        '''

        self.Ck = [] # the u-directional Curves
        self.Cl = [] # the v-directional Curves

        self.wing = wing # the Wing object

        self.nurbs = None # the Gordon Surface
        self.halves = [None, None] # the Gordon Surface, halved at the LE
        self.half = half # the half of interest

        self._hs = [None, None] # copies of the two halves

    def _network(self, l, dl, xb, ul, vk):
        ''' Populate a smooth network of Curves suitable for
        interpolation in the Gordon sense. '''

        wi = self.wing
        wh0, wh1 = wi.halves

        l *= wi.chords[1] / 100.0
        dl *= wi.chords[1] / 100.0
        xb *= 2.0

        def fnc(x):
            return x**xb * (1 - x)**(2 - xb)
        mfnc = fnc(xb / 2.0)

        pairs = [(wh0.extract(u, 0), wh1.extract(u, 0))
                 for u in ul]

        Cl = []
        for ip, pair in enumerate(pairs):
            P0, D0 = pair[0].eval_derivatives(1, 1)
            P2, D2 = pair[1].eval_derivatives(1, 1)

            Pm = (P0 + P2) / 2.0
            W = normalize((D0 + D2) / 2.0)

            dli = fnc(ul[ip]) / mfnc * dl
            Pmi = Pm + (l + dli) * W

            Pw = obj_mat_to_4D(np.array((P0, Pmi, P2)))
            C = Curve(ControlPolygon(Pw=Pw), (1,))

            if pair not in (pairs[0], pairs[-1]):
                N = np.cross(P2 - P0, W)
                D0 = point_to_plane((0,0,0), N, D0)
                D2 = point_to_plane((0,0,0), N, D2)

                P1, w1 = make_one_arc(P0, D0, P2, D2, Pmi)

                Pw = obj_mat_to_4D(np.array((P0, P1, P2)))
                if w1 != 0.0:
                    Pw[1,:] *= w1
                else: # infinite point
                    Pw[1,-1] = 0.0

                C2 = Curve(ControlPolygon(Pw=Pw), (2,))
                C2 = refit_curve(C2, 15) #knob

                n, dummy, dummy, CC2 = make_curves_compatible1([C, C2])
                C, C2 = CC2

                # linear morphing
                for i in xrange(1, n):
                    lmy = np.sin(np.pi * float(i - 1) / float(n - 2))
                    cpt, cpt2 = [c.cobj.cpts[i].xyz for c in C, C2]
                    x, y, z = lmy * cpt + (1.0 - lmy) * cpt2
                    C.cobj.cpts[i].xyzw = x, y, z, 1.0

            Cl.append(C)

        Ck0 = wh0.extract(1, 1)
        Ck2 = wh1.extract(1, 1)

        Q = np.array([C.eval_point(0.5) for C in Cl])
        r = Q.shape[0] - 1
        U, Pw = global_curve_interp(r, Q, 3, ul)
        Ck1 = Curve(ControlPolygon(Pw=Pw), (3,), (U,))

        Ck = [Ck0, Ck1, Ck2]
        return Ck, Cl

    def _halve(self, nurbs):
        ''' Halve (split) the Wingtip at the LE. '''
        hs = nurbs.split(0.5, 1)
        h0, h1 = hs
        h0 = h0.reverse(0)
        h1 = h1.reverse(0).reverse(1)
        for h in h0, h1:
            h.colorize()
        self.nurbs = nurbs
        self.halves = [h0, h1]

    def fill(self, l=5.0, dl=2.0, xb=0.35, ntip=20, show=True):

        ''' Fill the Wingtip with a Gordon Surface.

        The Gordon Surface (see nurbs.surface.make_gordon_surface) is
        interpolated from an automatically generated bi-directional
        Curve network.

        Aside from the number of v-directional Curves to compose the
        network with, ntip, the user is free to vary any of the other
        three shape parameters: l, dl and xb.

                  \                                 /
                   \             Wing              /
                    \                             /
                     \________ tip chord ________/_
                      \  \  \      |      /  /  / |
             ntip = 7  \  \  \  Wingtip  /  /  /  | l
                        \__\__\____|____/__/__/   v
                                                  | dl
                                       xb <-----| v

        Parameters
        ----------
        l = the length of the Wingtip extension, in tip chord
            percentage
        dl = the delta, in tip chord percentage, to supplement l with;
             this actually varies according to the smooth funtion:

             [ x ** 2*xb ] * [ (1.0 - x) ** (2.0 - 2*xb) ]

        xb = the location, between 0 and 1, where that function takes on
             a maximum
        ntip = the number of v-directional cross-sections used to
               construct the Curve network; in the u-direction, that
               number is fixed to 3
        show = whether or not to draw the filled Wingtip

        Returns
        -------
        fig = a Figure

        '''

        ul = np.linspace(0, 1, ntip)
        vk = np.linspace(0, 1, 3)
        Ck, Cl = self._network(l, dl, xb, ul, vk)
        nurbs = make_gordon_surface(Ck, Cl, ul, vk)
        nurbs = nurbs.reverse(0)
        self._halve(nurbs)
        self._hs = [h.copy() for h in self.halves]
        self.Ck, self.Cl = Ck, Cl
        if show:
            return draw(self, self.wing, *(Ck + Cl))

    def refit(self, ncp, mcp, ppc=10, show=True):

        ''' Refit the Wingtip's Surfaces in an attempt to make the
        number of control points more manageable.  The Wingtip is not
        guaranteed to be watertight w.r.t. the Wing anymore.

        Parameters
        ----------
        ncp, mcp = the desired number of control points in the chordwise
                   and spanwise directions, respectively, to use in the
                   fit
        ppc = the number of times per control point to evaluate the
              Surfaces with (see nurbs.fit.refit_surface)

        Returns
        -------
        fig = a Figure

        '''

        if not self.nurbs:
            raise UnfitWingtip()
        h0, h1 = [refit_surface(h, ncp, mcp, ppc * ncp, ppc * mcp)
                  for h in self._hs]
        h0 = h0.reverse(0)
        h1 = h1.reverse(0).reverse(1)
        for h in h0, h1:
            V = h.U[1]; remap_knot_vec(V, 0.0, 0.5)
        nurbs = make_composite_surface([h0, h1], reorient=False)
        self._halve(nurbs)
        if show:
            return draw(self)

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(Wingtip, self)._glue(parent)
        g = []
        if self.nurbs:
            g += [self.nurbs] + self.halves + self._hs
        return g

    def _draw(self):
        ''' See Part._draw. '''
        d = []
        if self.nurbs:
            hs = list(self.halves)
            if self.half in (0, 1):
                hs.pop((self.half + 1) % 2)
            d += hs
        return d


# EXCEPTIONS


class WingtipException(Exception):
    pass

class UnfitWingtip(WingtipException):
    pass


# TODO


#uls = ul[1] / 2.0
#Ck1a, Ck1 = Ck1.split(uls)

## Cl0
#Q0, Ts = wi.LE.eval_derivatives(1, 1)
#Q1, Te = Ck1.eval_derivatives(uls, 1)
#U, Pw = local_curve_interp(1, [Q0, Q1], Ts, Te)
#Cl0 = Curve(ControlPolygon(Pw=Pw), (3,), (U,))

## reperam Ck1
#Q = np.zeros((ul.size, 3))
#Q[:,0] = ul; Q[0,0] = uls
#r = Q.shape[0] - 1
#U, Pw = global_curve_interp(r, Q, 2, ul)
#f = Curve(ControlPolygon(Pw=Pw), (2,), (U,))
#Ck1 = reparam_func_curve(Ck1, f)

## remap Cl0
#Cl0.cobj.Pw[1] = Ck1a.cobj.Pw[0]
#U, = Cl0.U; remap_knot_vec(U, 0.0, 0.5)
#Cl0 = make_composite_curve(Cl0, Cl0.reverse(), reorient=False)

#Cl[0] = Cl0
