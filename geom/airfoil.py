import numpy as np
from scipy.optimize import brute, fmin

from part           import Part

from nurbs.fit      import (global_curve_approx_errbnd,
                            global_curve_approx_fixedn_fair,
                            global_curve_interp)
from nurbs.curve    import (ControlPolygon,
                            Curve,
                            make_composite_curve,
                            make_curves_compatible1,
                            arc_length_to_param)
from nurbs.knot     import (normalize_knot_vec,
                            KnotOutsideKnotVectorRange)
from nurbs.point    import (points_to_obj_mat,
                            Point)
from nurbs.util     import distance

from plot.figure    import Figure as draw


__all__ = ['Airfoil']


class Airfoil(Part):

    ''' Create a B-spline approximation to an airfoil from an ordered
    set of 2-D data points.

    Intended usage
    --------------
    >>> af = Airfoil('foils/n0012.dat')
    geom.airfoil.Airfoil._load :: foils/n0012.dat contains 132 points
    >>> af.name
    'NACA 0012 AIRFOILS'
    >>> af.issymmetric
    True
    >>> af.issharp
    False
    >>> af.fit() # or af.fit2()
    geom.airfoil.Airfoil.fit :: fit with X control points
    geom.airfoil.Airfoil.fit :: warning, fit but blunt airfoil
    >>> af.sharpen() # or af.sharpen2()
    geom.airfoil.Airfoil.sharpen :: intersection Point found (X)
    >>> af.issharp
    True
    >>> af.fit() # or af.fit2() (only if the airfoil has been sharpened)
    >>> af.transform()
    >>> af.chord
    1.0

    '''

    def __init__(self, foil_path=''):

        ''' Instantiate an Airfoil from a .dat file, see the
        ./play/foils folder (more .dat files can be found online, e.g.
        on the UIUC Airfoil Coordinates Database).  For a reasonable fit
        the file should contain a total of at least 100 data points.

        For now, the .dat file must be in Lednicer format:

          - The file is read from the top; blank lines are discarded;
          - The first line must contain the name of the airfoil;
          - The second line must indicate the number of coordinates on
            the top and bottom curves;
          - All subsequent lines must be two numeric values separated by
            white space characters; these correspond to the XZ
            coordinates of points running from the upper curve LE to the
            TE and then from the lower curve LE to the TE.

        Parameters
        ----------
        foil_path = the path leading to the .dat file

        '''

        self.name = '' # the Airfoil's name/description

        self.data = {'lo': [], 'up': []} # the data Points
        self.intersection = None # the intersection Point

        self.nurbs = None # the fitted Curve
        self.halves = [None, None] # the fitted Curve, halved at the LE

        self.CL = None # the camber line Curve

        self.chord = 0.0 # the distance from the LE to (x = x_TE)
        self.tc = 0.0 # the thickness-to-chord ratio

        if foil_path:
            self._load(foil_path)

    @property
    def issharp(self):
        ''' Is the Airfoil sharp? '''
        if not self.data['lo']:
            return
        if self.intersection:
            return True
        lo, up = [self.data[si][-1].xyz for si in 'lo', 'up']
        return (up == lo).all()

    @property
    def issymmetric(self):
        ''' Is the Airfoil symmetric? '''
        if not self.data['lo']:
            return
        zlo, zup = [np.array( [pt.xyz[-1] for pt in self.data[si]] )
                    for si in 'lo', 'up']
        return (zlo.size == zup.size) and (zlo == - zup).all()

    def _load(self, foil_path):
        ''' Load a .dat file.  The Points, assumed to be lying in the XZ
        plane, are separated into two halves: lower and upper.  In both
        cases the first Point corresponds to the LE. '''

        def read(si, npt):
            nr = 0
            while True:
                xz = fh.readline().strip()
                if not xz:
                    continue
                nr += 1
                x, z = xz.split()
                x, z = float(x), float(z)
                pt = Point(x=x, z=z)
                self.data[si].append(pt)
                if nr == npt:
                    break

        with open(foil_path) as fh:
            self.name = fh.readline().strip()
            nup, nlo = fh.readline().split()
            nup, nlo = int(float(nup)), int(float(nlo))
            read('up', nup)
            read('lo', nlo)
        print('geom.airfoil.Airfoil._load :: '
              '{} contains {} points'.format(foil_path, nlo + nup))

    def _get_xbounds(self):
        ''' Get the Airfoil's lower and upper bounds in the horizontal
        direction. '''
        n = self.nurbs
        LE, TE = n.eval_point(0.5), n.eval_point(0.0)
        return LE[0], TE[0]

    def _get_zbounds(self):
        ''' Get the Airfoil's lower and upper bounds in the vertical
        direction. '''
        n = self.nurbs
        max_z = lambda u: n.eval_point(u)[2]
        uz = brute(max_z, [(0.0, 0.5)], finish=None)
        uz, = fmin(max_z, uz, xtol=0.0, ftol=0.0, disp=False)
        z0 = max_z(uz)
        if self.issymmetric:
            z1 = - z0
        else:
            max_z = lambda u: - n.eval_point(u)[2]
            uz = brute(max_z, [(0.5, 1.0)], finish=None)
            uz, = fmin(max_z, uz, xtol=0.0, ftol=0.0, disp=False)
            z1 = - max_z(uz)
        return z0, z1

    def _size(self):
        ''' Size the Airfoil, e.g. determine its chord and t/c ratio.
        The chord is here taken as the distance from the vertical lines
        (x = x_LE) and (x = x_TE). '''
        x0, x1 = self._get_xbounds()
        z0, z1 = self._get_zbounds()
        self.chord = x1 - x0
        self.tc = (z1 - z0) / self.chord
        h0, h1 = make_curves_compatible1(self.halves)[-1]
        h0.cobj.Pw[:] = (h0.cobj.Pw + h1.cobj.Pw) / 2.0
        self.CL = h0

    def _halve(self, nurbs):
        ''' Halve (split) the fitted Airfoil at the LE.  Both halves are
        oriented along the same direction than their data points, i.e.
        from LE to TE.  Care is also taken to ensure that the tangent
        line passing through the LE is perfectly vertical. '''
        min_x = lambda u: nurbs.eval_point(u)[0]
        ule = brute(min_x, [(0.48, 0.52)], finish=None)
        ule, = fmin(min_x, ule, xtol=0.0, ftol=0.0, disp=False)
        Hs = nurbs.split(ule)
        for h in Hs:
            U, = h.U; normalize_knot_vec(U)
        if self.issymmetric:
            h1 = Hs[1]
            h1.cobj.Pw[0,2] = 0.0
            h1.cobj.Pw[1,0] = h1.cobj.Pw[0,0]
            h0 = h1.copy()
            h0.mirror(N=(0,0,1))
        else:
            h0, h1 = Hs
            h0 = h0.reverse()
            h0.cobj.Pw[1,0] = h0.cobj.Pw[0,0]
            h1.cobj.Pw[1,0] = h1.cobj.Pw[0,0]
        n = make_composite_curve([h0, h1], remove=False)
        U, = n.U; normalize_knot_vec(U)
        self.nurbs = n
        self.halves = [h0, h1]
        self._size()
        self.colorize()

    def get_curvature_cplots(self, npt=500):

        ''' Get the curvature Curve plots corresponding to the lower and
        upper halves of the Airfoil.  To ease inspection each Curve plot
        is clamped chordwise and vertically between 0 and 1.

        Parameters
        ----------
        npt = the number of points to sample each halve with

        Returns
        -------
        [Curve, Curve] = the lower and upper curvature Curve plots

        '''

        if not self.nurbs:
            raise UnfitAirfoil()
        us = np.linspace(0, 1, npt)
        Cs = []
        for h in self.halves:
            Q = np.zeros((npt, 3))
            Q[:,0] = us
            for i, u in enumerate(us):
                Q[i,2] = h.eval_curvature(u)
            U, Pw = global_curve_interp(npt - 1, Q, 1, uk=us)
            c = Curve(ControlPolygon(Pw=Pw), (1,), (U,))
            sf1 = self.chord
            sf2 = 1.0 / max(c.cobj.Pw[:,2])
            w = self.nurbs.eval_point(0.5); w[1:] = 0.0
            c.scale(sf1, L=(1,0,0))
            c.scale(sf2, L=(0,0,1))
            c.translate(w)
            c.visible.pop('cobj')
            Cs.append(c)
        Cs[0].mirror(N=(0,0,1))
        return Cs

    def _fit(self, func, p, xargs, show):
        ''' Fit a pth-degree Curve through the data Points using the
        approximation function func and the extra argument xargs. '''
        lo, up = [points_to_obj_mat(self.data[si])
                  for si in ('lo', 'up')]
        Q = np.vstack((lo[-1:0:-1], up))
        i = self.intersection
        if i:
            xyzw = i.xyzw[np.newaxis,:]
            Q = np.vstack((xyzw, Q, xyzw))
        r = Q.shape[0] - 1
        args = (r, Q, p) + xargs
        U, Pw = func(*args)
        nurbs = Curve(ControlPolygon(Pw=Pw), (p,), (U,))
        print('geom.airfoil.Airfoil.fit :: '
              'fit with {} control points'.format(Pw.shape[0]))
        self._halve(nurbs)
        if not self.issharp:
            print('geom.airfoil.Airfoil.fit :: '
                  'warning, fit but blunt airfoil')
        if show:
            d = self.get_curvature_cplots()
            d += self.data['lo'] + self.data['up']
            if i:
                d += [i]
            fig = draw(self, *d, stride=0.1)
            fig.c.setup_preset('xz')
            return fig

    def fit(self, p=5, E=5e-4, show=True):

        ''' Fit (approximate) all data Points to within a specified
        tolerance E with a (p + 1)th order B-spline Curve (see
        nurbs.fit.global_curve_approx_errbnd for details).  The fitted
        Curve is then further processed; different actions are taken
        depending on if the Airfoil is symmetric or not.  Either way it
        is halved at the LE.

        Getting a good fit, i.e. one that is characterized by a smooth
        curvature plot, is not as easy as it sounds.  Here, the quality
        of the fit is not only affected by the degree and error bound
        input parameters but also through other factors such as the
        number and sampling quality of the available data Points.  Thus,
        many attempts will often be necessary before obtaining a
        satisfactory fit.  If all else fails consider using
        Airfoil.fit2.

        Parameters
        ----------
        p = the degree of the fit
        E = the max norm deviation of the fit
        show = whether or not to draw the fitted Airfoil along with
               curvature plots

        Returns
        -------
        fig = a Figure

        '''

        return self._fit(global_curve_approx_errbnd, p, (E,), show)

    def fit2(self, p=5, n=30, B=60, show=True):

        ''' The default fit works best with large sets of data Points.
        If this is not the case the resulting Curve is likely to exhibit
        small undesirable wiggles.  A second option is to fit the data
        Points for both least-squares distances *and* optimal fairness,
        the last of which is controlled by some user-defined parameter B
        (see nurbs.fit.global_curve_approx_fixedn_fair).  This parameter
        can vary greatly in magnitude (from as low as 0 and to as high
        as 10^7) depending on the data at hand and on the choice of p
        and n.

        Parameters
        ----------
        p = the degree of the fit
        n = the number of control points to use in the fit
        B = the bending coefficient
        show = whether or not to draw the fitted Airfoil along with
               curvature plots

        Returns
        -------
        fig = a Figure

        '''

        return self._fit(global_curve_approx_fixedn_fair, p, (n - 1, B), show)

    def sharpen(self, l=0.05, show=True):

        ''' Sharpen the Airfoil's TE by (1) extending its extremities
        with curvature-continuous B-spline Curves and (2) finding where
        they intersect.  Upon success an intersection Point is created
        and set on the Airfoil.intersection attribute.  This Point will
        be automatically included when the Airfoil is refitted with any
        of the available fitting techniques, after which point the
        Airfoil should be sharp.  Should you later desire to discard the
        intersection Point, simply reset Airfoil.intersection equal to
        None.

        Parameters
        ----------
        l = the length of the two extensions (for phase 1)
        show = whether or not to draw the new set of data Points

        Returns
        -------
        fig = a Figure

        '''

        def dist(us):
            try:
                return distance(n0.eval_point(us[0]),
                                n1.eval_point(us[1]))
            except KnotOutsideKnotVectorRange:
                return np.inf

        n = self.nurbs
        if not n:
            raise UnfitAirfoil()
        if self.issharp:
            raise AlreadySharpAirfoil()
        u = arc_length_to_param(n, l)
        n0, n1 = [n.extend(u, end=end)
                  for end in (False, True)]
        U0, = n0.U; n0.colorize()
        U1, = n1.U; n1.colorize()
        bnds = [(U0[0], U0[-1]),
                (U1[0], U1[-1])]
        us = brute(dist, bnds, finish=None)
        u0, u1 = fmin(dist, us, xtol=0.0, ftol=0.0, disp=False)
        x0, x1 = (n0.eval_point(u0),
                  n1.eval_point(u1))
        l2n = distance(x1, x0)
        if l2n > 1e-8:
            draw(n, n0, n1, stride=0.1)
            raise NoIntersectionFoundAirfoil(l2n)
        i = np.average([x0, x1], axis=0)
        if self.issymmetric:
            i[2] = 0.0
        i = Point(*i); i.colorize()
        print('geom.airfoil.Airfoil.sharpen :: '
              'intersection Point found ({})'.format(i.xyz))
        self.intersection = i
        if show:
            d = [n, n0, n1, i]
            d += self.data['lo'] + self.data['up']
            fig = draw(*d, stride=0.1)
            fig.c.setup_preset('xz')
            return fig

    def sharpen2(self, show=True):

        ''' Use as a fallback to Airfoil.sharpen.  Here, the first and
        last control points of the Airfoil's underlying NURBS Curve are
        simply averaged out.  This works best when the Airfoil is
        parameterized will only a few control points.

        Parameters
        ----------
        show = whether or not to draw the new sharpened Airfoil along
               with curvature plots

        Returns
        -------
        fig = a Figure

        '''

        n = self.nurbs
        if not n:
            raise UnfitAirfoil()
        if self.issharp:
            raise AlreadySharpAirfoil()
        cpt0, cpt1 = [n.cobj.cpts[e]
                      for e in (0, -1)]
        ave = np.average([cpt0.xyzw, cpt1.xyzw], axis=0)
        if self.issymmetric:
            ave[2] = 0.0
        cpt0.xyzw = cpt1.xyzw = ave
        self._halve(n)
        if show:
            d = self.get_curvature_cplots()
            fig = draw(self, *d, stride=0.1)
            fig.c.setup_preset('xz')
            return fig

    def transform(self, show=True):

        ''' First scale the Airfoil by the inverse of its chord length,
        and then translate it so that its quarter chord coincides with
        the origin.  Note that the transformed Airfoil won't necessarily
        match the data Points anymore (depending if it has been apriori
        sharpened or not, etc.).  This step should be performed prior to
        passing the Airfoil to a Wing.

        Parameters
        ----------
        show = whether or not to draw the transformed Airfoil

        Returns
        -------
        fig = a Figure

        '''

        n = self.nurbs
        if not n:
            raise UnfitAirfoil()
        self.glue()
        sf = 1.0 / self.chord
        self.scale(sf)
        w = - n.eval_point(0.5) - (0.25,0,0)
        self.translate(w)
        self.unglue()
        self._size()
        if show:
            fig = draw(self, stride=0.1)
            fig.c.setup_preset('xz')
            return fig

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(Airfoil, self)._glue(parent)
        g = []
        if self.nurbs:
            g += [self.nurbs, self.CL] + self.halves
        return g

    def _draw(self):
        ''' See Part._draw. '''
        d = []
        if self.nurbs:
            d += self.halves
        else:
            d += self.data['lo'] + self.data['up']
            i = self.intersection
            if i:
                d += [i]
        return d


# EXCEPTIONS


class AirfoilException(Exception):
    pass

class UnfitAirfoil(AirfoilException):
    pass

class AlreadySharpAirfoil(AirfoilException):
    pass

class NoIntersectionFoundAirfoil(AirfoilException):
    pass
