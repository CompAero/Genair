import numpy as np

from nurbs.curve     import make_curves_compatible1
from nurbs.nurbs     import NURBSObject
from nurbs.point     import Point
from nurbs.surface   import (split_nsided_region,
                             find_central_point_nsided_region,
                             find_normal_vec_nsided_region,
                             generate_inner_curves_nsided_region,
                             generate_inner_cross_deriv_nsided_region,
                             make_nsided_region)
from nurbs.transform import translate
from part            import Part

from plot.figure     import Figure as draw


# WORK IN PROGRESS


__all__ = ['Filler']


class Filler(Part):

    ''' Create a generic Filler.  As its name suggests, a Filler
    attempts to fill an N-sided hole with as many NURBS patches.

    Presently, the Filler is used by other higher-level objects such as
    the Wingtip (N = 2) and the Fuselage (2 x (N = 2)).

    Intended usage
    --------------
    >>> fi = Filler(Bk, Gk)
    >>> fi.design()
    >>> fi.blend() # Not satisfied?
    >>> fi.design()
    >>> fi.blend()

    The Filler should not be transformed between calls to Filler.design
    and Filler.deoriginate.

    After deoriginating, the design cycle can restart from either
    Filler.pre_split (if new split points are desired) immediately
    followed by Filler.split, or directly from Filler.design.

    Source
    ------
    Piegl & Tiller, Filling n-sided regions with NURBS patches, The
    Visual Computer, 1999.

    '''

    def __init__(self, Bk, Gk):

        ''' Form a Filler from a set of N boundary curves and associated
        cross-derivative fields.  These are usually picked interactively
        whilst in plot.mode.ExtractCrossBoundariesMode.

        There are many subtleties involved in designing a Filler.  Some
        of them are pointed out nurbs.surface.make_nsided_region and
        many more in the reference cited therein.  For best results,
        all the boundary Curves should be about the same arc length.

        Parameters
        ----------
        Bk = the boundary Curves
        Gk = the cross-derivative Curves

        '''

        self.C = Point()
        self.CN = Point()

        self.Mk = []
        self.DMk = np.array([[]])

        self.Ck = []
        self.Dk = []

        self.CLRk = []
        self.DLRk = []

        self.CIk = []
        self.BLRk = []

        self.nurbs = []

        self.O = None # originating and deoriginating

        self._split(Bk, Gk)

    def __setstate__(self, d):
        ''' Unpickling. '''
        if d['C']:
            d['C'].originate = self._originate
        if d['CN']:
            d['CN'].innerize = self._innerize
        self.__dict__.update(d)

    def _innerize(self):
        ''' Generate the Filler's inner Curves. '''
        Mk = np.array([mk.xyz for mk in self.Mk])
        CIkn = generate_inner_curves_nsided_region(self.C.xyz, self.CN.xyz, Mk, self.DMk)
        if not self.CIk:
            self.CIk = CIkn
        else:
            for ci, cin in zip(self.CIk, CIkn):
                ci.cobj.Pw.setflags(write=True)
                ci.cobj.Pw[:] = cin.cobj.Pw
                ci._construct_GL_arrays()
                ci._fill_batch()
            for clr in self.CLRk:
                for c in clr:
                    c._fill_batch()

    def _originate(self):
        ''' Originate the whole Filler IN-PLACE, so that the Filler's
        central Point lies on the origin.  This considerably eases
        interactive manipulations. '''

        O = self.C.xyz
        if (O != 0.0).any():
            self.O = O if self.O is None else self.O + O
            os = [self.C] + self.Ck + self.Mk + \
                    [c for clr in self.CLRk for c in clr]
            for o in os:
                if isinstance(o, Point):
                    translate(o._xyzw, - O)
                elif isinstance(o, NURBSObject):
                    if o._figs:
                        o.cobj.Pw.setflags(write=True)
                        translate(o.cobj.Pw, - O)
                        o.cobj.Pw.setflags(write=False)
                        translate(o.cobj._Pwf, - O)
                        o._fill_batch()
                    else:
                        translate(o.cobj.Pw, - O)
        self._innerize()

    def _split(self, Bk, Gk):
        ''' Split all boundary Curves at their parametric midpoint. '''

        for BG in zip(Bk, Gk):
            CD = make_curves_compatible1(BG)[3]
            C, D = CD
            self.Ck.append(C); self.Dk.append(D)

        self.CLRk, self.DLRk, Mk, self.DMk = split_nsided_region(self.Ck, self.Dk)

        CLRk0 = None if not len(self.Ck) == 2 else self.CLRk[0]
        C = find_central_point_nsided_region(Mk, self.DMk)
        CN = find_normal_vec_nsided_region(C, Mk, self.DMk, CLRk0)

        self.C = Point(*C)
        self.C.originate = self._originate

        self.CN = Point(*CN)
        self.CN.innerize = self._innerize

        self.Mk = [Point(*mk) for mk in Mk]

    def design(self, show=True):

        ''' Design the Filler's default entities, i.e. its inner Curves,
        its central Point and/or normal vector.  All of these will have
        a direct impact on the shape of the resultant blended NURBS
        patches.  Choose them wisely!

        Parameters
        ----------
        show = whether or not to interactively design the Filler

        Returns
        -------
        fig = a Figure

        '''

        self.nurbs = []
        self.unglue()
        self._originate()
        if show:
            return draw(self, self.C, self.CN, *self.CIk)

    def blend(self, eps=1.0, show=True):

        ''' Blend all boundary and inner Curves to produce N smooth
        Surfaces.

        Parameters
        ----------
        eps = the tolerance on geometric continuity (in degrees)
        show = whether or not to draw the blended Surfaces

        Returns
        -------
        fig = a Figure

        '''

        if not self.CIk:
            raise UndesignedFiller(self)
        self.BLRk = generate_inner_cross_deriv_nsided_region(self.CN.xyz, self.CIk, self.CLRk)
        self.nurbs = make_nsided_region(self.CLRk, self.DLRk, self.CIk, self.BLRk, eps)

        self.glue()
        O = self.O; self.O = None
        if O is not None:
            self.C.translate(O)
        self.unglue()

        self.colorize()
        if show:
            return draw(self)

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(Filler, self)._glue(parent)
        g = [self.C] + self.Ck + self.Mk + self.CIk + self.nurbs + \
                [c for clr in self.CLRk for c in clr]
        return g

    def _draw(self):
        ''' See Part._draw. '''
        return (self.nurbs or
                [c for clr in self.CLRk for c in clr] or
                self.Ck)


# EXCEPTIONS


class FillerException(Exception):
    pass

class UndesignedFiller(FillerException):
    pass
