import numpy as np

from part            import Part

from ffd.ffd         import FFDVolume

from nurbs.conics    import make_ellipse
from nurbs.curve     import (make_linear_curve,
                             reparam_arc_length_curve,
                             make_composite_curve)
from nurbs.fit       import refit_curve
from nurbs.nurbs     import obj_mat_to_4D
from nurbs.point     import Point
from nurbs.surface   import make_revolved_surface_nrat
from nurbs.transform import (scale,
                             translate)
from nurbs.volume    import ControlVolume
from nurbs.util      import (distance,
                             intersect_3D_lines)

from plot.figure     import Figure as draw


__all__ = ['Fuselage', 'FuselageMolder', 'FairingMolder']


class Fuselage(Part):

    ''' Create the outer mold line of a Fuselage from a bullet shaped
    cylinder by successively sculpting the latter with a family of
    free-form deformation volumes, hereby referred to as "molders".

    For now there are two types of molders, both of which superclass
    FFDVolume: the FuselageMolder and the FairingMolder.  The idea is to
    use any number of those to progressively shape the evolving
    geometry.  For example, two FuselageMolders can be used to
    independently design the nose and rear parts of the Fuselage, while
    only one FairingMolder is usually adequate to emulate say a
    wing-body fairing.

    To ease their manipulations each molder has an associated set of
    "pilot points".  Depending on the type molder their behavior is
    different but the goal is always the same, i.e. to move groups of
    control points pertaining to the same FFD lattice together as
    opposed to individually.  This tends to create much smoother, a.k.a.
    fairer, shapes.

    Intended usage
    --------------
    >>> fus = Fuselage(N, B, R) # nose length, body length, radius
    >>> fus.setup()

    >>> m = fus.mold_fuselage(N, end=0) # m is a FuselageMolder
    >>> m.embed(fus) # since m superclasses FFDVolume
    >>> draw(fus, m) # manipulate "pilot points"

    repeat for rear, then, once satisfied,

    >>> mf = fus.mold_fairing(X) # mf is a FairingMolder
    >>> mf.embed(fus) # since mf also superclasses FFDVolume
    >>> draw(fus, mf0) # manipulate "pilot points"

    finally (and optionally),

    >>> fus.finalize()

    '''

    def __init__(self, nose_length, body_length, radius):

        ''' Instantiate the Fuselage given basic dimensions.

                          -------------------
                                   |
                   body_length <-- | --> nose_length
                                   |
                          -------------------
                                 x = 0

        Parameters
        ----------
        nose_length = the length of the nose i.e. of the cockpit
        body_length = the length of the body (overall fuselage length
                      minus nose_length)
        radius = the radius

        '''

        self.NL = float(nose_length)
        self.BL = float(body_length)
        self.R  = float(radius)

        self.nurbs = None # the OML Surface
        self._nurbs = None
        self.molders = [] # the (ordered) list of molders

    def setup(self, nlcpt=50, cap=True, show=True):

        ''' Setup the NURBS representation of the initial bullet head
        cylinder.  Analogous to modelling clay, it is this NURBS Surface
        that will be successively shaped by molders.

        Parameters
        ----------
        nlcpt = the number of longitudinal control points used to
                represent the cylinder with
        cap = whether or not to close the rear-end gap with a rounded
              Surface; for geometries destined for inviscid flow
              simulations this flag should be set to False
        show = whether or not to draw the newly setup cylinder

        Returns
        -------
        fig = a Figure

        '''

        R, BL, NL = self.R, self.BL, self.NL

        O, X, Y = [0,0,0], [-1,0,0], [0,0,-1]
        args = (O, X, Y, NL, R, 0, np.pi / 2.0)
        P0, P1 = Point(0,0,-R), Point(BL,0,-R)

        nose = make_ellipse(*args)
        body = make_linear_curve(P0, P1)
        for c in nose, body:
            reparam_arc_length_curve(c)

        n = make_composite_curve([nose, body])
        n = refit_curve(n, nlcpt, num=50000)
        n2, = n.cobj.n
        for i in xrange(n2 + 1):
            Pw = n.cobj.Pw[i]
            if Pw[0] > 0 or Pw[2] < - R:
                Pw[2] = - R

        if cap:
            P2 = Point(BL,0,0)
            rear = make_linear_curve(P1, P2)
            reparam_arc_length_curve(rear)
            n = make_composite_curve([n, rear])

        n = make_revolved_surface_nrat(n, O, [1,0,0], 180)

        if not cap:
            n.cobj.Pw[:,-1,1] = 0.0

        self.nurbs = n
        self.molders = []
        self.clamp()
        if show:
            return draw(self)

    def mold_fuselage(self, L, nslice=4, end=None, R=None):

        ''' Convenience method to instantiate a FuselageMolder.

        Parameters
        ----------
        L = the length of the FuselageMolder
        nslice = the number of slices used to define the FuselageMolder
                 with
        end = (optional) the end of the Fuselage at which the
              FuselageMolder will be applied to (0: nose, 1: rear); this
              simply translates the molder to the appropriate location
        R = the radius of the FuselageMolder (default: 1.1 * R)

        Returns
        -------
        m = the FuselageMolder

        '''

        assert L > 0, nslice > 3
        R = R if R else 1.1 * self.R
        m = FuselageMolder(L, R, nslice)
        if end is not None:
            if end == 0: # nose
                w = - self.NL - 1e-3
            elif end == 1: # rear
                w = - L + self.BL + 1e-3
            m.translate([w,0,0])
        m.nurbs = self.nurbs.copy()
        self.molders.append(m)
        return m

    def mold_fairing(self, L, mloc=0.5, R=None):

        ''' Convenience method to instantiate a FairingMolder.

        Parameters
        ----------
        L = the length of the FairingMolder
        mloc = the location, in fraction of L, of the middle slice
               pertaining to the FairingMolder
        R = the radius of the FairingMolder (default: 1.1 * R)

        Returns
        -------
        m = the FairingMolder

        '''

        assert L > 0, 0 < mloc < 1
        R = R if R else 1.1 * self.R
        m = FairingMolder(L, R, mloc)
        m.nurbs = self.nurbs.copy()
        self.molders.append(m)
        return m

    def finalize(self, d=1e-2, show=True):

        ''' Perform some post-processing steps on the OML Surface.  This
        should only be called after having applied all molders (but
        obviously before creating other objects that depend on the final
        Fuselage's shape, such as Junctions).

        For now, this method is just a wrapper around
        nurbs.surface.Surface.removes, which removes as many control
        points as possible up to a given tolerance.

        Parameters
        ----------
        d = the maximum deviation allowed from the current OML Surface
        show = whether or not to draw the new, simplified OML Surface

        Returns
        -------
        fig = a Figure

        '''

        if self._nurbs:
            n = self._nurbs
        else:
            n = self.nurbs
            self._nurbs = n.copy()
        e, n = n.removes(2, d=d)
        print('geom.fuselage.Fuselage.finalize :: '
              '{} control points removed (u, v)'.format(e))
        self.nurbs = n
        self.clamp()
        if show:
            return draw(self)

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(Fuselage, self)._glue(parent)
        return [self.nurbs] if self.nurbs else []

    def _draw(self):
        ''' See Part._draw. '''
        return [self.nurbs] if self.nurbs else []


class CylindricalFFDVolume(FFDVolume):

    def __getstate__(self):
        ''' Pickling. '''
        d = self.__dict__.copy()
        ds = d.viewkeys() - {'_cobj', '_p', '_U', 'nurbs'}
        for k in ds:
            del d[k]
        return d

    def _setup_control_volume(self, R, zs):
        C = make_linear_curve(Point(), Point(z=R))
        C = C.elevate(2)
        args = C, (0,0,0), (-1,0,0), 180, 3, 1e-4
        S = make_revolved_surface_nrat(*args)
        Ps = [S.cobj.Pw[:,:,:-1][:,:,np.newaxis,:]]
        for z in zs[1:]:
            P = Ps[0].copy(); P[...,0] += z
            Ps.append(P)
        P = np.concatenate(Ps, axis=2)
        Pw = obj_mat_to_4D(P)
        Pw[0,...,1] = Pw[-1,...,1] = 0.0
        return ControlVolume(Pw=Pw)

    def refresh(self):
        for p in self.pilot_points:
            p.transform_ffd(p)
        self._fill_batch()
        super(CylindricalFFDVolume, self).refresh()

    def rotate(self, *args, **kwargs):
        raise NotImplementedError

    def mirror(self, *args, **kwargs):
        raise NotImplementedError

    def scale(self, *args, **kwargs):
        raise NotImplementedError

    def shear(self, *args, **kwargs):
        raise NotImplementedError

    def copy(self):
        c = self.__class__()
        c._cobj = self.cobj.copy()
        c._setup()
        c._set_cpoint_color()
        return c

    def _draw(self, *args, **kwargs):
        super(CylindricalFFDVolume, self)._draw(*args, **kwargs)
        Ps = self.pilot_points
        for fig in self._figs:
            if Ps[0] not in fig.pos['points']:
                fig.inject(*Ps)


class FuselageMolder(CylindricalFFDVolume):

    def __init__(self, L=1.0, R=1.0, nslice=4):
        zs = np.linspace(0, L, nslice)
        cvol = self._setup_control_volume(R, zs)
        r = nslice - 1 if nslice < 4 else 3
        super(CylindricalFFDVolume, self).__init__(cvol, (3,3,r))

    def _setup(self):

        def transform_ffd0(O):
            R, D, Pw = O._R, O._D, O._Pw
            sf = O.xyz[1] / D
            if not np.allclose(sf, 1.0):
                scale(Pw, sf, L=[0,1,0])
            dz = O.xyz[2] - Pw[0,0,2]
            if not np.allclose(dz, 0.0):
                Om = [r._xyzw for r in R] + [Pw]
                for o in Om:
                    translate(o, [0,0,dz])
            O._D = O.xyz[1]

        def transform_ffd1(R):
            O, D, Pw = R._O, R._D, R._Pw
            z0 = O.xyz[2]
            sf = (R.xyz[2] - z0) / D
            if not np.allclose(sf, 1.0):
                x = R.xyz[0]
                scale(Pw, sf, [x,0,z0], [0,0,1])
            R._D = R.xyz[2] - z0

        super(FuselageMolder, self)._setup()
        n, dummy, l = self.cobj.n
        Ps = []
        for k in xrange(l + 1):
            Pw = self.cobj.Pw[:,:,k]

            O = Pw[n/2, 0,:3]
            R = Pw[n/2,-1].copy(); R[2] = O[2]
            L = R[:3] - O
            scale(R, 1.3, O, L) # knob
            R = Point(*R); Ps.append(R)
            R._D = R.xyz[1]
            R._Pw = Pw
            R.plane = O, [1,0,0]
            R.transform_ffd = transform_ffd0

            O = Pw[-1, 0,:3]
            R = Pw[-1,-1].copy()
            L = R[:3] - O
            scale(R, 1.3, O, L) # knob
            R = Point(*R); Ps.append(R)
            R._D = R.xyz[2] - O[2]
            R._Pw = Pw[n/2:]
            R.line = O, [0,0,1]
            R.transform_ffd = transform_ffd1

            O = Pw[0, 0,:3]
            R = Pw[0,-1].copy()
            L = R[:3] - O
            scale(R, 1.3, O, L) # knob
            R = Point(*R); Ps.append(R)
            R._D = R.xyz[2] - O[2]
            R._Pw = Pw[:n/2]
            R.line = O, [0,0,1]
            R.transform_ffd = transform_ffd1

            O, R0, R1 = Ps[-3:]
            O._R = (R0, R1)
            R0._O = O
            R1._O = O

        for p in Ps:
            p.ffd = self

        self.pilot_points = Ps
        self.glued = Ps + [self]


class FairingMolder(CylindricalFFDVolume):

    def __init__(self, L=1.0, R=1.0, mloc=3):
        zs = np.array([0, mloc * L, L])
        cvol = self._setup_control_volume(R, zs)
        super(CylindricalFFDVolume, self).__init__(cvol, (3,3,2))

    def _setup(self):

        def transform_ffd(R):
            O, D, S, Pw = R._O, R._D, R._S, R._Pw
            D_new = distance(O, R.xyz)
            sf = D_new / D
            if np.allclose(sf, 1.0):
                return
            scale(Pw, sf, R.line[0], R.line[1])
            R._D = D_new
            if S and not np.allclose(S.xyz[2], R.xyz[2]):
                args = S.line[0], S.line[1], R.xyz, [0,1,0]
                xyz = intersect_3D_lines(*args)
                S._xyzw[1:3] = xyz[1:3]
                S.transform_ffd(S)

        super(FairingMolder, self)._setup()
        Ps = []
        for i in xrange(self.cobj.n[0] + 1):
            Pw = self.cobj.Pw[i,:,1]

            R = Pw[-1].copy()
            O = Pw[ 0,:3]
            L = Pw[-1,:3] - O
            scale(R, 1.3, O, L) # knob
            R = Point(*R); Ps.append(R)

            R._O = O
            R._D = distance(O, R.xyz)
            R._S = None
            R._Pw = Pw
            R.line = O, L

            R.transform_ffd = transform_ffd
            R.ffd = self

        Ps[ 0]._S = Ps[ 1]
        Ps[ 1]._S = Ps[ 0]
        Ps[-2]._S = Ps[-1]
        Ps[-1]._S = Ps[-2]

        self.pilot_points = Ps
        self.glued = Ps + [self]
