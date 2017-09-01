import numpy as np

from airfoil         import UnfitAirfoil
from junction        import (TrimmedJunction,
                             WatertightJunction)
from misc            import WingStructure
from part            import Part
from wingtip         import Wingtip

from nurbs.fit       import (global_curve_approx_fixedn,
                             local_curve_interp)
from nurbs.curve     import (ControlPolygon,
                             Curve,
                             param_to_arc_length,
                             arc_length_to_param)
from nurbs.knot      import normalize_knot_vec
from nurbs.nurbs     import obj_mat_to_4D
from nurbs.point     import obj_mat_to_points
from nurbs.surface   import (get_sweep_orient_func,
                             make_swept_surface,
                             make_surfaces_compatible3)
from nurbs.transform import shear
from nurbs.util      import (angle,
                             signed_angle,
                             distance,
                             point_to_plane,
                             norm,
                             normalize,
                             intersect_line_plane)

from plot.figure     import Figure as draw
from plot.pobject    import (PlotObject,
                             update_figures)


__all__ = ['Wing', 'WingMerger', 'HalfWingMerger',
           'BSplineFunctioncreator1', 'BSplineFunctionCreator2',
           'BSplineFunctioncreator3']


class Wing(Part):

    ''' Create a Wing by sweeping an Airfoil along an arbitrary
    trajectory Curve.  This technique allows for linear and nonlinear
    Wings of varying sweep, dihedral, twist, taper and span.

    NOTE: The Wing's chordwise, spanwise, and vertical directions are
    assumed to be lying along the + X, Y, and Z coordinate axes,
    respectively.

    Intended usage
    --------------
    >>> wi = Wing(n0012) # n0012 being an Airfoil object
    >>> wi.orient(T) # T being a Curve object
    >>> wi.fit()

    '''

    def __init__(self, airfoil_root, airfoil_tip=None, half=2):

        ''' Instantiate a Wing object given a root Airfoil and,
        optionally, a tip Airfoil.  If no tip Airfoil is specified, the
        swept Wing will assume the shape of the root Airfoil only,
        otherwise its shape will result from a linear interpolation
        (also known as linear morphing) between the root and tip
        Airfoils.

        Parameters
        ----------
        airfoil_root = the Airfoil to sweep the Wing with
        airfoil_tip = the tip Airfoil, if any
        half = the half of the Wing we are interested in (useful for
               Wings lying on the symmetry plane): 0, 1 or 2 stands for
               lower, upper or both, respectively

        '''

        check_airfoil(airfoil_root)
        if airfoil_tip:
            check_airfoil(airfoil_tip)

        self.airfoils = [airfoil_root.copy(), # the root and tip Airfoils
                         airfoil_tip.copy() if airfoil_tip else None]

        self.T = None # the trajectory Curve
        self.Bv = None # the orientation function Curve

        self.Tw = None # the B-spline twisting function
        self.scs = [None, None, None] # the B-spline scaling functions

        self.nurbs = None # the swept Surface
        self.halves = [None, None] # the swept Surface, halved at the LE
        self.half = half # the half of interest

        self.LE = None # the leading edge Curve
        self.TE = None # the trailing edge Curve
        self.QC = None # the quarter chord Curve

        self.chords = [0.0, 0.0] # the root and tip chords

        self.tip = None # the Wingtip
        self.junctions = [None, None] # the root and tip *Junctions
        self.structure = None # the WingStructure

    @property
    def dihedral(self):
        ''' Get the dihedral angle w.r.t. the trajectory Curve T. '''
        if not self.T:
            raise UnorientedWing()
        T = self.T.eval_derivatives(0, 1)[1]; T[0] = 0
        s = 1.0 if T[2] >= 0.0 else - 1.0
        return s * angle(T, (0,1,0))

    @dihedral.setter
    def dihedral(self, dihedral):
        ''' Set the dihedral angle w.r.t. the trajectory Curve T. '''
        if not self.T:
            raise UnorientedWing()
        O = self.T.eval_point(0)
        dihedral -= self.dihedral
        for e in self, self.Bv:
            e.rotate(dihedral, (1,0,0), O)

    @property
    def sweep(self):
        ''' Get the sweep angle w.r.t. the trajectory Curve T. '''
        if not self.T:
            raise UnorientedWing()
        T = self.T.eval_derivatives(0, 1)[1]
        Tp = point_to_plane((0,0,0), (1,0,0), T)
        s = 1.0 if T[0] >= 0.0 else - 1.0
        return s * angle(T, Tp)

    @sweep.setter
    def sweep(self, sweep):
        ''' Set the sweep angle w.r.t. the trajectory Curve T. '''
        if not self.T:
            raise UnorientedWing()
        if not -90 < sweep < 90:
            raise InvalidSweepAngle()
        O, T = self.T.eval_derivatives(0, 1); Y = norm(T)
        Tp = point_to_plane((0,0,0), (1,0,0), T)
        X = Y * np.tan(np.deg2rad(self.sweep))
        X = Y * np.tan(np.deg2rad(sweep)) - X
        sweep = np.rad2deg(np.arctan(X / Y))
        self.shear(sweep, Tp, (1,0,0), O)

    def _size(self):
        ''' Size the Wing, e.g. determine its root and tip chords. '''
        h = self.halves[0]
        LE, TE = h.extract(0, 0), h.extract(1.0, 0)
        es = [(LE.eval_point(u), TE.eval_point(u)) for u in 0, 1]
        self.chords[0] = distance(es[0][0], es[0][1])
        self.chords[1] = distance(es[1][0], es[1][1])
        self.LE, self.TE = LE, TE
        self.QC = LE.copy()
        self.QC.cobj.Pw[:] = 0.75 * LE.cobj.Pw + 0.25 * TE.cobj.Pw

    def _halve(self, nurbs):
        ''' Halve (split) the Wing at the LE. '''
        hs = nurbs.split(0.5, 0)
        h0, h1 = hs
        h0 = h0.reverse(0)
        for h in h0, h1:
            U = h.U[0]; normalize_knot_vec(U)
        self.nurbs = nurbs
        self.halves = [h0, h1]
        self.colorize()
        self._size()

    def orient(self, T, Tw=None, m=10, show=True):

        ''' Generate an orientation function to the trajectory Curve T.

        An orientation function is one that, as its name suggests,
        orients the root/tip Airfoils along T during the sweeping
        process (see nurbs.surface.make_swept_surface and Wing.fit).

        It is initialized via the projection normal method located in
        nurbs.surface.get_sweep_orient_func.  Note that here this
        function is restricted in such a way that the Airfoil instances
        taken along T (once translated) are only allowed to rotate about
        the X axis (plus twist, if any).  This is to ensure that
        ultimately each spanwise cross-section of a fitted Wing lies in
        its own plane whose normal has no X component.

        Parameters
        ----------
        T = the trajectory Curve
        Tw = the B-spline twisting function, if any
        m = the number of points to construct the orientation function
            with
        show = whether or not to draw the orientation vectors

        Returns
        -------
        fig = a Figure

        Examples
        --------
        >>> T = nurbs.tb.make_linear_curve(Point(), Point(y=3))
        >>> tw = BSplineFunctionCreator1(end=(0.0,90.0)).fit()
        >>> wi.orient(T, Tw=tw)

        '''

        T = T.copy()
        if Tw:
            Tw = Tw.copy()

        U, = T.U; normalize_knot_vec(U)
        T0 = T.eval_derivatives(0, 1)[1]; T0[0] = 0
        B0 = np.cross(T0, (-1,0,0))
        Bv = get_sweep_orient_func(B0, T, (1,0,0), Tw, m)

        self.T, self.Tw, self.Bv = T, Tw, Bv
        if show:
            vb = np.linspace(0, 1, m)
            O, B = np.zeros((2, m, 3))
            for i, v in enumerate(vb):
                O[i], B[i] = T.eval_point(v), Bv.eval_point(v)
            V = O + 0.2 * B # knob
            O = obj_mat_to_points(obj_mat_to_4D(O)).tolist()
            V = obj_mat_to_points(obj_mat_to_4D(V)).tolist()
            for v in V:
                v.color = (0, 255, 255, 255)
            return draw(T, *(O + V))

    def fit(self, K=1, scs=(None, None, None), show=True):

        ''' Fit (sweep) the root/tip Airfoils along the Wing's
        trajectory Curve T with optional scaling.

        Use BSplineFunctionCreators if you do end up applying scaling.
        Recall that an Airfoil is defined in the XZ plane only,
        therefore applying Y-directional scaling won't affect the Wing.
        This method also uses the orientation function previously
        determined by Wing.orient.

        Note that for nonlinear Wings (in sweep, dihedral, twist, etc.),
        only an approximation is constructed.  As explained in
        nurbs.surface.make_swept_surface, a better approximation can be
        obtained by increasing the value of K.  As a rule of thumb, a
        fit is considered good when, while facing the YZ plane, each
        spanwise row of control points of the Wing falls on a straight
        line.

        Parameters
        ----------
        K + 1 = the (minimum) number of Airfoil instances taken along T
        scs = the B-spline scaling functions, if any (a 3-tuple
              corresponding to scaling in X, Y and Z, respectively)
        show = whether or not to draw the swept Wing

        Returns
        -------
        fig = a Figure

        Examples
        --------
        >>> sc = BSplineFunctionCreator1(end=(2.0,1.0)).fit()
        >>> wi.fit(K=5, scs=(sc,None,sc))

        '''

        if not self.T:
            raise UnorientedWing()

        scs = [(sc.copy() if sc else None) for sc in scs]
        if any(scs):
            q, = self.T.p
            if q == 1:
                q = np.array([sc.p[0] for sc in scs if sc])
                if (q > 1).any():
                    print('geom.wing.Wing.fit :: '
                          'warning, linear sweep with nonlinear scaling')

        self.tip = None
        self.junctions = [None, None]
        self.structure = None

        afr, aft = self.airfoils
        args = afr.nurbs, self.T, self.Bv, K, scs
        if aft:
            args += aft.nurbs,
        nurbs = make_swept_surface(*args, local=get_sweep_local_sys_proj)
        self._halve(nurbs)

        self.scs = scs
        if show:
            return draw(self, self.T)

    def snap(self, last=False):

        ''' Snap the first (or last) v-directional row of control points
        of each of the Wing's halves onto the (Y = 0) symmetry plane.
        The Wing's underlying NURBS Surface is untouched.

        '''

        check_wing(self)
        f, s = (0, 1) if not last else (-1, -2)
        for h in self.halves:
            P = h.cobj.Pw[...,:3]
            n, dummy = h.cobj.n
            for i in xrange(n + 1):
                L0, L = P[i,f], P[i,s] - P[i,f]
                P[i,f] = intersect_line_plane(L0, L, (0,0,0), (0,1,0))
                P[i,f,1] = 0.0
        self._size()
        update_figures(self.halves)

# AUXILIARY PART GENERATON

    def generate_tip(self):

        ''' Generate a Wingtip.

        Should you later desire to discard the Wingtip, simply reset
        Wing.tip equal to None.

        Examples
        --------
        >>> tip = wi.generate_tip()
        >>> tip.fill()
        >>> tip.refit() # optional

        Returns
        -------
        Wingtip = the wingtip

        '''

        tip = Wingtip(self, self.half)
        self.tip = tip
        return tip

    def generate_junction(self, S, tip=False, watertight=False):

        ''' Generate a Junction.

        Should you later desire to discard the Junction, simply reset
        Wing.junctions[X] equal to None, where X is 0 for the root
        Junction and 1 otherwise.  You may also want to call the
        `untrim' method on the Wing's halves.

        Parameters
        ----------
        S = the Surface with which to intersect the Wing
        tip = whether or not the junction is located at the Wing's tip
        watertight = whether or not to use a WatertightJunction instead
                     of a Junction (experimental)

        Returns
        -------
        Junction = the (trimmed) junction

        Examples
        --------
        >>> jnc = wi.generate_junction(S)
        >>> jnc.attach()
        >>> jnc.extend()
        >>> jnc.intersect()
        >>> jnc.trim()

        '''

        args = self, S, self.half, tip
        jnc = (TrimmedJunction if not watertight else
               WatertightJunction)(*args)
        self.junctions[0 if not tip else 1] = jnc
        return jnc

    def generate_structure(self):

        ''' Generate a WingStructure.

        Should you later desire to discard the WingStructure, simply
        reset Wing.structure equal to None.

        Examples
        --------
        >>> stc = wi.generate_structure()
        >>> stc.generate_spars()
        >>> stc.generate_ribs()

        Returns
        -------
        WingStructure = the wing structure

        '''

        stc = WingStructure(self)
        self.structure = stc
        return stc

# GLUING AND DRAWING

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(Wing, self)._glue(parent)
        g = []
        if self.T:
            g += [self.T]
        if self.nurbs:
            g += [self.nurbs, self.LE, self.TE, self.QC] + self.halves
        if self.tip:
            g += self.tip._glue(self)
        if self.structure:
            g += self.structure._glue(self)
        return g

    def _draw(self):
        ''' See Part._draw. '''
        d = []
        if self.nurbs:
            hs = list(self.halves)
            if self.half in (0, 1):
                hs.pop((self.half + 1) % 2)
            d += hs
        else:
            d += [self.T]
        if self.tip:
            d += [self.tip]
        if self.structure:
            d += [self.structure]
        return d


class WingMerger(Part):

    ''' Merge Wing objects one after the other.  Useful to generate any
    wing whose spanwise variation is discontinuous, e.g. cranked-wings,
    C-wings, boxed-wings, etc.

    Intended usage
    --------------
    >>> wm = WingMerger(w0, w1, w2) # w0, w1 and w2 being Wing objects
    >>> wm.merge()
    >>> wm0, wm1, wm2 = wm.merged_wings

    '''

    def __init__(self, *wings):

        ''' Instantiate the WingMerger with Wing objects.

        The Wings must be merge-compatible, that is a Wing's tip Airfoil
        (or root Airfoil if there is no tip Airfoil) must share the same
        fit as the next Wing's root Airfoil (the Airfoils need not be
        superimposed though).  Also, scaling and/or twisting, if any,
        should match exactly at the common end of the Wings.

        Parameters
        ----------
        wings = an (ordered) list of Wings to be merged

        '''

        if not len(wings) > 1:
            raise NeedMoreThanOneWingToMerge()

        for w in wings:
            check_wing(w)
        check_wings_merge_compatible(wings)

        self.wings = [w.copy() for w in wings] # the list of original Wings
        self.merged_wings = [] # the list of merged Wings

    def _translate(self, w0, w1):
        ''' Translate w1's root QC at w0's tip QC. '''
        O = w0.QC.eval_point(1)
        w = O - w1.QC.eval_point(0)
        if (w != 0.0).any():
            w1.glue()
            w1.translate(w)
            w1.unglue()
            print('geom.wing.WingMerger.merge :: '
                  'Wing half translated by {}'.format(w))

    def _segment(self, w0, w1, hf0, hf1, l):
        ''' Segment (see nurbs.surface.Surface.segment) the two halves
        w0.halves[hf0] and w1.halves[hf1] at their commond end by l. '''

        Cas = []
        if l is not None:

            for w, hf in zip((w0, w1), (hf0, hf1)):
                if w.halves[hf].p[1] == 1:
                    w.halves[hf] = w.halves[hf].elevate(1, 1)
                    print('geom.wing.WingMerger.merge :: '
                          'Wing halve object degree elevated by 1.')

            h0, h1 = w0.halves[hf0], w1.halves[hf1]

            L = param_to_arc_length(w0.QC)
            ui0 = arc_length_to_param(w0.QC, L - l)
            ui1 = arc_length_to_param(w1.QC, l)

            h0 = h0.segment(0, 1, ui0, 1)
            h1 = h1.segment(0, 1, 0, ui1)

            Ca0 = h0.extract(ui0, 1)
            Ca1 = h1.extract(ui1, 1)

            Cas += [Ca0, Ca1]

        else:
            h0, h1 = w0.halves[hf0], w1.halves[hf1]

        h0, h1 = make_surfaces_compatible3([h0, h1], di=0)
        w0.halves[hf0], w1.halves[hf1] = h0, h1

        return Cas

    def _shear(self, w0, w1, hf0, hf1):
        ''' Shear the last and first rows of control points pertaining
        to w0.halves[hf0] and w1.halves[hf1], respectively.  Both rows
        should then coincide exactly. '''

        h0, h1 = w0.halves[hf0], w1.halves[hf1]

        Pw0 = h0.cobj.Pw[:,-1]
        Pw1 = h1.cobj.Pw[:, 0]

        O = w0.QC.eval_point(1)

        Y0 = w0.T.eval_derivatives(1, 1)[1]
        Y1 = w1.T.eval_derivatives(0, 1)[1]

        Yp0 = point_to_plane((0,0,0), (1,0,0), Y0)
        Yp1 = point_to_plane((0,0,0), (1,0,0), Y1)

        Z0 = np.cross((1,0,0), Yp0)
        Z1 = np.cross((1,0,0), Yp1)

        phi = signed_angle(Z0[1:], Z1[1:]) / 2.0

        if phi != 0.0 and phi != 180.0:
            shear(Pw0, - phi, Z0, Yp0, O)
            shear(Pw1,   phi, Z1, Yp1, O)

        if not np.allclose(Pw0, Pw1):
            m = np.max(Pw0 - Pw1)
            print('geom.wing.WingMerger.merge :: '
                  'warning, misaligned control points ({})'.format(m))
        else:
            Pw0 = Pw1

    def merge(self, l=None, show=True):

        ''' Merge all Wings in sequential order.

        This is done by first forcing the to-be merged Wing to be
        spatially compatible with the last merged Wing, and then by
        finding the intersection Curve between the two.  In general,
        such an intersection Curve does not exist, and thus an
        approximation to the Wing's Surfaces immediately before and
        after the intersection is built.  Outside that interval
        (delimited by the Curves drawn if letting show=True) the Wing's
        Surfaces are kept intact.

        Note that the merged Wings are *copies* of those used to
        instantiate the WingMerger; they are however stored in the same
        order.

        Parameters
        ----------
        l = if not None, the (approximate) length before and after which
            two merged Wings are approximated
        show = whether or not to draw the merged Wings with Curves
               delimiting their reapproximations

        Returns
        -------
        fig = a Figure

        '''

        self.merged_wings = []

        Cas = []
        for w1 in self.wings:

            w1._halve(w1.nurbs)

            if not self.merged_wings:
                self.merged_wings.append(w1)
                continue

            w0 = self.merged_wings[-1]
            self.merged_wings.append(w1)

            afr0, aft0 = w0.airfoils
            afr1, aft1 = w1.airfoils

            n0 = aft0.nurbs if aft0 else afr0.nurbs
            n1 = afr1.nurbs

            for hf in (0, 1):

                # Step 1: translate w1, if necessary
                self._translate(w0, w1)

                # Step 2: force the approximation to stay within l (wrt the QC)
                Cas += self._segment(w0, w1, hf, hf, l)

                # Step 3: shear w0 and w1 equally, if necessary
                self._shear(w0, w1, hf, hf)

        for hf in (0, 1):
            Ns = [mw.halves[hf] for mw in self.merged_wings]
            Ns = make_surfaces_compatible3(Ns, di=0)
            for mw, N in zip(self.merged_wings, Ns):
                mw.halves[hf] = N

        for mw in self.merged_wings:
            mw._size()

        if show:
            return draw(self, *Cas)

    def _glue(self, parent=None):
        ''' See Part._glue. '''
        super(WingMerger, self)._glue(parent)
        g = []
        for mw in self.merged_wings:
            g += mw._glue(self)
        return g

    def _draw(self):
        ''' See Part._draw. '''
        return self.merged_wings


class HalfWingMerger(WingMerger):

    ''' Same as WingMerger, only here only one half of each Wing is
    modified, the other is left untouched.

    NOTE: Does not support Wings of nonzero twist.

    '''

    def _reverse_half(self, wi, hf):
        wi.T = wi.T.reverse()
        wi.QC = wi.QC.reverse()
        wi.halves[hf] = wi.halves[hf].reverse(1)

    def merge(self, hf, reverse, l=None, show=True):

        ''' Similar to WingMerger.  The idea here is however to merge
        only one half of each Wing, thus allowing the other half of a
        merged Wing to be reused in a subsequent merge with another
        HalfWingMerger.  This design allows to build Wings of virtually
        any topology.

        By default each Wing's root is merged at the tip of the previous
        Wing's tip; if this is not possible or the present Wings'
        orientation do not allow for it, the user can specify so by
        means of the `reverse` list (see example below).

        Parameters
        ----------
        hf = the half of the Wings to be merged (0 for lower or 1 for
             upper)
        reverse = the boolean list specifying whether or not a Wing's
                  orientation should be temporarily reversed
        l = same as WingMerger
        show = same as WingMerger

        Returns
        -------
        fig = a Figure

        Examples
        --------
        >>> w1 = w0.copy() # w0 being a Wing object
        >>> w2 = w0.copy()
        >>> w1.glue(); w1.dihedral = 90; w1.unglue()

        >>> hwm = HalfWingMerger(w0, w1)
        >>> hwm.merge(0, [False, True], None)
        >>> w0m, w1m = hwm.merged_wings

        >>> hwm = HalfWingMerger(w1m, w2)
        >>> hwm.merge(0, [False, False], None)
        >>> w1m, w2m = hwm.merged_wings

        '''

        self.merged_wings = []

        Cas, hfo = [], np.mod(hf + 1, 2)
        for w1, r in zip(self.wings, reverse):

            h1s = w1.halves
            w1._halve(w1.nurbs)

            hf1 = hf if r else hfo
            w1.halves[hf1] = h1s[hf1]

            if not self.merged_wings:
                if r:
                    self._reverse_half(w1, hfo if r else hf1)
                self.merged_wings.append(w1)
                continue

            w0 = self.merged_wings[-1]
            self.merged_wings.append(w1)

            afr0, aft0 = w0.airfoils
            afr1, aft1 = w1.airfoils

            af0 = aft0 if aft0 else afr0
            af1 = afr1

            if not (af0.issymmetric or af1.issymmetric):
                print('geom.wing.WingMerger.merge :: '
                      'warning, tip airfoil(s) not symmetric')

            hf0 = hfo if (w1 is self.wings[1] and reverse[0]) else hf
            hf1 = hfo if r else hf

            if r: self._reverse_half(w1, hf1)

            # Step 1: translate w1, if necessary
            self._translate(w0, w1)

            # Step 2: force the approximation to stay within l (wrt the QC)
            Cas += self._segment(w0, w1, hf0, hf1, l)

            # Step 3: shear w0 and w1 equally, if necessary
            self._shear(w0, w1, hf0, hf1)

            if r: self._reverse_half(w1, hf1)

        Ns = []
        for mw, r in zip(self.merged_wings, reverse):
            Ns.append(mw.halves[hfo if r else hf])

        Ns = make_surfaces_compatible3(Ns, di=0)

        for mw, r, N in zip(self.merged_wings, reverse, Ns):
            mw.halves[hfo if r else hf] = N

        if reverse[0]:
            w0 = self.merged_wings[0]
            self._reverse_half(w0, hfo)

        if show:
            return draw(self, *Cas)


class BSplineFunctionCreator(PlotObject):

    ''' Create a B-spline function.

    A B-spline function, B(v), is a Curve where only the first
    coordinate is considered.  It is typically used during the sweeping
    and/or orienting process of a Wing, where instances of an Airfoil
    are scaled and/or twisted depending on their locations along a
    trajectory Curve T (see Wing.orient and Wing.fit).  For example, an
    Airfoil's characterizing Curve whose associated parameter value on T
    is (v = 0.3) would be scaled/twisted by the value returned by
    B(0.3), i.e. B.eval_point(0.3)[0].

    '''

    pass


class BSplineFunctionCreator1(BSplineFunctionCreator):

    ''' Create a Type 1 B-spline function B(v).

    Just like any y = f(x) function, a Type 1 B-spline scaling function
    is best visualized in a XY coordinate system; here the X axis refers
    to v, (0 <= v <= 1), and the Y axis to B.  The idea is to design a
    so-called "representation Curve" in this plane, sample it at evenly
    spaced parameter values, and then finally fit the sampled points to
    retrieve the actual B(v).  The best way to understand this is to
    simply try it.

    Intended usage
    --------------
    >>> Bs = BSplineFunctionCreator1(end=(1.0,2.0))
    >>> Bs.design() # optional
    >>> B = Bs.fit()

    '''

    def __init__(self, end=(1.0, 1.0), p=1, n=2):

        ''' Instantiate a default representation Curve that linearly
        interpolates the end point values.

        Parameters
        ----------
        end = the end point values
        p, n = the degree and number of control points to define the
               representation Curve with

        '''

        Pw = np.zeros((n, 4)); Pw[:,-1] = 1.0
        Pw[ 0,1] = end[0]
        Pw[-1,1] = end[1]; Pw[-1,0] = 1.0
        if n > 2:
            xs = np.linspace(0, 1, n)
            ys = np.linspace(end[0], end[1], n)
            Pw[1:-1,0] = xs[1:-1]
            Pw[1:-1,1] = ys[1:-1]
        self.R = Curve(ControlPolygon(Pw=Pw), (p,))
        self._p = p

    def design(self):

        ''' Design the representation Curve interactively.  The first
        and last control points are constrained to the (x = 0) and (x =
        1) lines, respectively.

        '''

        cpts = self.R.cobj.cpts
        cpt0, cpt1 = cpts[0], cpts[-1]
        cpt0.line = (0,0,0), (0,1,0)
        cpt1.line = (1,0,0), (0,1,0)
        fig = draw(self.R)
        fig.c.setup_preset(r=(0,0,0))
        return fig

    def fit(self, r=100, n=50):

        ''' Sample and fit the representation Curve.

        Parameters
        ----------
        r = the number of points to sample the representation Curve with
        n = the number of control points to fit the sampled points with

        Returns
        -------
        B = the B-spline function B(v)

        '''

        us = np.linspace(0, 1, r)
        Q0 = self.R.eval_points(us).T
        Q = np.zeros_like(Q0)
        uk, Q[:,0] = Q0[:,0], Q0[:,1]
        U, Pw = global_curve_approx_fixedn(r - 1, Q, self._p, n, uk)
        return Curve(ControlPolygon(Pw=Pw), (self._p,), (U,))


class BSplineFunctionCreator2(BSplineFunctionCreator):

    ''' Create a Type 2 B-spline function B(v).

    The Type 1 approach is best suited for quick evaluation of simple
    B-spline functions.  If precise control is required (e.g. when
    designing a custom blended wing-body), it is often easier to first
    design the outer planform shape of a Wing, and to infer a B-spline
    scaling function from it (in which case no representation Curve is
    necessary).  Thus, here all that is required is to manipulate, in
    the XY plane, a TE Curve w.r.t. a fixed one representing a Wing's
    LE.  If desired the same TE Curve can also be manipulated in the YZ
    plane to come up with an independent B-spline function for vertical
    scaling.

    '''

    def __init__(self, T):

        ''' Create the TE planform Curve from a LE trajectory Curve, and
        translate it by 1 unit in the chordwise and vertical directions.

        Parameters
        ----------
        T = the trajectory Curve used to orient a Wing

        '''

        Ts = [T.copy(), T.copy()]
        Ts[1].translate([1,0,0])
        self.Ts = Ts

    def design(self):

        ''' Design a Wing's planform by manipulating the TE Curve in the
        XY plane (and/or the YZ plane if vertical scaling is desired).
        The LE Curve should remain untouched.

        '''

        T0, T1 = self.Ts
        cpts = [cpt for T in (T0, T1) for cpt in T.cobj.cpts]
        for cpt in cpts:
            if hasattr(cpt, 'line'):
                del cpt.line
            cpt.plane = cpt.xyz, (0,1,0)
        T1.colorize()
        fig = draw(T0, T1)
        fig.c.setup_preset('xy')
        return fig

    def fit(self, r=100, n=50, di=0):

        ''' Sample and fit a B-spline function that shall scale a Wing
        such that its XY planform view matches exactly the previously
        designed LE and TE Curves.

        Parameters
        ----------
        r = the number of points to sample the LE and TE Curves with
        n = the number of control points to fit the sampled points with
        di = the sampling direction (0 or 2)

        Returns
        -------
        B = the B-spline function B(v)

        '''

        T0, T1 = self.Ts
        us = np.linspace(0, 1, r)
        Q0 = T0.eval_points(us).T
        Q1 = T1.eval_points(us).T
        Q = np.zeros_like(Q0)
        Q[:,0] = Q1[:,di] - Q0[:,di]
        U, Pw = global_curve_approx_fixedn(r - 1, Q, T0.p[0], n, us)
        return Curve(ControlPolygon(Pw=Pw), (T0.p[0],), (U,))

    def match(self):

        '''

        '''

        T0, T1 = self.Ts
        for cpt0, cpt1 in zip(T0.cobj.cpts, T1.cobj.cpts):
            dxyz = cpt1.xyz - cpt0.xyz
            cpt1._xyzw[2] = cpt0._xyzw[2] + dxyz[0]
        update_figures([T1])


class BSplineFunctionCreator3(BSplineFunctionCreator):

    ''' Create a Type 3 B-spline function B(v).

    This creator is used solely to help design a smooth transitional
    Wing sandwiched between two closely spaced Wings, such as is the
    case when going from a main Wing to a vertical Wing extension
    (winglet).

    NOTE: Does not support Wings of nonzero twist.

    '''

    def __init__(self, w0, w1):

        ''' Copy and store the two Wings to transition.

        Parameters
        ----------
        w0, w1 = the two (previously fitted) Wings

        '''

        self.ws = [w.copy() for w in w0, w1]
        self.Ts = [None, None]

    def design(self):

        ''' Automatically create LE and TE Curve extensions attaching
        the two Wings.  If not satisfied, the Wing objects can be
        repositioned before repeating the process.

        '''

        w0, w1 = self.ws

        Q0, D0 = w0.LE.eval_derivatives(1, 1)
        Q1, D1 = w1.LE.eval_derivatives(0, 1)

        U, Pw = local_curve_interp(1, (Q0, Q1), D0, D1)
        T0 = Curve(ControlPolygon(Pw=Pw), (3,), (U,))

        Q0, D0 = w0.TE.eval_derivatives(1, 1)
        Q1, D1 = w1.TE.eval_derivatives(0, 1)

        U, Pw = local_curve_interp(1, (Q0, Q1), D0, D1)
        T1 = Curve(ControlPolygon(Pw=Pw), (3,), (U,))

        self.Ts = [T0, T1]
        return draw(w0, w1, T0, T1)

    def fit(self, r=100, n=50):

        ''' Sample and fit a B-spline function that shall fit a Wing
        such that it smoothly connects two other closely distanced
        Wings.

        Parameters
        ----------
        r = the number of points to sample the Curves extensions with
        n = the number of control points to fit the sampled points with

        Returns
        -------
        T = the trajectory Curve to use for the Wing transition
        B = the B-spline function B(v)

        '''

        T0, T1 = self.Ts
        T = T0.copy()
        #T.cobj.Pw[:] = 0.75 * T0.cobj.Pw + 0.25 * T1.cobj.Pw
        us = np.linspace(0, 1, r)
        Q0 = T0.eval_points(us).T
        Q1 = T1.eval_points(us).T
        Q = np.zeros_like(Q0)
        Q[:,0] = Q1[:,0] - Q0[:,0]
        U, Pw = global_curve_approx_fixedn(r - 1, Q, 3, n, us)
        return T, Curve(ControlPolygon(Pw=Pw), (3,), (U,))


# UTILITIES


def check_airfoil(af):
    if not af.nurbs:
        raise UnfitAirfoil(af)

def check_wing(wi):
    if not wi.nurbs:
        raise UnfitWing(wi)

def check_wings_merge_compatible(wis):
    for w0, w1 in zip(wis[:-1], wis[1:]):
        ar0, at0 = w0.airfoils
        ar1, at1 = w1.airfoils
        a0 = at0 if at0 else ar0
        a1 = ar1
        if a0.name != a1.name:
            raise MergeIncompatibleWings(w0, w1)
        n0, n1 = a0.nurbs, a1.nurbs.copy()
        w = n0.eval_point(0) - n1.eval_point(0)
        n1.translate(w)
        if not n0.isequivalent(n1):
            raise MergeIncompatibleWings(w0, w1)
        Tw0, (scx0, dummy, scz0) = w0.Tw, w0.scs
        Tw1, (scx1, dummy, scz1) = w1.Tw, w1.scs
        v0 = v1 = 0.0
        for f0, f1 in (Tw0, Tw1), (scx0, scx1), (scz0, scz1):
            if f0: v0 = f0.eval_point(1)[0]
            if f1: v1 = f1.eval_point(0)[0]
            if not np.allclose(v0, v1):
                raise MergeIncompatibleWings(w0, w1)
            v0 = v1 = 1.0

def get_sweep_local_sys_proj(v, T, Bv):
    ''' Idem nurbs.surface.get_sweep_local_sys, however here Y is first
    projected onto the (0,1,0) plane. '''
    O, Y = T.eval_derivatives(v, 1); Y[0] = 0.0
    Z = Bv.eval_point(v)
    X = np.cross(Y, Z)
    X, Y, Z = [normalize(V) for V in X, Y, Z]
    return O, X, Y, Z


# EXCEPTIONS


class WingException(Exception):
    pass

class UnfitWing(WingException):
    pass

class UnorientedWing(WingException):
    pass

class InvalidSweepAngle(WingException):
    pass

class NeedMoreThanOneWingToMerge(WingException):
    pass

class MergeIncompatibleWings(WingException):
    pass
