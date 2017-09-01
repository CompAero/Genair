#import multiprocessing

import numpy as np

from nurbs.nurbs   import (NURBSObject,
                           NewtonLikelyDiverged)
from nurbs.point   import Point
from nurbs.volume  import (Volume,
                           ControlVolume,
                           volume_point_projection)
from nurbs.util    import (distance,
                           distance_v,
                           construct_flat_grid)

from plot.pobject  import update_figures


__all__ = ['FFDVolume', 'make_ffd_volume']


class FFDVolume(Volume):

    ''' Conceptually, FFD is best visualized as embedding a flexible,
    rubber-like object into a transparent material having the same
    constitutive properties.  As the larger block deforms, so will the
    embedded object.  Here, the trivariate NURBS Volume is utilized as
    the embedding material.

    Formally, FFD is achieved by two functions.  The first one is the
    embedding function, which associates a parametric value to each
    vertex of an object.  This procedure needs only be performed once.
    Note that here the embedded vertices are taken to be the *control
    points* that define the object, not the surface points that
    discretize it as is usually the case.  The second function is the
    deformation function, which translates to simply reevaluating each
    vertex once the FFDVolume's own lattice has deformed.

    '''

    def __init__(self, cvol, p, U=None):

        ''' Instantiate an FFDVolume, see also make_ffd_volume.

        Parameters
        ----------
        Same as nurbs.nurbs.Volume.

        '''

        super(FFDVolume, self).__init__(cvol, p, U)
        self._setup()

    def __setstate__(self, d):
        ''' Unpickling. '''
        super(FFDVolume, self).__setstate__(d)
        self._setup()

    def _setup(self):
        ''' Perform some preliminary setup checks. '''
        self.embedded = {} # self.embedded[Point] = (u, v, w)
        self.ffd = self
        for cpt in self.cobj.cpts.flat:
            cpt.ffd = self

    def _brute(self, num):
        ''' Discretize the FFDVolume by brute forcing. '''
        global ffd, args, bmin, bmax, uvws, xyzs
        ffd = self; args = self.var()
        bs = self.bounds
        bmin, bmax = [np.asfarray(b) for b in zip(*bs)]
        us = construct_flat_grid(self.U, 3 * (num,))
        uvws = us; xyzs = self.eval_points(*us)

    def _get_pts(self, embeddables):
        ''' Get all 3D and control points from all embedabbles. '''
        pts = set()
        for e in embeddables:
            if isinstance(e, Point):
                pt = {e}
            elif isinstance(e, NURBSObject):
                pt = e.cobj.cpts.flatten()
            elif hasattr(e, '_draw'):
                pt = self._get_pts(e._draw())
            else:
                print('ffd.ffd.FFDVolume._get_pts :: '
                      'not embeddable ({})'.format(e))
            pts.update(pt)
        return pts

    def embed(self, *embeddables, **kwargs):

        ''' Embed any number of embeddables, i.e. nurbs.nurbs.Points,
        nurbs.nurbs.NURBSObjects, and/or geom.part.Parts, inside the
        FFDVolume using Newton's method.  Again, in all cases only
        control Points are embed, except for 3D Euclidean Points which
        are embed as is.

        Parameters
        ----------
        embeddables = the objects to be embed
        eps = the measure of convergence employed in Newton's method
              (default: 1e-12)
        num = the number of points in each direction to brute force the
              FFDVolume with during Newton's method startup (default:
              70)

        Returns
        -------
        fails = the number of (control) Points that could not be embed

        '''

        global eps; eps = kwargs.get('eps', 1e-12)
        num = kwargs.get('num', 70)
        self._brute(num)

        pts = self._get_pts(embeddables)
        pts = list(pts)

        # fix for Windows
        uvws = []
        for pt in pts:
            uvw = embed_point(pt)
            uvws.append(uvw)

        # run Newton's method on as many cores as possible
        #pool = multiprocessing.Pool()
        #uvws = pool.map(embed_point, pts)
        #pool.close()
        #pool.join()

        fails = uvws.count(0)
        nz = [i for i, uvw in enumerate(uvws) if uvw]
        pts, uvws = np.array(pts)[nz], np.array(uvws)[nz]
        self.unembed(*pts)
        embedded = dict(zip(pts, uvws))
        self.embedded.update(embedded)
        for pt in pts:
            pt._color[:] = (0, 255, 0, 255)
            pt.embed = self
        update_figures(pts)
        return fails

    def unembed(self, *embeddables):

        ''' Unembed any number of embeddables already embed in the
        FFDVolume.  This also resets their colors.

        Parameters
        ----------
        embeddables = the objects to unembed

        '''

        pts = self._get_pts(embeddables)
        for pt in pts:
            if hasattr(pt, 'embed'):
                del pt.embed.embedded[pt]
                del pt.embed
                pt.colorize(reset=True, fill_batch=False)
        update_figures(pts)

    def refresh(self):

        ''' Refresh the FFDVolume's embedded (control) Points by
        reevaluating them at their respective parametric values.  This
        should be done every time the FFD's ControlVolume has deformed.
        When used interactively, i.e. in conjunction with
        plot.mode.TranslatePointMode, this function is in fact called
        automatically.

        '''

        if not self.embedded:
            return
        pts, us = zip(*self.embedded.items())
        us = zip(*us); xyzs = self.eval_points(*us).T
        ns = set()
        for pt, xyz in zip(pts, xyzs):
            x, y, z = xyz; w = pt._xyzw[-1]
            pt._xyzw[:] = w * x, w * y, w * z, w
            if hasattr(pt, 'nurbs'):
                ns.add(pt.nurbs)
        for n in ns:
            n._fill_batch()


def make_ffd_volume(p=(2,2,2), n=None, bounds=[(-1,1),(-1,1),(-1,1)],
                    offset=0.0):

    ''' Convenience function to spawn an FFDVolume over a specifed xyz
    domain.

    Parameters
    ----------
    p = the u, v, w degrees of the FFDVolume
    n = the u, v, w number of control points; if not specified, a Bezier
        FFDVolume is assumed (n = p + 1)
    bounds = the x, y, z bounds of the FFDVolume
    offset = the offset to subtract/add from the bounds

    Returns
    -------
    FFDVolume = the newly spawned FFDVolume

    '''

    if n is None:
        n, m, l = [d + 1 for d in p]
    else:
        n, m, l = n
    nj, mj, lj = [eval(str(o) + 'j')
                  for o in (n, m, l)]
    xlo, xhi, ylo, yhi, zlo, zhi = \
        [eval(str(e) + s + str(offset))
         for bound in bounds for e, s in zip(bound, ('-', '+'))]
    xs, ys, zs = np.mgrid[xlo:xhi:nj,
                          ylo:yhi:mj,
                          zlo:zhi:lj]
    Pw = np.zeros((n, m, l, 4))
    Pw[...,0] = xs
    Pw[...,1] = ys
    Pw[...,2] = zs
    Pw[...,3] = 1.0
    return FFDVolume(ControlVolume(Pw=Pw), p)


# UTILITIES


def embed_point(point):
    ''' Embed a nurbs.point.Point inside an FFDVolume. '''
    xyz = point.xyz
    if (bmin <= xyz).all() and (xyz <= bmax).all():
        # NOTE: it is possible to get here even though the Point
        # lies outside the FFD's convex hull; in that case it is
        # normal for the assertion below to fail
        ds = distance_v(xyzs, xyz)
        i = np.argmin(ds)
        us = (u[i] for u in uvws)
        try:
            args2 = args + (xyz, us, eps)
            uvw = volume_point_projection(*args2)
            l2n = distance(ffd.eval_point(*uvw), xyz)
            assert l2n <= eps
        except (NewtonLikelyDiverged, AssertionError):
            return 0
        else:
            print('ffd.ffd.embed_point :: '
                  'embed {:10} successfully ({})'.format(point, l2n))
            return uvw
