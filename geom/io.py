import cPickle
import gzip

import numpy as np

from nurbs.curve   import (Curve,
                           ControlPolygon,
                           param_to_arc_length)
from nurbs.iges    import IGESFile
from nurbs.knot    import normalize_knot_vec
from nurbs.nurbs   import (obj_mat_to_3D,
                           obj_mat_to_4D)
from nurbs.surface import (Surface,
                           ControlNet)
from nurbs.util    import construct_flat_grid
from nurbs.volume  import (Volume,
                           ControlVolume)
from part          import _explode


__all__ = ['save', 'load',
           'save_IGES', 'load_IGES',
           'save_TECPLOT', 'load_TECPLOT']


def save(*os, **kwargs):

    ''' Save (pickle) any number of objects.

    Parameters
    ----------
    os = the objects to be pickled
    fn = the name of the pickle (default: 'tmp.p')

    Returns
    -------
    True = on success

    '''

    fn = kwargs.get('fn', 'tmp.p')
    with gzip.open(fn, 'wb') as fh:
        for o in os:
            cPickle.dump(o, fh, - 1)
    return True

def load(fn='tmp.p'):

    ''' Load (unpickle) a pickle.

    Parameters
    ----------
    fn = the name of the pickle (default: 'tmp.p')

    Returns
    -------
    os = the unpickled objects

    '''

    with gzip.open(fn) as fh:
        os = []
        while True:
            try:
                o = cPickle.load(fh)
                os.append(o)
            except EOFError:
                break
        return os

def save_IGES(*os, **kwargs):

    ''' Save, in IGES format, any number of Points, Curves and/or
    Surfaces, respectively corresponding to entities 116, 126 and 128.

    If a Part is passed to this function only what is returned by its
    '_draw' method will be saved, not the Part itself (unlike a pickle).

    Parameters
    ----------
    os = the objects to be saved
    fn = the name of the IGES file (default: 'tmp.igs')

    Returns
    -------
    True = on success

    '''

    fn = kwargs.get('fn', 'tmp.igs')
    I = IGESFile(fn)
    to_unparse = _explode(*os)
    return I.unparse(to_unparse)

def load_IGES(fn='tmp.igs'):

    ''' Load an IGES file.  Entities other than 116, 126 and 128 will be
    ignored.

    Parameters
    ----------
    fn = the name of the IGES file (default: 'tmp.igs')

    Returns
    -------
    os = the loaded objects

    '''

    I = IGESFile(fn)
    return I.parse()

def save_TECPLOT(*os, **kwargs):

    ''' Save, in TECPLOT format, any number of NURBSObjects (Curves,
    Surfaces and/or Volumes).  Note that only the discrete form (or
    optionally the ControlObject) of a NURBSObject is saved, *not* its
    exact representation.

    Parameters
    ----------
    os = the objects to be saved
    cobj = whether or not to store the ControlObjects of the
           NURBSObjects rather than their discretizations (default:
           False)
    ppu - 1 = if (cobj == False), the approximate number of points per
              unit to discretize the NURBSObjects with (default: 50)
    fn = the name of the TECPLOT file (default: 'tmp.dat')

    Returns
    -------
    True = on success

    '''

    fn = kwargs.get('fn', 'tmp.dat')
    cobj = kwargs.get('cobj', False)
    ppu = kwargs.get('ppu', 50)
    to_write = _explode(*os)
    with open(fn, 'w') as fh:
        fh.write('TITLE = "' + fn + '"\n')
        fh.write('FILETYPE = GRID\n')
        fh.write('VARIABLES = "X" "Y" "Z"\n')
        for o in to_write:
            if not isinstance(o, (Curve, Surface, Volume)):
                print('geom.io.save_TECPLOT :: '
                      'unrecognized object ({}), ignoring.'.format(o))
                continue
            o = o.copy()
            U = o.U; nu = len(U)
            if not cobj:
                for u in U:
                    normalize_knot_vec(u)
                nums = ppu * estimate_length(o)
                nums = nums.astype('i')
                print('geom.io.save_TECPLOT :: '
                      'object discretized with {} nodes'.format(nums))
                us = construct_flat_grid(U, nums)
                us = us[::-1]
                nums = nums[::-1]
                Q = o.eval_points(*us).T
            else:
                ns = o.cobj.n
                nums = np.array(ns) + 1
                Q = obj_mat_to_3D(o.cobj.Pw)
                if isinstance(o, Surface):
                    Q = np.transpose(Q, (1, 0, 2))
                elif isinstance(o, Volume):
                    Q = np.transpose(Q, (2, 1, 0, 3))
                Q = Q.reshape((-1, 3))
            placeholders= ['{' + str(i) + '[0]} = {' + str(i) + '[1]}'
                           for i in range(nu)]
            Imax = ','.join(placeholders).format(*zip('IJK'[:nu], nums))
            fh.write('ZONE\n')
            fh.write(Imax + ', DATAPACKING = POINT\n')
            for xyz in Q:
                xyz.tofile(fh, ' '); fh.write('\n')
    return True


def load_TECPLOT(fn='surfCPs.dat'):

    ''' Load a TECPLOT file generated by Jetstream.  Only ZONES
    containing control point xyz coordinates are read in; thus, this
    function only returns Curves, Surfaces and/or Volumes.  Uniform knot
    vectors and cubic splines are assumed for all entities.

    Parameters
    ----------
    fn = the name of the TECPLOT file (default: 'surfCPs.dat')

    Returns
    -------
    os = the loaded objects

    '''

    Os = []
    with open(fn) as fh:
        fh.readline()
        fh.readline()
        while True:
            ZONE = fh.readline()
            if not ZONE:
                break
            elif ('ZONE' not in ZONE or
                  'Surface Points' in ZONE):
                continue
            IJK = ZONE.split(',')[1:-1]
            IJK = [int(i.split()[-1]) for i in IJK]
            I = np.prod(IJK)
            P = np.zeros(I*3)
            for i in xrange(I*3):
                P[i] = float(fh.readline())
            for i in xrange(I*2):
                fh.readline()
            newshape = np.append(3, IJK[::-1])
            P = P.reshape(newshape)
            ndim = P.ndim - 1
            newaxes = range(ndim, -1, -1)
            P = np.transpose(P, newaxes)
            Pw = obj_mat_to_4D(P)
            if ndim == 1:
                O, Cobj = Curve, ControlPolygon
            elif ndim == 2:
                O, Cobj = Surface, ControlNet
            elif ndim == 3:
                O, Cobj = Volume, ControlVolume
            p = [min(i - 1, 3) for i in IJK]
            o = O(Cobj(Pw=Pw), p)
            Os.append(o)
            print('geom.io.load_TECPLOT :: '
                  'ncpts = {}, degrees = {}'.format(o.cobj.n, o.p))
        return Os


# UTILITIES


def estimate_length(o):
    if isinstance(o, Curve):
        lu = param_to_arc_length(o)
        return np.array((lu,))
    elif isinstance(o, Surface):
        Cs = [o.extract(0, 1), o.extract(1, 1),
              o.extract(0, 0), o.extract(1, 0)]
        ls = [param_to_arc_length(c) for c in Cs]
        lu, lv = ls[:2], ls[2:]
        lu, lv = np.average(lu), np.average(lv)
        return np.array((lv, lu))
    elif isinstance(o, Volume):
        return np.array((10, 10, 10)) # not implemented
