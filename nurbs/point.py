import numpy as np

import plot.pobject


__all__ = ['Point']


class Point(plot.pobject.PlotPoint):

    ''' A Point is defined in 4D homogeneous space.  Its purpose is
    twofold: to represent either a control Point, where then the last
    coordinate `w` may or may not be equal to one (but must be greater
    than zero), or more simply a generic 3D Point in Euclidean space, in
    which case it is customary to set `w` to one (default).

    '''

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):

        ''' Initialize the Point with zero or more of the four
        coordinates.  Defaults to the global origin.  Note that the x, y
        and z coordinates are automatically multiplied by the weight, so
        the formers need to be defined in 3D Euclidean space, which is
        more intuitive.

        '''

        super(Point, self).__init__()
        self._xyzw = np.asfarray([x * w, y * w, z * w, w])

    def __getstate__(self):
        ''' Pickling. '''
        d = self.__dict__.copy()
        ds = d.viewkeys() - {'_xyzw'}
        for k in ds:
            del d[k]
        return d

    def __setstate__(self, d):
        ''' Unpickling. '''
        self.__dict__.update(d)
        super(Point, self).__init__()

    @property
    def xyzw(self):
        ''' Get the xyzw coordinates.  The x, y and z coordinates are
        automatically divived by w. '''
        x, y, z, w = self._xyzw
        return np.asfarray([x / w, y / w, z / w, w])

    @xyzw.setter
    def xyzw(self, new_xyzw):
        ''' Set the xyzw coordinates.  The x, y and z coordinates are
        automatically multiplied by w.  Use None to keep one or more
        coordinates intact, e.g. pt.xyzw = None,2,3.4,None. '''
        x, y, z, w = new_xyzw
        if x is None: x = self.xyzw[0]
        if y is None: y = self.xyzw[1]
        if z is None: z = self.xyzw[2]
        if w is None: w = self.xyzw[3]
        self._xyzw[:] = np.asfarray([x * w, y * w, z * w, w])
        plot.pobject.update_figures([self])

    @property
    def xyz(self):
        ''' Get the xyz coordinates only. '''
        return self.xyzw[:3]

    @property
    def iscontrolpoint(self):
        ''' Is this a control Point? '''
        return hasattr(self, 'nurbs')

    @property
    def bounds(self):
        ''' Return the xyz coordinates, repeated. '''
        return [2 * (self.xyz[i],) for i in xrange(3)]

    def copy(self):
        ''' Self copy. '''
        return self.__class__(*self.xyzw)


def points_to_obj_mat(points):
    ''' Return a new object matrix given a list (Curve), a list of list
    (Surface) or a list of list of list (Volume) of Points. '''
    s = np.asarray(points, dtype='object').shape
    Pw = np.zeros(list(s) + [4])
    for i, point in np.ndenumerate(points):
        tmp = point._xyzw
        point._xyzw = Pw[i] # view
        point._xyzw[:] = tmp
        point._i = i
    return Pw

def obj_mat_to_points(Pw):
    ''' Idem points_to_obj_mat, vice versa. '''
    s = np.asfarray(Pw).shape[:-1]
    points = np.empty(s, dtype='object')
    for i in np.ndindex(s):
        points[i] = Point()
        points[i]._xyzw = Pw[i] # view
        points[i]._i = i
    return points
