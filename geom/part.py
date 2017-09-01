from copy            import deepcopy

from nurbs.point     import Point
from nurbs.surface   import Surface
from nurbs.transform import (translate,
                             rotate,
                             mirror,
                             scale,
                             shear,
                             transform)
from nurbs.util      import bounds

from plot.pobject    import (PlotObject,
                             update_figures)


class Part(PlotObject):

    def __getstate__(self):
        ''' Pickling. '''
        d = self.__dict__.copy()
        if '_p' in d:
            del d['_p']
        if '_c' in d:
            del d['_c']
        if 'glued' in d:
            del d['glued']
        return d

    @property
    def bounds(self):
        ''' Return the Part's min/max bounds. '''
        os = _explode(self)
        return bounds(*os)

    def symmetrize(self):

        ''' Mirror the Part about the XZ symmetry plane.

        Returns
        -------
        Mirrors = a list of mirrored Points and/or NURBSObjects

        '''

        osm = []
        for o in _explode(self):
            pom = o.copy()
            if isinstance(o, Surface):
                pom = pom.swap()
                if o.istrimmed:
                    tcs = [tc.reverse() for tc in o._trimcurves]
                    for tc in tcs:
                        tc.mirror([1,-1,0])
                    pom._trimcurves = tcs
            pom.mirror(N=[0,1,0])
            pom.color = o.color
            osm.append(pom)
        return osm

    def colorize(self, reset=False):

        ''' Colorize the whole Part.

        Parameters
        ----------
        reset = whether or not to reset the default colors of all Points
                and/or NURBSObjects

        '''

        for o in _explode(self):
            o.colorize(reset)

    def clamp(self, tol=1e-5):

        ''' Clamp the Part onto the XZ symmetry plane.

        Parameters
        ----------
        tol = the tolerance below which a (control) Point's y-coordinate
              is forced to be exactly zero

        '''

        pts = set()
        for o in _explode(self):
            pt = [o] if isinstance(o, Point) else o.cobj.cpts.flat
            pts.update(pt)
        for pt in pts:
            wx, wy, wz, w = pt._xyzw
            if - tol < (wy / w) < tol:
                pt._xyzw[:] = wx, 0.0, wz, w
        update_figures(pts)

    def copy(self):
        ''' Self copy. '''
        return deepcopy(self)

    def _draw(self):
        ''' Set what is to be drawn, saved (IGES and TECPLOT),
        symmetrized, colorized, and clamped. '''
        pass

# GLUING

    def _glue(self, parent=None):
        ''' Update the family tree.  While a parent can have multiple
        childs, any given child can have one and only one parent.  Note
        that this method must be overriden and must return the Points
        and/or NURBSObjects to be glued by Part.glue. '''
        if parent:
            self._p = parent
            if not hasattr(self._p, '_c'):
                self._p._c = {self}
            else:
                self._p._c.add(self)

    def _unbind_parent(self):
        ''' Unbind self from its parent. '''
        glued = self._glue()
        if hasattr(self, '_p'):
            for o in glued:
                self.glued.remove(o)
            self._p._c.remove(self)
            if not self._p._c:
                del self._p._c
            del self._p
        return glued

    def glue(self):
        ''' Glue the Part i.e. establish parent/child relationships
        starting from self and onward.  If this or another Part from the
        same family is transformed (via Part.translate, Part.rotate,
        etc., or interactively through plot.mode.TranslateNURBSMode)
        then so is its parent (if any) and all of its childs (if any),
        e.g. once glued, translating a Wing will also translate its
        Wingtip and vice versa. '''

        def glue_childs(c, g):
            if hasattr(c, '_c'):
                for cc in c._c:
                    glue_childs(cc, g)
            c.glued = g

        glued = self._unbind_parent()
        glue_childs(self, glued)
        for o in glued:
            o.glued = glued

    def unglue(self):
        ''' Unglue the Part i.e. destroy all parent/child relationships
        starting from self and onward. '''

        def unglue_childs(c):
            if hasattr(c, '_c'):
                for cc in c._c:
                    unglue_childs(cc)
                del c._c
            if hasattr(c, '_p'):
                del c._p
            if hasattr(c, 'glued'):
                del c.glued

        glued = self._unbind_parent()
        unglue_childs(self)
        for o in glued:
            if hasattr(o, 'glued'):
                del o.glued

Part.translate = transform(translate)
Part.rotate    = transform(rotate)
Part.mirror    = transform(mirror)
Part.scale     = transform(scale)
Part.shear     = transform(shear)


# UTILITIES


def _explode(*os):
    ''' Explode all Parts/Points/NURBSObjects to Points/NURBSObjects
    only. '''
    eos = []
    for o in os:
        if isinstance(o, Part):
            eos += _explode(*o._draw())
        else:
            eos.append(o)
    return eos
