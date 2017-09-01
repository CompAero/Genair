from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import pyglet


COLORMAP = {'3pts': (255, 255,   0, 255), # yellow
            'cpts': (255, 255, 255, 255), # white
            'fcpt': (255,   0, 255, 255), # fuchsia
            'scpt': (255,   0, 255, 125),
            'cobj': (255, 250, 255, 100),
            'crvs': (255,   0,   0, 255), # red
            'srfs': (204, 255,   0, 255), # lime green
            'vols': (  0, 125, 255, 255),
            'isoc': (  0, 255, 255, 255), # aqua
            'newp': (  0, 255,   0, 255), # green
            'pick': ( 64, 224, 208, 255)} # turquoise


def update_figures(pos):
    nurbs, ffds, figs = set(), set(), set()
    for po in pos:
        if isinstance(po, PlotPoint):
            fs = po._figs
            if po.iscontrolpoint:
                fs = po.nurbs._figs
                nurbs.add(po.nurbs)
        elif isinstance(po, PlotNURBS):
            fs = po._figs
            nurbs.add(po)
        else:
            continue
        if hasattr(po, 'ffd'):
            ffds.add(po.ffd)
        figs.update(fs)
    for fdd in ffds:
        fdd.refresh()
    for n in nurbs:
        n._fill_batch()
    for f in figs:
        f.a.adjust_axes()


class PlotObject(object):

    def _draw(self):
        pass


class PlotPoint(PlotObject):

    def __init__(self):
        self._figs = set()
        self._color = np.array(COLORMAP['3pts'], dtype=np.int32)
        self.visible = {'point': True}

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color[:] = color
        if hasattr(self, 'nurbs'):
            self.nurbs._fill_batch()

    def colorize(self, reset=False, fill_batch=True):
        if reset:
            if not hasattr(self, 'nurbs'):
                c = COLORMAP['3pts']
            else:
                c = (COLORMAP['vols'] if isinstance(self.nurbs, PlotVolume)
                                      else COLORMAP['cpts'])
        else:
            rc = np.random.random_integers(0, 255, 3)
            c = np.append(rc, 255)
        if fill_batch:
            self.color = c
        else:
            self._color[:] = c
        return c

    def _predraw(self, fig):
        self._figs.add(fig)

    def _postdraw(self, fig):
        self._figs.discard(fig)

    def toggle(self, attr):
        try:
            self.visible[attr] = not self.visible[attr]
        except KeyError:
            pass

    def _render(self):
        glColor4ub(*self.color)
        glBegin(GL_POINTS)
        glVertex4d(*self._xyzw)
        glEnd()

    def _draw(self, dummy):
        if self.visible.get('point'):
            self._render()


class PlotNURBS(PlotObject):

    def __init__(self):
        super(PlotNURBS, self).__init__()
        self._figs = set()
        self.visible = {'nurbs': True, 'cobj': False}

    def _set_cpoint_color(self):
        for cpt in self.cobj.cpts.flat:
            cpt._color = np.array(COLORMAP['cpts'], dtype=np.int32)

    def _fill_batch_points(self):
        cpts = self.cobj.cpts
        if isinstance(self, PlotCurve):
            fcpt = cpts[1]
        elif isinstance(self, PlotSurface):
            fcpt = cpts[1,0]
        else:
            fcpt = cpts[1,0,0]
            scpt = cpts[0,1,0]
        fcpt._color = np.array(COLORMAP['fcpt'], dtype=np.int32)
        if isinstance(self, PlotVolume):
            scpt._color = np.array(COLORMAP['scpt'], dtype=np.int32)
        for cpt in self.cobj.cpts.flat:
            self._batch.add(1, pyglet.gl.GL_POINTS, None,
                            ('v4d', cpt._xyzw),
                            ('c4B', cpt.color))

    def _fill_batch(self):
        if self._figs:
            self._batch = pyglet.graphics.Batch()
            self._fill_batch_points()
            self._fill_batch_lines()

    def _predraw(self, fig):
        self._figs.add(fig)
        if len(self._figs) == 1:
            self._fill_batch()

    def _postdraw(self, fig):
        self._figs.discard(fig)
        if not self._figs:
            del self._batch

    def toggle(self, attr):
        try:
            self.visible[attr] = not self.visible[attr]
        except KeyError:
            pass

    def _draw(self, NR):
        if self.visible.get('cobj'):
            self._batch.draw()
        if self.visible.get('nurbs'):
            self._render(NR)


class PlotCurve(PlotNURBS):

    def __init__(self):
        super(PlotCurve, self).__init__()
        self._color = np.array(COLORMAP['crvs'], dtype=np.int32)
        self._set_cpoint_color()
        self.line_width = 3.0

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color[:] = color

    def colorize(self, reset=False):
        if reset:
            c = COLORMAP['crvs']
            self._set_cpoint_color()
            self._fill_batch()
        else:
            rc = np.random.random_integers(0, 255, 3)
            c = np.append(rc, 255)
        self.color = c
        return c

    def _fill_batch_lines(self):
        n, = self.cobj.n
        Pw = self.cobj.Pw
        for i in xrange(n):
            cpt1, cpt2 = Pw[i], Pw[i+1]
            self._batch.add(2, pyglet.gl.GL_LINES, None,
                            ('v4d', np.hstack((cpt1, cpt2))),
                            ('c4B', COLORMAP['cobj'] * 2))

    def _render(self, NR):
        glColor4ub(*self.color)
        glLineWidth(self.line_width)
        gluBeginCurve(NR[0])
        gluNurbsCurve(NR[0], self.U[0].astype('float32'),
                             self.cobj.Pw.astype('float32'), GL_MAP1_VERTEX_4)
        gluEndCurve(NR[0])
        glLineWidth(1.6)


class PlotSurface(PlotNURBS):

    def __init__(self):
        super(PlotSurface, self).__init__()
        self._color = np.array(COLORMAP['srfs'], dtype=np.float32) / 255
        self._set_cpoint_color()
        self.visible.update({'mesh': False})

    @property
    def color(self):
        return np.array(self._color * 255, dtype=np.int32)

    @color.setter
    def color(self, color):
        self._color[:] = np.array(color, dtype=np.float32) / 255

    def colorize(self, reset=False):
        if reset:
            c = COLORMAP['srfs']
            self._set_cpoint_color()
            self._fill_batch()
        else:
            rc = np.random.random_integers(0, 255, 3)
            c = np.append(rc, 255)
        self.color = c
        return c

    def _fill_batch_lines(self):
        n, m = self.cobj.n
        Pw = self.cobj.Pw
        for i in xrange(n + 1):
            for j in xrange(m):
                cpt1, cpt2 = Pw[i,j], Pw[i,j+1]
                self._batch.add(2, pyglet.gl.GL_LINES, None,
                                ('v4d', np.hstack((cpt1, cpt2))),
                                ('c4B', COLORMAP['cobj'] * 2))
        for j in xrange(m + 1):
            for i in xrange(n):
                cpt1, cpt2 = Pw[i,j], Pw[i+1,j]
                self._batch.add(2, pyglet.gl.GL_LINES, None,
                                ('v4d', np.hstack((cpt1, cpt2))),
                                ('c4B', COLORMAP['cobj'] * 2))

    def mesh(self, num=2, uvs=None, visible=True):
        if uvs is None:
            U, V = self.U
            us, vs = (np.linspace(U[0], U[-1], num),
                      np.linspace(V[0], V[-1], num))
        else:
            us, vs = uvs
        ics = ([self.extract(u, 0) for u in us] +
               [self.extract(v, 1) for v in vs])
        for ic in ics:
            ic.color = COLORMAP['isoc']
            ic.line_width = 1.0
        self._isocurves = ics
        self.visible['mesh'] = True if visible else False

    def _render(self, NR):
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self._color)
        gluBeginSurface(NR[1])
        gluNurbsSurface(NR[1], self.U[0].astype('float32'),
                               self.U[1].astype('float32'),
                               self.cobj.Pw.astype('float32'), GL_MAP2_VERTEX_4)
        if hasattr(self, '_trimcurves'):
            for tc in self._trimcurves:
                gluBeginTrim(NR[1])
                Pw = np.delete(tc.cobj.Pw, 2, 1)
                gluNurbsCurve(NR[1], tc.U[0], Pw, GLU_MAP1_TRIM_3)
                gluEndTrim(NR[1])
        gluEndSurface(NR[1])
        glDisable(GL_LIGHTING)

    def _draw(self, NR):
        super(PlotSurface, self)._draw(NR)
        if self.visible['mesh'] and hasattr(self, '_isocurves'):
            for ic in self._isocurves:
                ic._render(NR)


class PlotVolume(PlotNURBS):

    def __init__(self):
        super(PlotVolume, self).__init__()
        self._color = np.array(COLORMAP['vols'], dtype=np.int32)
        self._set_cpoint_color()
        self.visible.update({'mesh': False})

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color[:] = color
        self._set_cpoint_color()
        self._fill_batch()

    def colorize(self, reset=False):
        if reset:
            c = COLORMAP['vols']
        else:
            rc = np.random.random_integers(0, 255, 3)
            c = np.append(rc, 255)
        self.color = c
        return c

    def _set_cpoint_color(self):
        for cpt in self.cobj.cpts.flat:
            cpt._color = np.array(self.color, dtype=np.int32)

    def _fill_batch_lines(self):
        n, m, l = self.cobj.n
        Pw = self.cobj.Pw
        for i in xrange(n + 1):
            for j in xrange(m + 1):
                for k in xrange(l):
                    cpt1, cpt2 = Pw[i,j,k], Pw[i,j,k+1]
                    self._batch.add(2, pyglet.gl.GL_LINES, None,
                                    ('v4d', np.hstack((cpt1, cpt2))),
                                    ('c4B', COLORMAP['cobj'] * 2))
        for j in xrange(m + 1):
            for k in xrange(l + 1):
                for i in xrange(n):
                    cpt1, cpt2 = Pw[i,j,k], Pw[i+1,j,k]
                    self._batch.add(2, pyglet.gl.GL_LINES, None,
                                    ('v4d', np.hstack((cpt1, cpt2))),
                                    ('c4B', COLORMAP['cobj'] * 2))
        for k in xrange(l + 1):
            for i in xrange(n + 1):
                for j in xrange(m):
                    cpt1, cpt2 = Pw[i,j,k], Pw[i,j+1,k]
                    self._batch.add(2, pyglet.gl.GL_LINES, None,
                                    ('v4d', np.hstack((cpt1, cpt2))),
                                    ('c4B', COLORMAP['cobj'] * 2))

    def mesh(self, num=2, uvws=None, visible=True):
        U, V, W = self.U
        if uvws is None:
            us = np.linspace(U[0], U[-1], num)
            vs = np.linspace(V[0], V[-1], num)
            ws = np.linspace(W[0], W[-1], num)
        else:
            us, vs, ws = uvws
        uisosurfs = [self.extract(u, 0) for u in us]
        visosurfs = [self.extract(v, 1) for v in vs]
        wisosurfs = [self.extract(w, 2) for w in ws]
        ics = []
        for isos in uisosurfs:
            for v in vs:
                ics.append(isos.extract(v, 0))
            for w in ws:
                ics.append(isos.extract(w, 1))
        for isos in visosurfs:
            for u in us:
                ics.append(isos.extract(u, 0))
            for w in ws:
                ics.append(isos.extract(w, 1))
        for isos in wisosurfs:
            for u in us:
                ics.append(isos.extract(u, 0))
            for v in vs:
                ics.append(isos.extract(v, 1))
        for ic in ics:
            ic.color = COLORMAP['isoc']
            ic.line_width = 1.0
        self._isocurves = ics
        self.visible['mesh'] = True if visible else False

    def _draw(self, NR):
        if self.visible.get('nurbs'):
            self._batch.draw()
        if self.visible.get('mesh') and hasattr(self, '_isocurves'):
            for ic in self._isocurves:
                ic._render(NR)
