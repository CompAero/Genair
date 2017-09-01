import os

from OpenGL.GL import *
from OpenGL.GLU import *

import Image
import numpy as np
import pyglet

import axes
import camera
import controller
import mode
import plot
import pobject


class Figure(pyglet.window.Window):

    ID = 0

    def __init__(self, *pos, **kwargs):

        ''' Draw any number of PlotObjects inside a pyglet Window.

        Only Point, Curve, Surface and Volume objects can actually be
        rendered onscreen; other subclasses of PlotObject, e.g. Airfoil
        and Wing, must define a `_draw` method that must return a
        sequence-like of either:

          1) any number of any of the four plottable PlotObjects, or;
          2) any number of any other non-plottable PlotObject, or;
          3) any combination thereof.

        PlotObjects can also be added (removed) to (from) a Figure at
        run-time; see Figure.inject (Figure.deject).

        For efficiency reasons GLU renders NURBS by tesselating them
        with polygons.  For Surfaces the size of these polygons can be
        controlled with both the `step` input parameter and/or at
        run-time through a Figure property of the same name (see example
        below).

        Parameters
        ----------
        pos = a list of PlotObjects
        stride = the distance used to separate ticks on the axes
                 (default: 1.0)
        step = the number of sample points per unit length taken along
               the parametric coordinates to tesselate the Surfaces
               (default: 5.0)
        color = the backgound color (default: (0,0,0,255))

        The stride, step and color parameters can be changed at run-time
        through the Figure.a.stride, Figure.step and Figure.color
        properties, respectively.

        Returns
        -------
        fig = a Figure

        Intended usage
        --------------
        >>> fig = draw(pt, crv, wi, stride=0.5) # Point, Curve, Wing
        >>> fig.inject(vol, fus) # Volume, Fuselage
        >>> fig.deject(wi)
        >>> fig.step
        5.0
        >>> fig.step = 20.0
        >>> fig.color = 255, 175, 0, 255
        >>> fig.photograph()

        '''

        Figure.ID += 1
        self.name = 'fig' + str(Figure.ID)

        super(Figure, self).__init__(caption=self.name,
                                     width=800,
                                     height=600,
                                     resizable=True,
                                     vsync=False)

        self.NR = setup_gl()
        self.step = kwargs.get('step', 5.0)

        self.fps_display = pyglet.clock.ClockDisplay()
        self.mode_label = pyglet.text.Label(text='TranslatePointMode',
                                            font_size=36,
                                            bold=True,
                                            color=(127,127,127,127),
                                            y=10)

        self.color = kwargs.get('color', np.array((0,0,0,255), dtype=np.int32))

        self.a = axes.Axes(self, kwargs.get('stride', 1.0))
        self.c = camera.Camera(self)
        self.m = mode.ModeManager(self)
        self.p = Photographer(self)

        self.C = controller.Controller(self)
        self.push_handlers(self.C)

        self.pos = dict(points=set(), curves=set(), surfaces=set(),
                        volumes=set())
        self.inject(*pos)

        self.IP = get_ipython()
        self.IP.user_ns[self.name] = self

        # XXX: on windows, this strategy works with 1 window at a time
        if len(pyglet.app.windows) == 1:
            self.IP.enable_gui('pyglet2')

    @property
    def color(self):
        return np.asarray(self._color * 255, dtype=np.int32)

    @color.setter
    def color(self, color):
        self._color = np.asarray(color, dtype=np.float32) / 255

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        gluNurbsProperty(self.NR[1], GLU_U_STEP, step)
        gluNurbsProperty(self.NR[1], GLU_V_STEP, step)
        self._step = step

    def inject(self, *pos, **kwargs):

        ''' Recursively inject any number of PlotObjects.

        Parameters
        ----------
        pos = a list of PlotObjects
        adjust_axes = whether or not to adjust the axes (default: True)

        '''

        adjust_axes = kwargs.get('adjust_axes', True)
        for po in pos:
            if not isinstance(po, pobject.PlotObject):
                print('plot.figure.Figure.inject :: '
                      'not a PlotObject ({})'.format(type(po)))
                continue
            k = get_key(po)
            try:
                if not k:
                    self.inject(*po._draw(), adjust_axes=False)
                else:
                    self.pos[k].add(po)
                    po._predraw(self)
            except Exception:
                print('plot.figure.Figure.inject :: '
                      'could not inject PlotObject ({})'.format(po))
        if adjust_axes:
            self.a.adjust_axes()

    def deject(self, *pos, **kwargs):

        ''' Recursively deject any number of PlotObjects.

        Parameters
        ----------
        pos = a list of PlotObjects
        adjust_axes = whether or not to adjust the axes (default: True)

        '''

        adjust_axes = kwargs.get('adjust_axes', True)
        for po in pos:
            k = get_key(po)
            try:
                if not k:
                    self.deject(*po._draw(), adjust_axes=False)
                else:
                    self.pos[k].discard(po)
                    po._postdraw(self)
            except Exception:
                print('plot.figure.Figure.deject :: '
                      'could not deject PlotObject ({})'.format(po))
        if adjust_axes:
            self.a.adjust_axes()

    def toggle_viz(self, names, attr):
        for name in names:
            for po in self.pos[name]:
                po.toggle(attr)

    def draw_pobjects(self):
        self.c.setup_projection()
        self.c.apply_transformations()
        glEnable(GL_DEPTH_TEST)
        self.a._draw()
        for k in ('points', 'curves', 'surfaces', 'volumes'):
            for po in self.pos[k]:
                po._draw(self.NR)
        glDisable(GL_DEPTH_TEST)
        self.c.unset_projection()

    def on_draw(self):
        glClearColor(*self._color)
        self.clear()
        self.draw_pobjects()
        if self.mode_label.text != 'TranslatePointMode':
            self.mode_label.draw()
        self.fps_display.draw()

    def on_close(self):
        for po in [po
                   for pos in self.pos.values()
                   for po in pos]:
            po._postdraw(self)
        if self.name in self.IP.user_ns:
            del self.IP.user_ns[self.name]
        self.close()

        # XXX: on windows, this strategy works with 1 window at a time
        if not pyglet.app.windows:
            self.IP.enable_gui()
            assert not self.IP.active_eventloop

    def photograph(self, fn=None):

        ''' Store the current framebuffer in PNG format.

        Parameters
        ----------
        fn = the name of the outputted PNG file

        '''

        self.p.photograph(fn)


class Photographer(object):

    def __init__(self, figure):
        self.f = figure

    def photograph(self, out=None):
        if not out:
            out = self.create_unique_filename()
        self.read_write_framebuffer(out)

    def create_unique_filename(self):
        cwd = os.getcwd()
        files = os.listdir(cwd)
        i = 0
        while True:
            f = 'fig{}.png'.format(i)
            if not f in files:
                path = cwd + '/' + f
                break
            i += 1
        return path

    def read_write_framebuffer(self, out):
        w, h = self.f.width, self.f.height
        image = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.fromstring('RGBA', (w, h), image)
        image.transpose(Image.FLIP_TOP_BOTTOM).save(out)


# UTILITIES


def setup_gl():

    LIG = np.array((0.6,0.6,0.6,1.0), dtype=np.float32)
    SPF = np.array((1.0,1.0,1.0,1.0), dtype=np.float32)
    SHF = np.array(128.0, dtype=np.float32)
    SPB = np.array((0.2,0.2,0.2,1.0), dtype=np.float32)
    SHB = np.array(40.0, np.float32)
    DIF = np.array((0.0,0.4,0.6,1.0), dtype=np.float32)

    glEnable(GL_AUTO_NORMAL)

    glEnable(GL_LIGHT0)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, LIG)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)

    glMaterialfv(GL_FRONT, GL_SPECULAR, SPF)
    glMaterialfv(GL_FRONT, GL_SHININESS, SHF)
    glMaterialfv(GL_BACK, GL_SPECULAR, SPB)
    glMaterialfv(GL_BACK, GL_SHININESS, SHB)
    glMaterialfv(GL_BACK, GL_DIFFUSE, DIF)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(1.6)

    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    glPointSize(8.0)

    NC = gluNewNurbsRenderer()
    gluNurbsProperty(NC, GLU_SAMPLING_TOLERANCE, 10.0)

    NS = gluNewNurbsRenderer()
    gluNurbsProperty(NS, GLU_SAMPLING_METHOD, GLU_DOMAIN_DISTANCE)
    gluNurbsProperty(NS, GLU_DISPLAY_MODE, GLU_FILL)
   #gluNurbsProperty(NS, GLU_DISPLAY_MODE, GLU_OUTLINE_POLYGON)

    return NC, NS


def get_key(po):
    if isinstance(po, pobject.PlotPoint):
        return 'points'
    elif isinstance(po, pobject.PlotCurve):
        return 'curves'
    elif isinstance(po, pobject.PlotSurface):
        return 'surfaces'
    elif isinstance(po, pobject.PlotVolume):
        return 'volumes'
