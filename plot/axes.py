import numpy as np
import pyglet


class Axes(object):

    def __init__(self, figure, stride):
        self.f = figure

        self.visible = True
        self._stride = stride
        self.origin = [0,0,0]

        self.reset_bounding_box()
        self.render_object = PlotAxes(self)

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride):
        self._stride = stride
        self.adjust_axes()

    @property
    def tick_length(self):
        return self._stride / 6.0

    def reset_bounding_box(self):
        self.bounding_box = [(0,0), (0,0), (0,0)]
        self.axis_ticks = [(), (), ()]

    def adjust_bounds(self):
        self.reset_bounding_box()
        b = self.bounding_box
        for k in self.f.pos:
            for po in self.f.pos[k]:
                n = po.bounds
                for i in xrange(3):
                    b[i] = (min(b[i][0], n[i][0]),
                            max(b[i][1], n[i][1]))

    def recalculate_axis_ticks(self, axis):
        b = self.bounding_box
        self.axis_ticks[axis] = self.strided_range(b[axis][0],
                                                   b[axis][1],
                                                   self.stride)

    def strided_range(self, rmin, rmax, stride):
        maxd = self.f.c.max_dist
        rmins = np.mod(rmin, stride)
        rmaxs = stride - np.mod(rmax, stride)
        if np.allclose(rmaxs, stride):
            rmaxs = 0.0
        rmin -= rmins
        rmax += rmaxs
        rmin = max(rmin, - maxd)
        rmax = min(rmax,   maxd)
        rsteps = int((rmax - rmin) / stride)
        return ([rmin] +
                [rmin + e * stride for e in xrange(1, rsteps + 1)] +
                [rmax])

    def adjust_axes(self):
        self.adjust_bounds()
        for i in xrange(3):
            self.recalculate_axis_ticks(i)
        self.render_object.fill_batch()

    def toggle_axes(self):
        self.visible = not self.visible
        if self.visible:
            self.adjust_axes()

    def _draw(self):
        if self.visible:
            self.render_object.batch.draw()


class PlotAxes(object):

    def __init__(self, parent_axes):
        self.p = parent_axes
        self.color = [(229,  76, 127, 255),
                      (127, 255, 127, 255),
                      ( 76,  76, 229, 255)]

    def fill_batch(self):
        self.batch = pyglet.graphics.Batch()
        self.fill_batch_axis(0, self.color[0])
        self.fill_batch_axis(1, self.color[1])
        self.fill_batch_axis(2, self.color[2])

    def fill_batch_axis(self, axis, color):
        ticks = self.p.axis_ticks[axis]
        if not ticks:
            return
        self.fill_batch_axis_line(axis, ticks[0], ticks[-1], color)
        radius = self.p.tick_length / 2.0
        for tick in ticks:
            if not abs(tick) < 0.001:
                self.fill_batch_axis_tick_lines(axis, radius, tick, color)

    def fill_batch_axis_line(self, axis, a_min, a_max, color):
        axis_line = [[0,0,0], [0,0,0]]
        axis_line[0][axis], axis_line[1][axis] = a_min, a_max
        self.fill_batch_line(axis_line, color)

    def fill_batch_axis_tick_lines(self, axis, radius, tick, color):
        tick_axis = {0:2, 1:2, 2:0}[axis]
        tick_line = [[0,0,0], [0,0,0]]
        tick_line[0][axis] = tick_line[1][axis] = tick
        tick_line[0][tick_axis], tick_line[1][tick_axis] = -radius, radius
        self.fill_batch_line(tick_line, color)

    def fill_batch_line(self, v, color):
        o = self.p.origin
        self.batch.add(2, pyglet.gl.GL_LINES, None,
                        ('v3f', (v[0][0] + o[0],
                                 v[0][1] + o[1],
                                 v[0][2] + o[2],
                                 v[1][0] + o[0],
                                 v[1][1] + o[1],
                                 v[1][2] + o[2])),
                        ('c4B', color * 2))
