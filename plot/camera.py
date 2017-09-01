from OpenGL.GL import *

import numpy as np

import util


class Camera(object):

    min_dist = 0.005
    max_dist = 500.0
    default_dist = 5.0
    rot_presets = {'xy': (  0, 0, 180),
                   'xz': (-90, 0, 180),
                   'yz': (-90, 0,  90),
                   'perspective': (-45, 0, 135)}

    def __init__(self, figure):
        self.f = figure
        self.reset()

    def setup_projection(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        ar = float(self.f.width) / float(self.f.height)
        glOrtho(- self.dist * ar, self.dist * ar,
                - self.dist, self.dist,
                self.min_dist - self.max_dist, self.max_dist)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def unset_projection(self):
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

    def apply_transformations(self):
        glTranslatef(self.x, self.y, - self.dist)
        glMultMatrixf(self.rot)

    def euler_rotate(self, angle, x, y, z):
        self.setup_projection()
        self.apply_transformations()
        glLoadMatrixf(self.rot)
        glRotatef(angle, x, y, z)
        self.rot = util.get_model_matrix()
        self.unset_projection()

    def setup_preset(self, preset_name='', r=None):
        self.rot = np.eye(4, dtype=np.float32)
        if r is None:
            r = self.rot_presets[preset_name]
        self.euler_rotate(r[0], 1, 0, 0)
        self.euler_rotate(r[1], 0, 1, 0)
        self.euler_rotate(r[2], 0, 0, 1)

    def reset(self):
        self.x, self.y = 0.0, 0.0
        self.dist = self.default_dist
        self.setup_preset('perspective')

    def mult_rot_matrix(self, rot):
        glPushMatrix()
        glLoadMatrixf(rot)
        glMultMatrixf(self.rot)
        self.rot = util.get_model_matrix()
        glPopMatrix()

    def zoom_relative(self, click):
        mind, maxd = self.min_dist, self.max_dist
        newd = self.dist * (1.0  - click / 100.0)
        if newd < maxd and newd > mind:
            self.dist = newd

    def spherical_rotate(self, p1, p2, sensi=1.0):
        self.setup_projection()
        glTranslatef(self.x, self.y, - self.dist)
        mat = util.get_spherical_rotation(p1, p2,
                                          self.f.width, self.f.height,
                                          sensi)
        if mat is not None:
            self.mult_rot_matrix(mat)
        self.unset_projection()

    def mouse_translate(self, x, y, dx, dy):
        self.setup_projection()
        glTranslatef(0, 0, - self.dist)
        z = util.model_to_screen(0, 0, 0)[2]
        d = (util.screen_to_model(x, y, z) -
             util.screen_to_model(x - dx, y - dy, z))
        self.unset_projection()
        self.x += d[0]; self.y += d[1]
