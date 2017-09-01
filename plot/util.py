from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np


def get_model_matrix(GetMethod=glGetFloatv):
    return GetMethod(GL_MODELVIEW_MATRIX)

def get_projection_matrix(GetMethod=glGetFloatv):
    return GetMethod(GL_PROJECTION_MATRIX)

def get_viewport():
    return glGetIntegerv(GL_VIEWPORT)

def get_direction_vectors():
    m = get_model_matrix()
    return m[:3,0], m[:3,1], m[:3,2]

def screen_to_model(x, y, z):
    m = get_model_matrix(glGetDoublev)
    p = get_projection_matrix(glGetDoublev)
    w = get_viewport()
    mx, my, mz = gluUnProject(x, y, z, m, p, w)
    return np.array((mx, my, mz))

def model_to_screen(x, y, z):
    m = get_model_matrix(glGetDoublev)
    p = get_projection_matrix(glGetDoublev)
    w = get_viewport()
    mx, my, mz = gluProject(x, y, z, m, p, w)
    return np.array((mx, my, mz))

def normalize(V):
    norm = np.linalg.norm(V)
    if norm == 0.0:
        return V
    return np.asfarray(V) / norm

def get_sphere_mapping(x, y, width, height):
    xo, yo = model_to_screen(0, 0, 0)[0:2]
    sr = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    sx = ((x - xo) / sr)
    sy = ((y - yo) / sr)
    sz = 1.0 - sx ** 2 - sy ** 2
    if sz > 0.0:
        sz = np.sqrt(sz)
        return (sx, sy, sz)
    else:
        sz = 0
        return normalize((sx, sy, sz))

def get_spherical_rotation(p1, p2, width, height, sensi):
    v1 = get_sphere_mapping(p1[0], p1[1], width, height)
    v2 = get_sphere_mapping(p2[0], p2[1], width, height)
    d = min(max(np.dot(v1, v2), -1), 1)
    raxis = normalize(np.cross(v1, v2))
    rtheta = np.rad2deg(sensi) * np.math.acos(d)
    glPushMatrix()
    glLoadIdentity()
    glRotatef(rtheta, *raxis)
    mat = glGetFloatv(GL_MODELVIEW_MATRIX)
    glPopMatrix()
    return mat
