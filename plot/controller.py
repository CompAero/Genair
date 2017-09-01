''' Controller for Figure objects.

Screen Rotation:

    X (Roll)            Numpad 4, 6
    Y (Pitch)           Numpad 8, 2
    Z (Yaw)             Numpad 7, 9

Model Rotation:

    Z (Yaw)             Numpad 1, 3

Zoom:                   Numpad +, -, Mouse Scroll, Mouse Drag

Camera:

    XY                  F1
    XZ                  F2
    YZ                  F3
    Perspective         F4
    Reset Camera        F5, Numpad 5
    Fullscreen          F6

Pick:                   Left Mouse

    Point               CTRL
    Curve               SHIFT
    Surface             SHIFT + CTRL
    Volume              SHIFT + CTRL + ALT

Keymap Toggles:

    Axes                F7
    Control Points      F8
    Points              F9
    Curves              F10
    Surfaces            F11
    Volumes             F12

Mode:                   CTRL + ALT

    End Selection 1     Q
    End Selection 2     W
    Undo                U

Modemap Toggles:

    Translate Point     Default
    Translate NURBS     N
    Compose Curve       C
    Compose Surface     V
    Split Curve         S
    Split Surface       D
    Split Surface 2     F
    Interpolate Curve   I
    Make Cubic Bezier   E
    Make Bilinear Coons Surface R

Close Figure:           ESCAPE

'''

import sys

from pyglet.window import key
from pyglet.window import mouse
import pyglet

import util

KEYMAP = {key.NUM_1: 'rotate_z_neg',
          key.NUM_2: 'down',
          key.NUM_3: 'rotate_z_pos',
          key.NUM_4: 'left',
          key.NUM_5: 'reset_camera',
          key.NUM_6: 'right',
          key.NUM_7: 'spin_left',
          key.NUM_8: 'up',
          key.NUM_9: 'spin_right',
          key.NUM_ADD: 'zoom_in',
          key.NUM_SUBTRACT: 'zoom_out',

          key.F1: 'rot_preset_xy',
          key.F2: 'rot_preset_xz',
          key.F3: 'rot_preset_yz',
          key.F4: 'rot_preset_perspective',
          key.F5: 'reset_camera',
          key.F6: 'toggle_fullscreen',

          key.F7: 'toggle_axes',
          key.F8: 'toggle_cobjects',
          key.F9: 'toggle_points',
          key.F10: 'toggle_curves',
          key.F11: 'toggle_surfaces',
          key.F12: 'toggle_volumes'}

MODEMAP = {key.N: 'toggle_translate_nurbs',
           key.C: 'toggle_compose_curve',
           key.V: 'toggle_compose_surface',
           key.S: 'toggle_split_curve',
           key.D: 'toggle_split_surface',
           key.F: 'toggle_split_surface2',
           key.I: 'toggle_interpolate_curve',
           key.E: 'toggle_make_cubic_bezier',
           key.R: 'toggle_make_bilinear_coons_surface',

           key.Q: 'end_selection1',
           key.W: 'end_selection2',
           key.U: 'undo'}

if sys.platform == 'darwin':
    key.MOD_ALT = key.MOD_OPTION


class Controller(object):

    mouse_sensi = 5.0
    key_sensi = 4.0

    def __init__(self, figure):
        self.f = figure
        self.a = figure.a
        self.c = figure.c
        self.m = figure.m
        self.p = figure.p

    def update(self, a):
        z = rz = dx = dy = dz = 0.0
        if a == 'rot_preset_xy':
            self.c.setup_preset('xy')
        elif a == 'rot_preset_xz':
            self.c.setup_preset('xz')
        elif a == 'rot_preset_yz':
            self.c.setup_preset('yz')
        elif a == 'rot_preset_perspective':
            self.c.setup_preset('perspective')
        elif a == 'reset_camera':
            self.c.reset()
        elif a == 'toggle_fullscreen':
            fs = self.f.fullscreen
            self.f.set_fullscreen(not fs)
            self.f.activate()
        elif a == 'toggle_axes':
            self.a.toggle_axes()
        elif a == 'toggle_points':
            self.f.toggle_viz(['points'], 'point')
        elif a == 'toggle_cobjects':
            self.f.toggle_viz(['curves', 'surfaces'], 'cobj')
        elif a == 'toggle_curves':
            self.f.toggle_viz(['curves'], 'nurbs')
        elif a == 'toggle_surfaces':
            self.f.toggle_viz(['surfaces'], 'nurbs')
        elif a == 'toggle_volumes':
            self.f.toggle_viz(['volumes'], 'nurbs')
        elif a == 'end_selection1':
            self.m.mode.end_selection1()
        elif a == 'end_selection2':
            self.m.mode.end_selection2()
        elif a == 'undo':
            self.m.undo()
        elif a == 'toggle_translate_nurbs':
            self.m.toggle_mode('TranslateNURBSMode')
        elif a == 'toggle_compose_curve':
            self.m.toggle_mode('ComposeCurveMode')
        elif a == 'toggle_compose_surface':
            self.m.toggle_mode('ComposeSurfaceMode')
        elif a == 'toggle_split_curve':
            self.m.toggle_mode('SplitCurveMode')
        elif a == 'toggle_split_surface':
            self.m.toggle_mode('SplitSurfaceMode')
        elif a == 'toggle_split_surface2':
            self.m.toggle_mode('SplitSurface2Mode')
        elif a == 'toggle_interpolate_curve':
            self.m.toggle_mode('InterpolateCurveMode')
        elif a == 'toggle_make_cubic_bezier':
            self.m.toggle_mode('MakeCubicBezierMode')
        elif a == 'toggle_make_bilinear_coons_surface':
            self.m.toggle_mode('MakeBilinearCoonsSurfaceMode')
        elif a == 'zoom_out':
            z -= 1
        elif a == 'zoom_in':
            z += 1
        elif a == 'left':
            dx -= 1
        elif a == 'right':
            dx += 1
        elif a == 'up':
            dy -= 1
        elif a == 'down':
            dy += 1
        elif a == 'spin_left':
            dz += 1
        elif a == 'spin_right':
            dz -= 1
        elif a == 'rotate_z_neg':
            rz -= 1
        elif a == 'rotate_z_pos':
            rz += 1
        if z != 0:
            self.c.zoom_relative(z)
        elif dx != 0:
            xyz = util.get_direction_vectors()[1]
            self.c.euler_rotate(dx * self.key_sensi, *xyz)
        elif dy != 0:
            xyz = util.get_direction_vectors()[0]
            self.c.euler_rotate(dy * self.key_sensi, *xyz)
        elif dz != 0:
            xyz = util.get_direction_vectors()[2]
            self.c.euler_rotate(dz * self.key_sensi, *xyz)
        elif rz != 0:
            self.c.euler_rotate(rz * self.key_sensi, *(0, 0, 1))

    def on_key_press(self, symbol, modifiers):
        m = None
        if symbol in KEYMAP:
            m = KEYMAP
        elif (symbol in MODEMAP and modifiers == (key.MOD_CTRL |
                                                  key.MOD_ALT)):
            m = MODEMAP
        if m:
            self.update(m[symbol])

    def on_mouse_press(self, x, y, button, modifiers):
        if button & mouse.LEFT:
            if modifiers == key.MOD_CTRL:
                self.m.pick_point(x, y)
            elif modifiers == key.MOD_SHIFT:
                self.m.pick_curve(x, y)
            elif modifiers == (key.MOD_CTRL | key.MOD_SHIFT):
                self.m.pick_surface(x, y)
            elif modifiers == (key.MOD_ALT | key.MOD_CTRL |
                               key.MOD_SHIFT):
                self.m.pick_volume(x, y)
        elif button & mouse.RIGHT:
            if modifiers == key.MOD_SHIFT:
                self.m.pick_curve(x, y, True)
            elif modifiers == (key.MOD_CTRL | key.MOD_SHIFT):
                self.m.pick_surface(x, y, True)

    def on_mouse_release(self, x, y, button, modifiers):
        if self.m.p.success:
            self.m.unpick_object()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT and modifiers:
            self.m.translate_object(x, y, dx, dy)
        elif buttons & mouse.LEFT:
            self.c.spherical_rotate((x - dx, y - dy), (x, y),
                                    self.mouse_sensi)
        elif buttons & mouse.MIDDLE:
            self.c.zoom_relative(dy)
        elif buttons & mouse.RIGHT:
            self.c.mouse_translate(x, y, dx, dy)

    def on_mouse_scroll(self, x, y, dx, dy):
        self.c.zoom_relative(dy)

    def on_resize(self, width, height):
        label = self.f.mode_label
        label.x = width - label.content_width - 10
