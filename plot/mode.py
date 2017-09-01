import traceback

from OpenGL.GL import *

import numpy as np

import pobject
import util

import nurbs


class ModeManager(object):

    def __init__(self, figure):
        self.f = figure
        self.p = Picker(self)
        self.undoers = []
        self.mode = TranslatePointMode(self)

    def toggle_mode(self, mode):
        self.mode.cleanup()
        if self.mode.__class__.__name__ == mode:
            mode = 'TranslatePointMode'
        self.mode = eval(mode)(self)
        f = self.f
        f.mode_label.text = mode
        f.dispatch_event('on_resize', f.width, f.height)

    def pick_point(self, x, y):
        self.p.pick_object('points', x, y)

    def pick_curve(self, x, y, crn=False):
        self.p.pick_object('curves', x, y, crn)

    def pick_surface(self, x, y, crn=False):
        self.p.pick_object('surfaces', x, y, crn)

    def pick_volume(self, x, y):
        self.p.pick_object('volumes', x, y)

    def unpick_object(self):
        self.p.reset_picked_color()
        self.f.deject(self.p._pp, adjust_axes=False)
        if isinstance(self.mode, (TranslatePointMode, TranslateNURBSMode)):
            self.mode.unpick_object()
        self.p.success = False

    def translate_object(self, x, y, dx, dy):
        if self.p.success and isinstance(self.mode, (TranslatePointMode,
                                                     TranslateNURBSMode)):
            self.mode.translate(x, y, dx, dy)

    def undo(self):
        try:
            u = self.undoers.pop()
            u.undo(u.args)
        except IndexError:
            return


class Picker(object):

    def __init__(self, manager):
        self.m = manager
        self.success = False
        self.last_color = None
        self._pp = None
        self.picked_pos = []

    def spawn_picked_point(self, pxyz, po):
        self._pp = nurbs.point.Point(*pxyz)
        self._pp.color = pobject.COLORMAP['pick']
        if not (isinstance(po, pobject.PlotPoint) and
                not po.iscontrolpoint): # anything but a 3D Point
            self.m.f.inject(self._pp, adjust_axes=False)

    def set_picked_color(self, po):
        if not (isinstance(po, pobject.PlotPoint) and
                po.iscontrolpoint): # anything but a control Point
            self.last_color = po.color.copy()
            po.color = pobject.COLORMAP['newp']

    def reset_picked_color(self):
        last_po = self.picked_pos[-1]
        if not (isinstance(last_po, pobject.PlotPoint) and
                last_po.iscontrolpoint): # anything but a control Point
            last_po.color = self.last_color

    def update_mode(self, e):
        k = ('points' if isinstance(e.args[0], pobject.PlotPoint)
                      else 'nurbs')
        self.m.mode.picked_pos[k].append(e.args)
        self.m.mode.do()

    def pick_point_screen(self, p, x, y):
        sx, sy, z = util.model_to_screen(*p.xyz)
        if x - 5 < sx < x + 5 and y - 5 < sy < y + 5: # knob
            raise PickedPlotObject(p, p.xyz, z)

    def pick_point_candidate(self, o, x, y, mxyz):
        P = nurbs.nurbs.obj_mat_to_3D(o.cobj.Pw)
        ba = (mxyz - 1 <= P) & (P <= mxyz + 1) # knob
        ind = ba.all(-1)
        for cp in o.cobj.cpts[ind]:
            self.pick_point_screen(cp, x, y)

    def pick_point(self, x, y, mxyz):
        pos = self.m.f.pos
        for po in pos['points']:
            if po.visible.get('point'):
                self.pick_point_screen(po, x, y)
        for po in pos['curves'] | pos['surfaces']:
            if po.visible.get('cobj'):
                self.pick_point_candidate(po, x, y, mxyz)
        for po in pos['volumes']:
            if po.visible.get('nurbs'):
                self.pick_point_candidate(po, x, y, mxyz)

    def pick_nurbs_end(self, n, pxyz, crns):
        mind = np.inf
        for i in crns:
            cxyz = n.cobj.cpts[i].xyz
            d = nurbs.util.norm(pxyz - cxyz)
            if d < mind:
                mind, xyz, j = d, cxyz, i
        return xyz, j

    def pick_nurbs(self, z, mxyz, pon, crn):
        n, mind = None, np.inf
        pos = self.m.f.pos[pon]
        for po in pos:
            if po.visible.get('nurbs'):
                bs = po.bounds
                bmin, bmax = [np.array(b) for b in zip(*bs)]
                bmin -= 0.1; bmax += 0.1 # knob
                if (bmin <= mxyz).all() and (mxyz <= bmax).all():
                    try:
                        u = po.project(mxyz)
                    except nurbs.nurbs.NewtonLikelyDiverged:
                        continue
                    xyz = po.eval_point(*u)
                    d = nurbs.util.norm(mxyz - xyz)
                    if d < mind:
                        n, pxyz, pu, mind = po, xyz, u, d
                        if crn:
                            if pon == 'curves':
                                crns = [0, -1]
                                pxyz, j = self.pick_nurbs_end(n, pxyz, crns)
                                U, = n.U
                                if   j ==  0: pu = U[ 0],
                                elif j == -1: pu = U[-1],
                            else:
                                crns = [(0, 0), (-1, 0), (-1, -1), (0, -1)]
                                pxyz, j = self.pick_nurbs_end(n, pxyz, crns)
                                U, V = n.U
                                if   j == ( 0,  0): pu = U[ 0], V[ 0]
                                elif j == (-1,  0): pu = U[-1], V[ 0]
                                elif j == ( 0, -1): pu = U[ 0], V[-1]
                                elif j == (-1, -1): pu = U[-1], V[-1]
        if n:
            raise PickedPlotObject(n, pxyz, z, pu)

    def pick_object(self, pon, x, y, crn=False):
        self.m.f.c.setup_projection()
        self.m.f.c.apply_transformations()
        try:
            z = glReadPixels(x, y, 5, 5, GL_DEPTH_COMPONENT, GL_FLOAT) # knob
            zc = z != 1.0
            if zc.any():
                z = np.average(z[zc])
                mxyz = util.screen_to_model(x, y, z)
                if pon == 'points':
                    self.pick_point(x, y, mxyz)
                else:
                    self.pick_nurbs(z, mxyz, pon, crn)
        except PickedPlotObject as e:
            self.success = True
            po, pxyz = e.args[:2]
            self.spawn_picked_point(pxyz, po)
            self.set_picked_color(po)
            self.picked_pos.append(po)
            self.update_mode(e)
        else:
            self.__init__(self.m)
        finally:
            pp, ps = self._pp, self.picked_pos
            pxyz = pp.xyz if pp else None
            p = ps[-1] if ps else None
            self.m.f.IP.user_ns.update(dict(last_picked_xyz=pxyz,
                                            last_picked_object=p,
                                            last_picked_objects=ps))
        self.m.f.c.unset_projection()


class Undoer(object):

    def __init__(self, args):
        self.args = args


class Mode(object):

    def __init__(self, manager):
        self.m = manager
        self.picked_pos = dict(points=[], nurbs=[])

    def save(self, *args):
        u = Undoer(args)
        u.undo = self.undo
        self.m.undoers.append(u)

    def do(self):
        pass

    def undo(self):
        pass

    def end_selection1(self):
        pass

    def end_selection2(self):
        pass

    def cleanup(self):
        pass


class TranslateMode(Mode):

    def __init__(self, m):
        super(TranslateMode, self).__init__(m)
        self.t = None

    def deltas(self, x, y, z, dx, dy):
        self.m.f.c.setup_projection()
        self.m.f.c.apply_transformations()
        dxdydz = (util.screen_to_model(x, y, z) -
                  util.screen_to_model(x - dx, y - dy, z))
        dxdydz = np.round(dxdydz, decimals=6) # knob
        self.m.f.c.unset_projection()
        return dxdydz

    def translate(self, x, y, dx, dy):
        if not self.t:
            return
        dxdydz = self.deltas(x, y, self.z, dx, dy)
        nurbs.transform.translate(self.m.p._pp._xyzw, dxdydz)
        for po in self.pos:
            if isinstance(po, pobject.PlotPoint):
                nurbs.transform.translate(po._xyzw, dxdydz)
            elif isinstance(po, pobject.PlotNURBS):
                nurbs.transform.translate(po.cobj.Pw, dxdydz)

    def do(self, t):
        self.t, self.pxyz, self.z = t[:3]
        if hasattr(self.t, 'glued') and self.t.glued:
            self.pos = set(self.t.glued)
        else:
            self.pos = {self.t}
        self.to_save = []
        for po in self.pos:
            if isinstance(po, pobject.PlotPoint):
                c = po._xyzw.copy()
            elif isinstance(po, pobject.PlotNURBS):
                c = po.cobj.Pw.copy()
            self.to_save.append(c)

    def undo(self, args):
        pos, saves = args
        nurbs, ffds, figs = set(), set(), set()
        for po, save in zip(pos, saves):
            if isinstance(po, pobject.PlotPoint):
                po._xyzw[:] = save
                fs = po._figs
                if po.iscontrolpoint:
                    nurbs.add(po.nurbs)
                    fs = po.nurbs._figs
            elif isinstance(po, pobject.PlotNURBS):
                po.cobj.Pw[:] = save
                fs = po._figs
                nurbs.add(po)
            if hasattr(po, 'ffd'):
                ffds.add(po.ffd)
            figs.update(fs)
        for n in nurbs:
            n._fill_batch()
        for f in ffds:
            f.refresh()
        for f in figs:
            f.a.adjust_axes()

    def unpick_object(self):
        t = self.t
        if t and not np.allclose(self.m.p._pp.xyz, self.pxyz):
            pobject.update_figures(self.pos)
            self.save(self.pos, self.to_save)
            self.t = None


class TranslatePointMode(TranslateMode):

    def project_to(self, t, SN, pto):
        S, N = SN
        x, y, z = pto(S, N, t.xyz); w = t._xyzw[-1]
        t._xyzw[:] = self.m.p._pp._xyzw[:] = w * x, w * y, w * z, w

    def translate(self, *args):
        super(TranslatePointMode, self).translate(*args)
        t = self.t
        if hasattr(t, 'line'):
            self.project_to(t, t.line, nurbs.util.point_to_line)
        if hasattr(t, 'plane'):
            self.project_to(t, t.plane, nurbs.util.point_to_plane)
        #if hasattr(t, 'originate'):
        #    t.originate()
        #if hasattr(t, 'innerize'):
        #    t.innerize()
        #if hasattr(t, 'sphere'):
        #    O, R = t.sphere; T = t.xyz
        #    sf = R / np.linalg.norm(T - O)
        #    nurbs.transform.scale(t._xyzw, sf, Q=O)

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            return
        super(TranslatePointMode, self).do(t)


class TranslateNURBSMode(TranslateMode):

    def project_to(self, t, N, pto):
        xyz0 = self.m.p._pp.xyz
        dxdydz = pto(self.pxyz, N, xyz0) - xyz0
        for om in self.m.p._pp._xyzw, t.cobj.Pw:
            nurbs.transform.translate(om, dxdydz)

    def translate(self, *args):
        super(TranslateNURBSMode, self).translate(*args)
        t = self.t
        if hasattr(t, 'line'):
            self.project_to(t, t.line, nurbs.util.point_to_line)
        if hasattr(t, 'plane'):
            self.project_to(t, t.plane, nurbs.util.point_to_plane)

    def do(self):
        try:
            t = self.picked_pos['nurbs'].pop()
        except IndexError:
            return
        super(TranslateNURBSMode, self).do(t)


class ComposeMode(Mode):

    def compose(self, composer, ns):
        try:
            return composer(ns)
        except nurbs.nurbs.NURBSException:
            traceback.print_exc()

    def end_selection1(self, cls, composer):
        ns = self.picked_pos['nurbs']
        if len(ns) < 2:
            return
        ns = [n[0] for n in ns if isinstance(n[0], cls)]
        n = self.compose(composer, ns)
        if n:
            for ni in ns:
                self.m.f.deject(ni, adjust_axes=False)
            self.m.f.inject(n, adjust_axes=False)
            n.colorize()
            self.save(n, ns)
        self.picked_pos['nurbs'] = []

    def undo(self, args):
        n, ns = args
        self.m.f.deject(n, adjust_axes=False)
        for n in ns:
            self.m.f.inject(n, adjust_axes=False)


class ComposeCurveMode(ComposeMode):

    def end_selection1(self):
        p = super(ComposeCurveMode, self)
        p.end_selection1(pobject.PlotCurve,
                         nurbs.curve.make_composite_curve)


class ComposeSurfaceMode(ComposeMode):

    def end_selection1(self):
        p = super(ComposeSurfaceMode, self)
        p.end_selection1(pobject.PlotSurface,
                         nurbs.surface.make_composite_surface)


class SplitCurveMode(Mode):

    def __init__(self, m):
        super(SplitCurveMode, self).__init__(m)
        self.C = None

    def split(self, xyz):
        try:
            ui, = self.C.project(xyz)
            return self.C.split(ui)
        except nurbs.nurbs.NURBSException:
            traceback.print_exc()
            return

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            try:
                t = self.picked_pos['nurbs'].pop()
            except IndexError:
                return
        if not self.C:
            t = t[0]
            if isinstance(t, pobject.PlotCurve):
                self.C = t
            return
        xyz = t[1]
        Cs = self.split(xyz)
        if Cs:
            self.m.f.deject(self.C, adjust_axes=False)
            for c in Cs:
                self.m.f.inject(c, adjust_axes=False)
                c.colorize()
            self.save(self.C, Cs)
        self.__init__(self.m)

    def undo(self, args):
        c, Cs = args
        self.m.f.inject(c, adjust_axes=False)
        for c in Cs:
            self.m.f.deject(c, adjust_axes=False)


class SplitSurfaceMode(Mode):

    ''' Split 1 Surface into 4 at once. '''

    def __init__(self, m):
        super(SplitSurfaceMode, self).__init__(m)
        self.S = None

    def project(self, xyz):
        try:
            return self.S.project(xyz)
        except nurbs.nurbs.NewtonLikelyDiverged:
            traceback.print_exc()

    def split(self, uv):
        u, v = uv
        Sss = []
        try:
            Ss = self.S.split(u, 0)
            for s in Ss:
                slr = s.split(v, 1)
                Sss += slr
            return Sss
        except nurbs.nurbs.NURBSException:
            traceback.print_exc()

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            try:
                t = self.picked_pos['nurbs'].pop()
            except IndexError:
                return
        if not self.S:
            t = t[0]
            if isinstance(t, pobject.PlotSurface):
                self.S = t
            return
        if self.S is None:
            return
        xyz = t[1]
        uv = self.project(xyz)
        if uv:
            Ss = self.split(uv)
            if Ss:
                self.m.f.deject(self.S, adjust_axes=False)
                for s in Ss:
                    self.m.f.inject(s, adjust_axes=False)
                    s.colorize()
                self.save(self.S, Ss)
        self.__init__(self.m)

    def undo(self, args):
        S, Ss = args
        self.m.f.inject(S, adjust_axes=False)
        for s in Ss:
            self.m.f.deject(s, adjust_axes=False)


class SplitSurface2Mode(Mode):

    ''' Split 1 Surface into 4 at once. '''

    def __init__(self, m):
        super(SplitSurface2Mode, self).__init__(m)
        self.S = None
        self.ui, self.uis = None, []
        self.ci, self.cis = None, []

    def extract(self, ui, di):
        try:
            return self.S.extract(ui, di)
        except nurbs.nurbs.NewtonLikelyDiverged:
            traceback.print_exc()

    def split(self, us, vs):
        sr, Ss, Sss = self.S, [], []
        try:
            for u in us:
                if u:
                    sl, sr = sr.split(u, 0)
                    Ss.append(sl)
            Ss.append(sr)
            for s in Ss:
                su = s
                for v in vs:
                    if v:
                        sd, su = su.split(v, 1)
                        Sss.append(sd)
                Sss.append(su)
            return Sss
        except nurbs.nurbs.NURBSException:
            traceback.print_exc()

    def do(self):
        try:
            n = self.picked_pos['nurbs'].pop()
        except IndexError:
            return
        S, dummy, dummy, uv = n
        if not isinstance(S, pobject.PlotSurface):
            return
        if not self.S:
            self.S = S
        elif not self.S is S:
            return
        di = 0 if len(self.uis) < 2 else 1
        ui = uv[di]
        ci = self.extract(ui, di)
        if ci:
            if len(self.uis) < 4:
                self.m.f.inject(ci, adjust_axes=False)
                if self.ci:
                    self.m.f.deject(self.ci, adjust_axes=False)
        self.ui, self.ci = ui, ci

    def undo(self, args):
        S, Ss = args
        self.m.f.inject(S, adjust_axes=False)
        for s in Ss:
            self.m.f.deject(s, adjust_axes=False)

    def end_selection1(self):
        if len(self.uis) < 4:
            self.uis.append(self.ui)
            self.cis.append(self.ci)
            self.ui, self.ci = None, None
        if len(self.uis) == 4:
            us, vs = self.uis[:2], self.uis[2:]
            us, vs = sorted(us), sorted(vs)
            Ss = self.split(us, vs)
            if Ss:
                self.m.f.deject(self.S, adjust_axes=False)
                for s in Ss:
                    self.m.f.inject(s, adjust_axes=False)
                    s.colorize()
                self.save(self.S, Ss)
            self.cleanup()
            self.__init__(self.m)

    def cleanup(self):
        for ci in [self.ci] + self.cis:
            if ci:
                self.m.f.deject(ci, adjust_axes=False)


class InterpolateCurveMode(Mode):

    def __init__(self, m):
        super(InterpolateCurveMode, self).__init__(m)
        self.Q, self.C = [], None
        self.T = self.Ts = self.Te = None

    def interpolate(self):
        Q = self.Q
        n = len(Q) - 1
        if n:
            Ts, Te = self.Ts, self.Te
            try:
                U, Pw = nurbs.fit.local_curve_interp(n, Q, Ts, Te)
                cpol = nurbs.curve.ControlPolygon(Pw=Pw)
                return nurbs.curve.Curve(cpol, (3,), (U,))
            except nurbs.nurbs.NURBSException:
                traceback.print_exc()

    def interpolate_inject_save(self):
        C = self.interpolate()
        if C:
            if self.C:
                self.m.f.deject(self.C, adjust_axes=False)
            self.m.f.inject(C, adjust_axes=False)
            C.colorize()
            self.C = C
            self.save(self.Q[:-1])

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            try:
                t = self.picked_pos['nurbs'].pop()
                if isinstance(t[0], pobject.PlotCurve):
                    self.T = t[0], t[-1]
            except IndexError:
                return
        xyz = t[1]
        self.Q.append(xyz)
        self.interpolate_inject_save()

    def undo(self, args):
        self.Q, = args
        self.m.f.deject(self.C, adjust_axes=False)
        C = self.interpolate()
        if C:
            self.m.f.inject(C, adjust_axes=False)
            C.colorize()
            self.C = C

    def end_selection1(self):
        if self.T:
            c, u = self.T
            t = c.eval_derivatives(u, 1)[1]
            if len(self.Q) == 1:
                self.Ts = t
            else:
                self.Te = t
                self.interpolate_inject_save()


class MakeCubicBezierMode(Mode):

    def __init__(self, m):
        super(MakeCubicBezierMode, self).__init__(m)
        self.Q = []

    def spawn_cubic(self):
        try:
            U, Pw = nurbs.fit.local_curve_interp(1, self.Q)
        except nurbs.fit.FitException:
            traceback.print_exc()
            return
        cpol = nurbs.curve.ControlPolygon(Pw=Pw)
        return nurbs.curve.Curve(cpol, (3,), (U,))

    def attach_bnd(self, c3):
        for cpt in c3.cobj.cpts.flat:
            x, y, dummy = cpt.xyz
            if np.allclose(x, 0.0):
                cpt.line = (0,0,0), (0,1,0)
            elif np.allclose(x, 1.0):
                cpt.line = (1,0,0), (0,1,0)
            elif np.allclose(y, 0.0):
                cpt.line = (0,0,0), (1,0,0)
            elif np.allclose(y, 1.0):
                cpt.line = (0,1,0), (1,0,0)

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            try:
                t = self.picked_pos['nurbs'].pop()
            except IndexError:
                return
        xyz = t[1]
        self.Q.append(xyz)
        if len(self.Q) == 2:
            c3 = self.spawn_cubic()
            if c3:
                self.attach_bnd(c3) # Junction
                self.m.f.inject(c3, adjust_axes=False)
                c3.colorize()
                self.save(c3)
            self.__init__(self.m)

    def undo(self, args):
        c3, = args
        self.m.f.deject(c3, adjust_axes=False)


class MakeBilinearCoonsSurfaceMode(Mode):

    def __init__(self, m):
        super(MakeBilinearCoonsSurfaceMode, self).__init__(m)
        self.Ckls = []

    def reorient(self, Ckls):

        def FindCl(end):
            cpte = Ck0.cobj.cpts[end].xyz
            for i, c in enumerate(Ckls):
                cpts = c.cobj.cpts
                for cpt in cpts[0], cpts[-1]:
                    xyz = cpt.xyz
                    if np.allclose(cpt.xyz, cpte, atol=1e-5):
                        C = Ckls.pop(i)
                        if cpt is cpts[-1]:
                            C = C.reverse()
                        return C

        Ck0 = Ckls.pop(0)
        Cl0 = FindCl(0)
        Cl1 = FindCl(-1)
        Ck1 = Ckls.pop()

        if not Cl0 or not Cl1:
            print('plot.mode.MakeBilinearCoonsSurfaceMode.reorient :: '
                  'one or more corner points do not overlap, aborting')
            return

        cpte0 = Ck1.cobj.cpts[ 0].xyz
        cpte1 = Cl1.cobj.cpts[-1].xyz
        if np.allclose(cpte0, cpte1, atol=1e-5):
            Ck1 = Ck1.reverse()

        return [Ck0, Ck1], [Cl0, Cl1]

    def interpolate(self, Ck, Cl):
        try:
            for c in Ck + Cl:
                U, = c.U; nurbs.knot.normalize_knot_vec(U)
            ul, vk = [0, 1], [0, 1]
            return nurbs.surface.make_gordon_surface(Ck, Cl, ul, vk)
            #ders = BC.eval_derivatives(0, 0, 1)
            #N = np.cross(ders[1,0], ders[0,1])
            #if N[2] < 0.0:
            #    BC = BC.swap()
            #return BC
        except nurbs.nurbs.NURBSException:
            traceback.print_exc()

    def do(self):
        try:
            n = self.picked_pos['nurbs'].pop()
        except IndexError:
            return
        c = n[0]
        if not isinstance(c, pobject.PlotCurve):
            return
        self.Ckls.append(c.copy())
        if len(self.Ckls) == 4:
            Ckls = self.reorient(self.Ckls)
            if Ckls:
                BC = self.interpolate(*Ckls)
                if BC:
                    self.m.f.inject(BC, adjust_axes=False)
                    BC.colorize()
                    self.save(BC)
            self.__init__(self.m)

    def undo(self, args):
        BC, = args
        self.m.f.deject(BC, adjust_axes=False)


class _AttachWingJunctionMode(Mode):

    def do(self):
        try:
            t = self.picked_pos['points'].pop()
        except IndexError:
            try:
                t = self.picked_pos['nurbs'].pop()
            except IndexError:
                return
        jnc = self.junction; wi = jnc.wing
        tip = 0 if not jnc.tip else -1
        xyz, xyzi = t[1], wi.TE.cobj.cpts[tip].xyz
        wi.glue()
        wi.glued += [jnc._nurbs]
        wi.translate(xyz - xyzi)
        wi.unglue()
        self.save(xyzi)

    def undo(self, args):
        jnc = self.junction; wi = jnc.wing
        tip = 0 if not jnc.tip else -1
        xyz, xyzi = args[0], wi.TE.cobj.cpts[tip].xyz
        wi.glue()
        wi.glued += [jnc._nurbs]
        wi.translate(xyz - xyzi)
        wi.unglue()


# EXCEPTIONS


class PickedPlotObject(Exception):
    pass


# TODO


#class ExtractBoundariesMode(Mode):
#
#    sidemap = {(0, 0): 0, (0, -1): 1, (1, 0): 2, (1, -1): 3}
#
#    def __init__(self, m):
#        super(ExtractBoundariesMode, self).__init__(m)
#        self.bcs = []
#        self.BGk = []
#
#    def extract(self, s, xyz):
#        borders = [(s.extract(s.U[di][i], di), (di, i))
#                   for di, i in self.sidemap.keys()]
#        mind = np.inf
#        for b, dii in borders:
#            try:
#                u, = b.project(xyz)
#            except nurbs.nurbs.NewtonLikelyDiverged:
#                continue
#            d = nurbs.util.distance(xyz, b.eval_point(u))
#            if d < mind:
#                C, si = b, self.sidemap[dii]
#                mind = d
#        D = nurbs.surface.extract_cross_boundary_deriv(s, si)
#        self.bcs.append(C)
#        return C, D
#
#    def do(self):
#        try:
#            n = self.picked_pos['nurbs'].pop()
#        except IndexError:
#            return
#        n, dummy, dummy, xyz = n
#        if isinstance(n, pobject.PlotSurface):
#            self.C, D = self.extract(n, xyz)
#        else:
#            self.C, D = n, 'undefined'
#        self.C.visible['cobj'] = True
#        self.m.f.inject(self.C)
#        self.BGk.append((self.C, D))
#        self.m.f.IP.user_ns['last_extracted_BGk'] = zip(*self.BGk)
#
#    def end_selection1(self):
#        try:
#            C, D = self.BGk[-1]
#            Cr, Dr = C.reverse(), D.reverse()
#        except IndexError:
#            return
#        except AttributeError:
#            Cr, Dr = C.reverse(), 'undefined'
#        self.BGk[-1] = Cr, Dr
#        Cr._fill_batch()
#        Cr.visible['cobj'] = True
#        self.m.f.deject(self.C)
#        self.m.f.inject(Cr)
#        self.C = self.bcs[-1] = Cr
#        self.m.f.IP.user_ns['last_extracted_BGk'] = zip(*self.BGk)
#
#    def end_selection2(self):
#        try:
#            C, D = self.BGk[-1]
#            D.cobj.Pw[:,:-1] *= - 1.0
#        except IndexError, AttributeError:
#            return
#        self.BGk[-1] = C, D
#        self.m.f.IP.user_ns['last_extracted_BGk'] = zip(*self.BGk)
#
#    def cleanup(self):
#        for bc in self.bcs:
#            try:
#                self.m.f.deject(bc)
#            except ValueError:
#                pass
#
#
#class ReparameterizeCurveMode(Mode):
#
#    def __init__(self, m):
#        super(ReparameterizeCurveMode, self).__init__(m)
#        self.C, self.f = None, None
#        self.reparam_pts = []
#        self.nchpts = 30
#
#    def do(self):
#        try:
#            c, dummy, u, dummy = self.picked_pos['nurbs'].pop()
#        except IndexError:
#            return
#        if not isinstance(c, pobject.PlotCurve):
#            return
#        if not self.C:
#            self.C = c
#        if self.C is not c:
#            return
#        pt = self.m.p._pp.copy()
#        self.reparam_pts.append((pt, u))
#        self.m.f.inject(pt)
#
#    def undo(self, args):
#        Cr, C = args
#        Cr._cobj = C.cobj
#        Cr._p = C.p
#        Cr._U = C.U
#        Cr._set_cpoint_association()
#        Cr._set_point_colors()
#        Cr.color = C.color
#        Cr._fill_batch()
#        Cr._construct_GL_arrays()
#        if hasattr(self, 'chpts2'):
#            self.m.f.deject(*self.chpts2)
#        self.cleanup()
#
#    def refresh(self, dummy):
#        n, p, U, Pw = self.f.var()
#        us = np.linspace(0, 1, self.nchpts)
#        args = n, p, U, Pw, us, self.nchpts
#        C = nurbs.curve.rat_curve_point_v(*args).T
#        chpts1 = [nurbs.point.Point(*c) for c in C]
#        f = self.f._figs[0]
#        if hasattr(self, 'chpts1'):
#            f.deject(*self.chpts1)
#        f.inject(*chpts1)
#        self.chpts1 = chpts1
#
#    def end_selection1(self):
#        n = len(self.reparam_pts) - 1
#        if self.C and n > 1:
#            pts, us = zip(*self.reparam_pts)
#            us = np.asarray(us).reshape((n + 1, 1))
#            us = np.column_stack((us, np.zeros((n + 1, 2))))
#            uk = np.linspace(0, 1, n + 1)
#            U, Pw = nurbs.fit.global_curve_interp(n, us, 2, uk)
#            cpol = nurbs.curve.ControlPolygon(Pw=Pw)
#            f = nurbs.curve.Curve(cpol, (2,), (U,)); self.f = f
#            for cpt in f.cobj.cpts:
#                cpt.line = [0,0,0], [1,0,0]
#                cpt.refresh = self.refresh
#            self.m.f.__class__(f)
#            self.m.f.deject(*pts)
#            self.refresh()
#
#    def end_selection2(self):
#        C, f = self.C, self.f
#        if C and f:
#            Cr = nurbs.curve.reparam_func_curve(C, self.f)
#            self.save(Cr, C)
#            Cr.colorize()
#            self.m.f.deject(C)
#            chpts = [nurbs.point.Point(*Cr.eval_point(u))
#                     for u in np.linspace(0, 1, self.nchpts)]
#            self.m.f.inject(Cr, *chpts)
#            self.chpts2 = chpts
#            self.__init__(self.m)
#
#    def cleanup(self):
#        if self.reparam_pts:
#            pts = zip(*self.reparam_pts)[0]
#            self.m.f.deject(*pts)
#
#
#class CompatibilizeSurfacesMode(Mode):
#
#    def in_place(self, ss, sss):
#        for s, ns in zip(ss, sss):
#            s._cobj = ns.cobj
#            s._p = ns.p
#            s._U = ns.U
#            s._set_cpoint_association()
#            s._set_point_colors()
#            s._fill_batch()
#            s._construct_GL_arrays()
#
#    def undo(self, args):
#        ns, saves = args
#        self.in_place(ns, saves)
#
#    def end_selection1(self):
#        ss = self.picked_pos['nurbs']
#        self.__init__(self.m)
#        if len(ss) < 2:
#            return
#        ss = [n[0] for n in ss
#              if isinstance(n[0], pobject.PlotSurface)]
#        to_save = [s.copy() for s in ss]
#        self.save(ss, to_save)
#
#        try:
#            sss = nurbs.surface.reorient_surfaces(ss)
#        except nurbs.surface.NoCommonEdgeCouldBeFound:
#            traceback.print_exc()
#            return
#        sss = nurbs.surface.make_surfaces_compatible2(sss)
#        self.in_place(ss, sss)
#
#
#class AssociateVertexMode(Mode):
#
#    def do(self):
#        try:
#            v0 = self.picked_pos['points'][0][0]
#            v1 = self.picked_pos['points'][1][0]
#        except IndexError:
#            return
#        del self.picked_pos['points'][:]
#
#        vs, xyzs = [v0], [v0.xyzw.copy()]
#        if hasattr(v0, '_vertices'):
#            for v in v0._vertices:
#                xyz = v._xyzw.copy()
#                vs.append(v); xyzs.append(xyz)
#        self.save(vs, xyzs)
#
#        dxdydz = v1.xyz - v0.xyz
#        for v in vs:
#            nurbs.transform.translate(v._xyzw, dxdydz)
#            if hasattr(v, 'nurbs'):
#                v.nurbs._fill_batch()
#                v.nurbs._construct_GL_arrays()
#
#    def undo(self, args):
#        os, saves = args
#        for o, s in zip(os, saves):
#            o._xyzw[:] = s
#            if hasattr(o, 'nurbs'):
#                o.nurbs._fill_batch()
#                o.nurbs._construct_GL_arrays()
