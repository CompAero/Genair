import time

import numpy as np
from scipy.optimize  import brute, fmin

from ffd.ffd         import FFDVolume

from nurbs.curve     import (Curve,
                             make_composite_curve,
                             arc_length_to_param,
                             param_to_arc_length)
from nurbs.fit       import refit_curve
from nurbs.knot      import (averaging_knot_vec,
                             normalize_knot_vec,
                             KnotOutsideKnotVectorRange)
from nurbs.nurbs     import obj_mat_to_3D
from nurbs.point     import Point
from nurbs.surface   import (Surface,
                             ControlNet)
from nurbs.transform import (shear,
                             translate,
                             scale,
                             rotate,
                             custom)
from nurbs.util      import (point_to_plane,
                             normalize,
                             distance,
                             signed_angle,
                             intersect_line_plane)
from nurbs.volume    import ControlVolume

from plot.figure     import Figure as draw
from plot.pobject    import update_figures

try:
    from io.libplot3d  import (open_file, close_file,
                               write_nblk,
                               write_header, write_header_1d,
                               write_one_block, write_one_block_1d,
                               write_param)
except ImportError:
    pass


__all__ = ['make_axial_from_wing',
           'make_joints_from_axials',
           'set_module_variables',
           'update_module_variables',
           'write_axial_connectivity']


AXIALS, JOINTS = [], []

def set_module_variables(axials, joints):
    global AXIALS, JOINTS
    AXIALS, JOINTS = list(axials), list(joints)
    for J in JOINTS:
        J.glued = [J]
        for A in AXIALS:
            for cpt in A.cobj.cpts:
                if np.allclose(cpt.xyz, J.xyz):
                    J.glued.append(cpt)

def update_module_variables():
    for J in JOINTS:
        J.attach(refresh=False)
    for A in AXIALS:
        A.transform_ffd()
    for J in JOINTS:
        J.connect(refresh=False)
    for A in AXIALS:
        A.ffd.refresh()
        A.ffd._fill_batch()


# AXIAL, JOINT


class Axial(Curve):

    def __init__(self, *args, **kwargs):

        self.ffd = kwargs.pop('ffd')
        self.vb = kwargs.pop('vb')
        super(Axial, self).__init__(*args, **kwargs)

        self.A0 = Curve(self.cobj, self.p, self.U)
        self.F0 = self.ffd.cobj.Pw.copy()

    def __getstate__(self):
        d = self.__dict__.copy()
        ds = d.viewkeys() - \
                {'_cobj', '_p', '_U', 'ffd', 'vb', 'A0', 'F0'}
        for k in ds:
            del d[k]
        return d

    def snap_ffd(self, last=False):
        f, s = (0, 1) if not last else (-1, -2)
        P = self.ffd.cobj.Pw[...,:3]
        n, dummy, l = self.ffd.cobj.n
        for i in xrange(n + 1):
            for k in xrange(l + 1):
                L0, L = P[i,f,k], P[i,s,k] - P[i,f,k]
                P[i,f,k] = intersect_line_plane(L0, L, (0,0,0), (0,1,0))
                P[i,f,k,1] = 0.0
        self.F0 = self.ffd.cobj.Pw.copy()
        update_figures([self.ffd])

    def transform_ffd(self):

        ffd, vb = self.ffd, self.vb
        A0, A1 = self.A0, self

        Pw = ffd.cobj.Pw
        Pw[:] = A1.F0

        for j in xrange(vb.size):
            v = vb[j]

            O0, Y0 = A0.eval_derivatives(v, 1); Y0[0] = 0
            O1, Y1 = A1.eval_derivatives(v, 1); Y1[0] = 0

            Z0 = np.cross((1,0,0), Y0)
            Z1 = np.cross((1,0,0), Y1)

            # taper, t/c, twist

            X, Y, Z = get_local_sys_xsec(Pw[:,j,:])
            #scale(Pw[:,j,:], 0.85, Q=O0)
            #scale(Pw[:,j,:], 0.7, Q=O0, L=X)
            #scale(Pw[2,j,2], 0.7, Q=O0, L=Z)
            #rotate(Pw[:,j,:], 10, Q=O0, L=Y)

            # span, sweep, dihedral

            w = O1 - O0
            translate(Pw[:,j,:], w)

            theta = signed_angle(Z0[1:], Z1[1:])
            rotate(Pw[:,j,:], theta, (1,0,0), O1)

    def _draw(self, *args, **kwargs):
        super(Axial, self)._draw(*args, **kwargs)
        for fig in self._figs:
            if self.ffd not in fig.pos['volumes']:
                fig.inject(self.ffd)

    def print_info(self):

        if not (AXIALS and JOINTS):
            return

        index = AXIALS.index(self) + 1
        print(72 * '-')
        print('{} = {}'.format('Axial', index))
        print('{} = {}'.format('xsec', self.vb))
        print('{} = {}'.format('dof', 7))

    def copy(self):
        return self.__class__(self.cobj, self.p, self.U,
                              ffd=self.ffd, vb=self.vb)


class Joint(Point):

    def __init__(self, *args, **kwargs):
        super(Joint, self).__init__(*args, **kwargs)
        self.attachment = None
        self._setup()

    def _setup(self):
        self.ffd = self
        self.refresh = update_module_variables

    @property
    def isconnected(self):
        return len(self.glued) == 3

    @property
    def isattached(self):
        return (True if self.attachment
                     is not None else False)

    @property
    def type(self):
        if self.isconnected:
            return 1
        elif self.isattached:
            return 2
        return 0

    def __getstate__(self):
        d = self.__dict__.copy()
        ds = d.viewkeys() - {'_xyzw', 'attachment'}
        for k in ds:
            del d[k]
        return d

    def __setstate__(self, d):
        super(Joint, self).__setstate__(d)
        self._setup()

    def connect(self, refresh=True):

        if not (AXIALS and JOINTS) or not self.isconnected:
            return
        self, cpt0, cpt1 = self.glued

        A0 = cpt0.nurbs; i0, = cpt0._i
        A1 = cpt1.nurbs; i1, = cpt1._i

        T0 = 0 if (i0 == 0) else - 1
        T1 = 0 if (i1 == 0) else - 1

        Pw0 = A0.ffd.cobj.Pw[:,T0,:]
        Pw1 = A1.ffd.cobj.Pw[:,T1,:]

        O = A0.eval_point(0 if (i0 == 0) else 1)

        Y0 = get_local_sys_xsec(Pw0)[1]
        Y1 = get_local_sys_xsec(Pw1)[1]

        Z0 = np.cross((1,0,0), Y0)
        Z1 = np.cross((1,0,0), Y1)

        phi = signed_angle(Z0[1:], Z1[1:]) / 2.0

        shear(Pw0, - phi, Z0, Y0, O)
        shear(Pw1,   phi, Z1, Y1, O)

        if not np.allclose(Pw0, Pw1):
            m = np.max(Pw0 - Pw1)
            print('opti.axial.Joint.connect :: '
                  'warning, misaligned control points ({})'
                  .format(m))
        else:
            Pw0 = Pw1

        if refresh: self.refresh()

    def attach(self, refresh=True):

        if not (AXIALS and JOINTS) or not self.isattached:
            return
        a, v = self.attachment

        A = AXIALS[a]
        w = A.eval_point(v) - self.xyz
        for pt in self.glued:
            translate(pt._xyzw, w)
            if pt.iscontrolpoint:
                pt.nurbs._fill_batch()

        if refresh: self.refresh()

    def print_info(self):

        if not (AXIALS and JOINTS):
            return

        def print_connection(i):
            axl, idx = get_axial_joint_indices(self, i)
            print('{} = {}'.format('axial', axl + 1))
            print('{} = {}'.format('index', idx + 1))

        def print_attachment():
            axl, xi = self.attachment
            print('{} = {}'.format('axial', axl + 1))
            print('{} = {}'.format('xi', xi))

        print(72 * '-')
        print('{} = {}'.format('Joint', JOINTS.index(self) + 1))
        print('{} = {}'.format('type', self.type))
        print('{} = {}'.format('dof', 7))
        print_connection(1)
        if self.isconnected:
            print_connection(2)
        elif self.isattached:
            print_attachment()


# TOOLBOX


def make_axial_from_wing(wing, loc, n, p, nFFD, pFFD,
                         offsets=[(0,0),(0,0),(0,0)], vb=None):

    ''' Make an "Axial" given an arbitrary Wing.

    An Axial is a B-spline Curve that dictates the movement of a
    FFDVolume which in turn dictates the movement of an embedded Wing.
    This is achieved by 1) manipulating the Axial's control points, 2)
    automatically transforming the FFD's spanwise cross-sections
    according to the new orientation of the Axial Curve and 3)
    re-evaluating the embedding Wing.  The second step ensures that a
    Wing retains its desirable characteristics (e.g. airfoil sections
    perpendicular to the flow) after deformation.  This mechanism was
    inspired from the paper cited below.

    The Axial Curve is constructed by closely reapproximating one of the
    Wing's characterizing Curves (LE, TE or QC), and its underlying
    FFDVolume is constructed by sweeping end cross-sections (derived
    from the Wing's tip Airfoil(s)) along it using the Wing's trajectory
    Curve and scaling/twisting functions (if any).  Together with the
    "offsets" parameter this process ensures that the resulting
    FFDVolume be just large enough to house the Wing.

    Note that the number of control points and degree used to construct
    the Axial Curve is completely decoupled from those used to construct
    the FFDVolume.  This is desirable, for example, in cases where more
    FFD cross-sections are required for increased local surface control
    but without the typical increase in number of planform design
    variables.

    Also note that each control point making up the Axial must be
    assigned to exactly one Joint prior to manipulation, since it is the
    xyz coordinates of Joints (NOT of the Axial's control points) that
    make up the actual design variables (see make_joints_from_axials).

    Parameters
    ----------
    wing = the Wing to construct the Axial from
    loc = the location of the Axial on the Wing: 'LE, 'QC', 'TE'
    n, p = the number of control points and degree to create the Axial
           with
    nFFD, pFFD = the number of control points and degree to create the
                 FFDVolume with (two 3-tuples); for the vertical
                 direction the degree must be 1 and hence the number of
                 control points must be 2
    offsets = the xyz deltas to expand the FFDVolume with; this is to
              ensure that the Wing can be fully embedded
    vb = (optional) the parameter locations [0,1] corresponding to the
         spanwise FFD cross-sections; if None, a uniform vector is used

    Returns
    -------
    axial = the Axial (a Curve), holding a FFDVolume whose planform is
            similar than the input Wing

    Source
    ------
    Lazarus et al., Axial deformations: an intuitive deformation
    technique, Computer-Aided Design, 1994.

    '''

    pX, pY, pZ = pFFD
    nX, nY, nZ = nFFD
    dX, dY, dZ = offsets

    assert p < n and pX < nX and pY < nY and pZ == 1 and nZ == 2

    wi = wing.copy()

    # Step 1
    if (np.array(dY) != 0.0).any():
        T, Tw, (scx, dummy, scz) = wi.T, wi.Tw, wi.scs
        (n0,), (n1,) = wi.Bv.cobj.n, wi.QC.cobj.n
        T, Tw, scx, scz = \
                resize_wing_curves([T, Tw, scx, scz], dY)
        h0, h1 = wi.halves
        snap = np.allclose(h0.cobj.Pw[:,0,1], 0.0)
        wi.orient(T, Tw=Tw, m=np.max((n0, 4)), show=False)
        wi.fit(K=n1, scs=(scx,None,scz), show=False)
        if snap:
            wi.snap()

    # Step 2
    af0, af1 = wi.airfoils
    sec0 = airfoil_to_sec(af0, nX, pX, dX, dZ)
    if af1:
        sec1 = airfoil_to_sec(af1, nX, pX, dX, dZ)

    # Step 3
    if nY < wing.nurbs.cobj.n[1] + 1:
        print('opti.axial.make_axial_from_wing :: '
              'warning, nFFD[1] < # of CPs parameterizing the wing')
    Qw = np.zeros((nX, nY, nZ, 4))
    if vb is None:
        vb = np.linspace(0, 1, nY)
    V = averaging_knot_vec(nY - 1, pY, vb)
    for j in xrange(nY):
        v = vb[j]
        if af1:
            Qw[:,j,:] = (1.0 - v) * sec0.cobj.Pw + \
                    v * sec1.cobj.Pw
        else:
            Qw[:,j,:] = sec0.cobj.Pw.copy()
        for di, sc in zip(np.identity(3), wi.scs):
            if sc:
                sf = sc.eval_point(v)[0]
                scale(Qw[:,j,:], sf, L=di)
        O, X, Y, Z = get_local_sys_proj(v, wi.T, wi.Bv)
        custom(Qw[:,j,:], R=np.column_stack((X, Y, Z)), T=O)
    U, W = sec0.U
    ffd = FFDVolume(ControlVolume(Pw=Qw), (pX,pY,pZ), (U,V,W))

    # Step 4
    A = refit_curve(eval('wi.' + loc), n, p)
    vb = np.zeros(nY)
    for j in xrange(1, nY - 1):
        O = Qw[0,j,0,:3]
        Y = get_local_sys_xsec(Qw[:,j,:])[1]
        args = (A, O, Y)
        v = brute(isec_plane_curve, [(0.0, 1.0)], args,
                  finish=None)
        v, = fmin(isec_plane_curve, v, args, xtol=0.0, ftol=0.0,
                  disp=False)
        vb[j] = v
    vb[-1] = 1.0

    return Axial(A.cobj, A.p, A.U, ffd=ffd, vb=vb)


def make_joints_from_axials(axials):

    ''' Make Joints given a list of Axials.

    Joints are what manipulate Axials (which in turn manipulate
    FFDVolumes, which in turn manipulate embedded Wings).  While each
    control point of an Axial can only be assigned to one Joint, a Joint
    on the other hand is not restricted to one control point.  This is
    desirable when connecting multiple Axial tips together, where then a
    single Joint is used to group the overlapping tip control points.

    Although the Axial/Joint mechanism is meant to accomodate Wing
    systems of any topology, currently this function only supports
    Joints that connect at most two adjacent Axials.

    Note that the module variables should be set before interactive use
    (see set_module_variables) and before exporting to Jetstream for
    optimization purposes (see write_axial_connectivity).

    Parameters
    ----------
    axials = a list of Axials to extract the Joints from

    Returns
    -------
    joints = a list of Joint design variables

    '''

    class NonUniqueXYZ(Exception):
        pass

    joints = []
    for A in axials:
        cpts = A.cobj.cpts
        for cpt in cpts:
            try:
                for J in joints:
                    if np.allclose(J.xyz, cpt.xyz):
                        raise NonUniqueXYZ
                J = Joint(*cpt.xyz)
                joints.append(J)
            except NonUniqueXYZ:
                continue
    print('opti.axial.make_joints_from_axials :: '
          '{} unique joints were extracted'.format(len(joints)))
    return joints


def write_axial_connectivity(con_file='axial.con',
        axial_file='axial.b', ffd_file='ffd.b'):

    if not (AXIALS and JOINTS):
        print('opti.axial.write_axial_connectivity :: '
              'set module variables first, aborting')
        return

    try:
        import io.libplot3d
    except ImportError:
        print('opti.axial.write_axial_connectivity :: '
              'could not import the plot3d utility library, aborting')
        return

    naxl = len(AXIALS)
    njnt = len(JOINTS)

    with open(con_file, 'w') as fh:

        fh.write('Axial connectivity file for jetstream\n')
        fh.write('number of axials\n')
        fh.write('{:11}\n'.format(naxl))
        fh.write(23 * '_' + '\n')
        fh.write((3 * '| {} ' + '|\n')
                  .format('axial', 'nxsec', 'dof'))
        FRMT0 = '{:5}{:8}\n'
        FRMT1 = '{:20}{:18}\n'

        for i in xrange(naxl):
            axl = AXIALS[i]
            nxsec = axl.vb.size
            fh.write(FRMT0.format(i + 1, nxsec))
            for j in xrange(nxsec):
                fh.write(FRMT1.format(7, axl.vb[j]))

        fh.write('\nnumber of points\n') # Joints are Points in Jtstrm
        fh.write('{:11}\n'.format(njnt))
        fh.write(38 * '_' + '\n')
        fh.write((5 * '| {} ' + '|\n')
                  .format('point', 'type', 'dof', 'axial', 'index'))
        FRMT0 = '{:5}{:7}{:7}{:7}{:8}\n'
        FRMT1 = '{:26}{:27}\n'

        for i in xrange(njnt):
            jnt = JOINTS[i]
            axl, idx = get_axial_joint_indices(jnt, 1)
            fh.write(FRMT0.format(i + 1, jnt.type, 7, axl + 1, idx + 1))
            if jnt.isconnected:
                axl, idx = get_axial_joint_indices(jnt, 2)
                fh.write(FRMT0.format('', '', '', axl + 1, idx + 1))
            elif jnt.isattached:
                a = jnt.attachment
                fh.write(FRMT1.format(a[0] + 1, a[1]))

        fh.write('\nThis file was automatically generated')
        fh.write('\nby Genair on ' + time.asctime())

    ffd = [A.ffd for A in AXIALS]
    write_bspline_curves(AXIALS, axial_file)
    write_bspline_volumes(ffd, ffd_file)


# UTILITIES


def airfoil_to_sec(af, nX, pX, dX, dZ):
    x0, x1 = af._get_xbounds()
    z0, z1 = af._get_zbounds()
    x0 -= dX[0]; x1 += dX[1]
    z0 -= dZ[0]; z1 += dZ[1]
    PwX = np.linspace(x0, x1, nX)
    Pw = np.zeros((nX, 2, 4))
    Pw[:,0,0] = PwX
    Pw[:,1,0] = PwX
    Pw[:,0,2] = z0
    Pw[:,1,2] = z1
    Pw[:,:,3] = 1.0
    return Surface(ControlNet(Pw=Pw), (pX,1))


def resize_wing_curves(Cs, dY):

    Cs, C0s, C1s = list(Cs), 4 * [None], 4 * [None]

    if dY[0] != 0.0:
        a0 = abs(dY[0])
        if dY[0] > 0.0:
            C0s[0] = Cs[0].extend(a0, end=False)
            u0 = arc_length_to_param(Cs[0], a0)
            for i, C in enumerate(Cs[1:], start=1):
                if C:
                    a0 = param_to_arc_length(C, u0)
                    C0s[i] = C.extend(a0, end=False)
        else:
            u0 = arc_length_to_param(Cs[0], a0)
            for i, C in enumerate(Cs):
                if C:
                    dummy, Cs[i] = C.split(u0)

    if dY[1] != 0.0:
        a1 = abs(dY[1])
        if dY[1] > 0.0:
            C1s[0] = Cs[0].extend(a1, end=True)
            u1 = arc_length_to_param(Cs[0].reverse(), a1)
            for i, C in enumerate(Cs[1:], start=1):
                if C:
                    a1 = param_to_arc_length(C.reverse(), u1)
                    C1s[i] = C.extend(a1, end=True)
        else:
            u1 = arc_length_to_param(Cs[0],
                                     param_to_arc_length(Cs[0]) - a1)
            for i, C in enumerate(Cs):
                if C:
                    Cs[i], dummy = C.split(u1)

    Cs1 = []
    for C0, C, C1 in zip(C0s, Cs, C1s):
        Css = []
        if C0: Css.append(C0)
        if C: Css.append(C)
        if C1: Css.append(C1)
        if len(Css) > 1: C = make_composite_curve(Css)
        if C: U, = C.U; normalize_knot_vec(U)
        Cs1.append(C)
    return Cs1


def get_local_sys_proj(v, T, Bv):
    O, Y = T.eval_derivatives(v, 1); Y[0] = 0.0
    Z = Bv.eval_point(v)
    X = np.cross(Y, Z)
    X, Y, Z = [normalize(V) for V in X, Y, Z]
    return O, X, Y, Z


def get_local_sys_xsec(Q):
    Q = obj_mat_to_3D(Q)
    O = Q[ 0, 0]
    X = Q[-1, 0] - O
    Z = Q[ 0,-1] - O
    Y = np.cross(Z, X)
    return [normalize(v) for v in X, Y, Z]


def isec_plane_curve(u, C, S, T):
    try:
        xyz0 = C.eval_point(u)
        xyz1 = point_to_plane(S, T, xyz0)
        return distance(xyz0, xyz1)
    except KnotOutsideKnotVectorRange:
        return np.inf


def get_axial_joint_indices(joint, i):
    cpt = joint.glued[i]
    return AXIALS.index(cpt.nurbs), cpt._i[0]


def write_bspline_curves(crvs, bspline_file):
    ncrv = len(crvs)
    jmax = np.zeros(ncrv, dtype='i', order='F')
    degree = np.zeros(ncrv, dtype='i', order='F')
    for i in xrange(ncrv):
        jmax[i] = np.array(crvs[i].cobj.cpts.shape)
        degree[i] = np.array(crvs[i].p)
    try:
        open_file(bspline_file, 'big_endian')
        write_nblk(ncrv)
        write_header_1d(jmax, ncrv)
        write_header_1d(degree + 1, ncrv)
        for i, crv in enumerate(crvs):
            cpxyz, U = obj_mat_to_3D(crv.cobj.Pw), crv.U
            write_one_block_1d(jmax[i], 3, cpxyz)
            write_param(jmax[i] + degree[i] + 1, np.hstack(U))
    finally:
        close_file()


def write_bspline_volumes(vols, bspline_file):
    nvol = len(vols)
    jkmmax = np.zeros((3, nvol), dtype='i', order='F')
    degree = np.zeros((3, nvol), dtype='i', order='F')
    for i in xrange(nvol):
        jkmmax[:,i] = np.array(vols[i].cobj.cpts.shape)
        degree[:,i] = np.array(vols[i].p)
    try:
        open_file(bspline_file, 'big_endian')
        write_nblk(nvol)
        write_header(jkmmax, nvol)
        write_header(degree + 1, nvol)
        for i, vol in enumerate(vols):
            cpxyz, U = obj_mat_to_3D(vol.cobj.Pw), vol.U
            write_one_block(jkmmax[:,i], 3, cpxyz)
            write_param(sum(jkmmax[:,i] + degree[:,i] + 1), np.hstack(U))
    finally:
        close_file()


# TODO


#def append_two_axials(a0, a1):
#    a0, a1 = a0.copy(), a1.copy()
#    for a in a0, a1:
#        reparam_arc_length_curve(a)
#    for a in a0, a1:
#        U, = a.U
#        for ffd in a0.ffd, a1.ffd:
#            remap_knot_vec(ffd.U[1], U[0], U[-1])
#        a.vb *= U[-1]
#    A = make_composite_curve([a0, a1], remove=False)
#    ffd = make_composite_volume([a0.ffd, a1.ffd], di=1)
#    vb = np.hstack((a0.vb, a1.vb[1:] + a0.vb[-1]))
#    normalize_knot_vec(A.U[0])
#    normalize_knot_vec(ffd.U[1])
#    vb /= vb[-1]
#    return Axial(A.cobj, A.p, A.U, ffd=ffd, vb=vb)
