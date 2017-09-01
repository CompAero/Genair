import sys

import numpy as np

from nurbs.curve   import Curve
from nurbs.nurbs   import (obj_mat_to_3D,
                           obj_mat_to_4D)
from nurbs.point   import Point
from nurbs.surface import Surface
from nurbs.volume  import Volume, ControlVolume

from plot.figure   import Figure as draw
from plot.pobject  import PlotObject

try:
    from io.libplot3d  import (open_file, close_file,
                               read_nblk, write_nblk,
                               read_header, write_header,
                               read_one_block, write_one_block,
                               read_param, write_param)
except ImportError:
    pass


__all__ = ['Grid']


class Grid(PlotObject):

    EXTRACTMAP = {1: (0, 0), 2: (1, 0),
                  3: (0, 1), 4: (1, 1),
                  5: (0, 2), 6: (1, 2)}

    def __init__(self):

        '''

        '''

        try:
            import io.libplot3d
        except ImportError:
            print('opti.grid.Grid.__init__ :: '
                  'warning, could not import the plot3d utility library')

        self.blk = []
        self.iface = []
        self.bcface = []

        self.ptch = []
        self.stch = []

# FLOW IN / OUT

    def read_grid(self, plot3d_file='grid.g', endian='big_endian', show=False):

        ''' Pre: None '''

        self.blk = []

        try:
            open_file(plot3d_file, endian)
            nblk = read_nblk()
            jkmmax = np.zeros((3, nblk), dtype='i', order='F')
            read_header(jkmmax, nblk)
            for ib in xrange(nblk):
                nxyz = np.append(jkmmax[:,ib], 3)
                xyz = np.zeros(nxyz, order='F')
                read_one_block(jkmmax[:,ib], 3, xyz)
                P0 = Point(*xyz[ 0, 0, 0])
                P1 = Point(*xyz[ 0, 0,-1])
                P2 = Point(*xyz[ 0,-1, 0])
                P3 = Point(*xyz[ 0,-1,-1])
                P4 = Point(*xyz[-1, 0, 0])
                P5 = Point(*xyz[-1, 0,-1])
                P6 = Point(*xyz[-1,-1, 0])
                P7 = Point(*xyz[-1,-1,-1])
                cvol = ControlVolume([[[P0, P1], [P2, P3]],
                                      [[P4, P5], [P6, P7]]])
                blk = Block(cvol, (1,1,1))
                blk.indx, blk.xyz = ib + 1, xyz
                self.blk.append(blk)
            nnode = sum([blk.xyz.size for blk in self.blk]) / 3
            print('{} has {} nonunique nodes.'.format(plot3d_file, nnode))
        finally:
            close_file()

        if show:
            return draw(self)

    def read_connectivity(self, con_file='grid.con', show=True):

        ''' Pre: Grid '''

        def read_line(query_type=True):
            line = read_line_split(fh)
            dit = np.array(line[-6:]).reshape((2,3))
            typ = line[-9] if query_type else 0
            blk, side = line[-8:-6]
            return typ, blk, side, dit

        self.iface = []
        self.bcface = []

        with open(con_file) as fh:

            fh.readline()
            fh.readline()
            nblk = int(fh.readline()); fh.readline()
            nsfc = int(fh.readline()); fh.readline()
            nifc = int(fh.readline())
            fh.readline()
            fh.readline()

            assert nblk == len(self.blk)

            i0, i1 = 1, 1
            for i in xrange(nsfc):
                typ, ib, side, dit = read_line()
                blk, udi = self.blk[ib-1], self.EXTRACTMAP[side]
                face = blk.extract(*udi)
                if typ > 0:
                    bcface = Boundary(face.cobj, face.p, face.U)
                    bcface.indx = i0; i0 += 1
                    bcface.type = typ
                    bcface.blk  = ib
                    bcface.side = side
                    bcface.dit  = dit
                    self.bcface.append(bcface)
                    continue
                dummy, ib1, side1, dit1 = read_line(False)
                iface = Interface(face.cobj, face.p, face.U)
                iface.indx = i1; i1 += 1
                iface.type = 0
                iface.blk  = np.array([ib, ib1])
                iface.side = np.array([side, side1])
                iface.dit1 = dit
                iface.dit2 = dit1
                self.iface.append(iface)

        assert((nsfc == len(self.bcface) + len(self.iface)) and
               (nifc == len(self.iface)))

        self._colorize()
        if show:
            return draw(self)

    def read_solution(self, plot3d_file='results.q', endian='big_endian'):

        ''' Pre: Grid '''

        try:
            open_file(plot3d_file, endian)
            nblk = read_nblk()
            jkmmax = np.zeros((3, nblk), dtype='i', order='F')
            read_header(jkmmax, nblk)
            for ib, blk in enumerate(self.blk):
                read_param(4, np.zeros(4))
                nq = np.append(jkmmax[:,ib], 5)
                q = np.zeros(nq, order='F')
                read_one_block(jkmmax[:,ib], 5, q)
                blk.q = q
        finally:
            close_file()

    def write_grid(self, plot3d_file='grid.g.out', endian='big_endian'):

        ''' Pre: Grid '''

        nblk = len(self.blk)
        jkmmax = np.zeros((3, nblk), dtype='i', order='F')
        for ib, blk in enumerate(self.blk):
            jkmmax[:,ib] = blk.xyz.shape[:-1]
        try:
            open_file(plot3d_file, endian)
            write_nblk(nblk)
            write_header(jkmmax, nblk)
            for ib, blk in enumerate(self.blk):
                write_one_block(jkmmax[:,ib], 3, blk.xyz)
        finally:
            close_file()

    def write_connectivity(self, con_file='grid.con.wb'):

        ''' Pre: Grid, Connectivity '''

        def write_face(indx, type, blk, side, dit):
            fh.write(FRMT.format(indx, type, blk, side,
                                 dit[0,0], dit[0,1], dit[0,2],
                                 dit[1,0], dit[1,1], dit[1,2], ''))

        FRMT = '{:5}{:7}{:7}{:7}{:6}{:11}{:11}{:6}{:11}{:11}{:18}\n'

        nblk = len(self.blk)
        nifc = len(self.iface)
        nsfc = nifc + len(self.bcface)

        with open(con_file, 'w') as fh:
            fh.write('Block connectivity file for Diablo\n')
            fh.write('number of blocks\n')
            fh.write('{:12}\n'.format(nblk))
            fh.write('number of subfaces\n')
            fh.write('{:12}\n'.format(nsfc))
            fh.write('number of interfaces\n')
            fh.write('{:12}\n'.format(nifc))
            fh.write(100 * '_' + '\n')
            fh.write((' ' +
                      4 * '| {} ' +
                      2 * '| {} | {} |   {} ' +
                      '|\n')
                      .format('face', 'type', 'blk', 'side',
                              'it1', 'it1begin', 'it1num',
                              'it2', 'it2begin', 'it2num'))
            for f in self.iface + self.bcface:
                if f.type > 0:
                    write_face(f.indx, f.type, f.blk, f.side, f.dit)
                elif f.type == 0:
                    write_face(f.indx, f.type, f.blk[0], f.side[0], f.dit1)
                    write_face(     '',    '', f.blk[1], f.side[1], f.dit2)

    def write_solution(self, tecplot_file='results.plt'):

        ''' Pre: Grid, Connectivity, Solution '''

        try:
            from io.libtecplot import (open_file_tecplot, close_file_tecplot,
                                       write_zone_header,
                                       write_zone_data,
                                       write_face_connections)
        except ImportError:
            print('opti.grid.Grid.write_solution :: '
                  'could not import the tecplot utility library, aborting')
            return

        build_face_connection_lists(self)
        try:
            open_file_tecplot(tecplot_file)
            for ib, blk in enumerate(self.blk):
                FName = 'BLOCK' + str(ib + 1)
                xyz, q, fcl = blk.xyz, blk.q, blk.fcl
                jkmmax = xyz.shape[:-1]
                xyzq = np.concatenate((xyz, q), axis=-1)
                xyzq = xyzq.flatten(order='F')
                fcl = np.array(fcl); nfcl = fcl.size / 4
                write_zone_header(FName, nfcl, *jkmmax)
                write_zone_data(xyzq.size, xyzq)
                write_face_connections(fcl)
        finally:
            close_file_tecplot()

    def write_FVBND(self, fvbnd_file='grid.g.fvbnd'):

        ''' Pre: Grid, Connectivity '''

        sidemap = {1: '1 1 1 $ 1 $', 2: '$ $ 1 $ 1 $',
                   3: '1 $ 1 1 1 $', 4: '1 $ $ $ 1 $',
                   5: '1 $ 1 $ 1 1', 6: '1 $ 1 $ $ $'}

        b1 = [bcface for bcface in self.bcface if (bcface.type == 1 or
                                                   bcface.type >= 101)]
        b1 = sorted(b1, key=lambda b: b.blk)
        nptch = len(b1)

        with open(fvbnd_file, 'w') as fh:
            fh.write('FVBND 1 4\n')
            for b in b1:
                fh.write('BoundaryFace' + str(b.blk) + '\n')
            fh.write('BOUNDARIES\n')
            for i, b in enumerate(b1):
                fh.write('{} {} {} F 0\n'.format(i + 1, b.blk,
                                                 sidemap[b.side]))

# OPTI IN / OUT

    def read_map(self, map_file='grid.map', endian='big_endian', show=False):

        ''' Pre: Grid '''

        try:
            open_file(map_file, endian)
            nmap = read_nblk()
            assert nmap == len(self.blk)
            jkmmax = np.zeros((3, nmap), dtype='i', order='F')
            read_header(jkmmax, nmap)
            for im, blk in enumerate(self.blk):
                ncpxyz = np.append(jkmmax[:,im], 3)
                cpxyz = np.zeros(ncpxyz, order='F')
                read_one_block(jkmmax[:,im], 3, cpxyz)
                Pw = obj_mat_to_4D(cpxyz)
                cvol = ControlVolume(Pw=Pw)
                blk.map = Map(cvol, (3,3,3))
                blk.map.indx = blk.indx
            nnode = sum([blk.map.cobj.Pw.size for blk in self.blk]) / 4
            print('{} has {} nonunique nodes.'.format(map_file, nnode))
        finally:
            close_file()

        if show:
            maps = [blk.map for blk in self.blk]
            return draw(*maps)

    def read_patch(self, patch_file='patch.con', show=True):

        ''' Pre: Grid, Map '''

        self.ptch = []
        self.stch = []

        with open(patch_file) as fh:

            fh.readline()
            fh.readline()
            nptch = int(fh.readline())
            fh.readline()
            fh.readline()

            for ip in xrange(nptch):
                line = read_line_split(fh)
                im, side, dof = line[1:]
                map, udi = self.blk[im-1].map, self.EXTRACTMAP[side]
                ptch = map.extract(*udi)
                ptch = Patch(ptch.cobj, ptch.p, ptch.U)
                ptch.indx = ip + 1
                ptch.map = im
                ptch.side = side
                ptch.dof = dof
                self.ptch.append(ptch)

            fh.readline()
            fh.readline()
            nstch = int(fh.readline())
            fh.readline()
            fh.readline()

            for i in xrange(nstch):
                line = read_line_split(fh)
                ip, edge, di = (np.zeros(2, dtype='i'),
                                np.zeros(2, dtype='i'),
                                np.zeros(2, dtype='i'))
                typ, dof, conty, ip[0], edge[0], di[0] = line[1:]
                if typ == 0:
                    line = read_line_split(fh)
                    ip[1], edge[1], di[1] = line
                ptch, udi = self.ptch[ip[0]-1], self.EXTRACTMAP[edge[0]]
                stch = ptch.extract(*udi)
                stch = Stitch(stch.cobj, stch.p, stch.U)
                stch.indx = i + 1
                stch.joined = True if typ == 0 else False
                stch.dof = dof
                stch.conty = conty
                stch.ptch = ip
                stch.edge = edge
                stch.dir = di
                self.stch.append(stch)

        self._colorize()
        if show:
            return draw(*(self.stch+self.ptch))

    def write_map(self):

        ''' '''

        raise NotImplementedError

    def write_patch(self, ffds=None, patch_file='patch.ffd.con'):

        ''' Pre: Grid, Map, Patch '''

        nptch = len(self.ptch)
        nstch = len(self.stch)
        nffd = len(ffds) if ffds is not None else 0

        with open(patch_file, 'w') as fh:

            fh.write('Patch connectivity file for jetstream\n')
            fh.write('number of patches\n')
            fh.write('{:11}\n'.format(nptch))
            fh.write(28 * '_' + '\n')
            fh.write((4 * '| {} ' + '|\n')
                      .format('patch', 'map', 'side', 'dof'))
            FRMT = '{:5}{:8}{:4}{:7}\n'

            for i in xrange(nptch):
                ptch = self.ptch[i]
                fh.write(FRMT.format(ptch.indx, ptch.map, ptch.side, ptch.dof))

            fh.write('\nnumber of stitches\n')
            fh.write('{:11}\n'.format(nstch))
            fh.write(52 * '_' + '\n')
            fh.write((7 * '| {} ' + '|\n')
                      .format('stitch', 'type', 'dof', 'conty', 'patch',
                              'edge', 'dir'))
            FRMT = '{:5}{:7}{:7}{:6}{:10}{:6}{:8}\n'

            for i in xrange(nstch):
                stch = self.stch[i]
                fh.write(FRMT.format(stch.indx, 0 if stch.joined else 1,
                                     stch.dof, stch.conty, stch.ptch[0],
                                     stch.edge[0], stch.dir[0]))
                if stch.joined:
                    fh.write(FRMT.format('', '', '', '', stch.ptch[1],
                                         stch.edge[1], stch.dir[1]))

            if not nffd:
                return

            fh.write('\nnumber of FFD volumes\n')
            fh.write('{:11}\n'.format(nffd))
            fh.write(42 * '_' + '\n')
            fh.write((6 * '| {} ' + '|\n')
                      .format('ffd', 'dof', 'embedded', 'patch', ' j', ' k'))
            FRMT0 = '{:5}{:4}{:12}\n'
            FRMT1 = '{:29}{:6}{:5}\n'

            for i in xrange(nffd):
                ffd = ffds[i]
                ne = len(ffd.embedded)
                fh.write(FRMT0.format(i + 1, 7, ne))
                for j, ptch in enumerate(self.ptch, start=1):
                    cpts = ptch.cobj.cpts
                    for nm, cpt in np.ndenumerate(cpts):
                        if hasattr(cpt, 'embed') and cpt.embed is ffd:
                            n, m = np.array(nm) + 1
                            fh.write(FRMT1.format(j, n, m))

# MISC

    def _colorize(self):
        ''' '''
        for o in self.iface + self.bcface + self.ptch + self.stch:
            o._colorize()

    def _draw(self):
        ''' '''
        d = []
        if self.blk:
            d += self.blk
        if self.bcface:
            d += self.bcface
        return d


# BLOCK, INTERFACE, BOUNDARY


class Block(Volume):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', 'indx', 'xyz', 'q', 'map', 'fcl'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('block', self.indx))


class Face(Surface):

    def __setstate__(self, d):
        super(Face, self).__setstate__(d)
        self._colorize()

    def _colorize(self):
        if self.type == 2: self.visible['nurbs'] = False
        if   self.type == 0: c = (50, 50, 0, 155) # interface
        elif self.type == 1: c = (255, 0, 0, 255) # wall
        elif self.type == 2: c = (0, 255, 0, 255) # farfield
        elif self.type == 3: c = (0, 0, 255, 255) # symmetry
        elif self.type >= 1: c = (255, 0, 0, 255)
        self.color = c

    def print_info(self, blk, side, dit):
        print('{} = {}'.format('blk', blk))
        print('{} = {}'.format('side', side))
        print('{} = {}'.format('it1, it1begin, it1num', dit[0,:]))
        print('{} = {}'.format('it2, it2begin, it2num', dit[1,:]))


class Interface(Face):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', '_trimcurves', 'indx', 'type', 'blk',
                'side', 'dit1', 'dit2'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('interface', self.indx))
        super(Interface, self).print_info(self.blk[0], self.side[0], self.dit1)
        super(Interface, self).print_info(self.blk[1], self.side[1], self.dit2)


class Boundary(Face):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', '_trimcurves', 'indx', 'type', 'blk',
                'side', 'dit'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('boundary', self.indx))
        print('{} = {}'.format('type', self.type))
        super(Boundary, self).print_info(self.blk, self.side, self.dit)


# MAP, PATCH, STITCH


class Map(Volume):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', 'indx'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('Map', self.indx))


class Patch(Surface):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', '_trimcurves', 'indx', 'map', 'side',
                'dof'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def __setstate__(self, d):
        super(Patch, self).__setstate__(d)
        self._colorize()

    def _colorize(self):
        self.color = (255, 125, 75, 255)

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('patch', self.indx))
        print('{} = {}'.format('map', self.map))
        print('{} = {}'.format('side', self.side))
        print('{} = {}'.format('dof', self.dof))


class Stitch(Curve):

    def __getstate__(self):
        d = self.__dict__.copy()
        save = {'_cobj', '_p', '_U', 'indx', 'joined', 'dof', 'conty', 'ptch',
                'edge', 'dir'}
        ds = d.viewkeys() - save
        for k in ds:
            del d[k]
        return d

    def __setstate__(self, d):
        super(Stitch, self).__setstate__(d)
        self._colorize()

    def _colorize(self):
        self.color = (255, 255, 0, 155) if self.joined else \
                     (0, 255, 255, 255)

    def print_info(self):
        print(72 * '-')
        print('{} = {}'.format('stitch', self.indx))
        print('{} = {}'.format('joined', self.joined))
        print('{} = {}'.format('dof', self.dof))
        print('{} = {}'.format('conty', self.conty))
        print('{} = {}'.format('ptch', self.ptch[0]))
        print('{} = {}'.format('edge', self.edge[0]))
        print('{} = {}'.format('dir', self.dir[0]))
        if self.joined:
            print('{} = {}'.format('ptch', self.ptch[1]))
            print('{} = {}'.format('edge', self.edge[1]))
            print('{} = {}'.format('dir', self.dir[1]))


# UTILITIES


def build_face_connection_lists(grid):
    ''' '''

    jkms = np.zeros(3, 'i')
    jkmf = np.zeros(3, 'i')
    bit = np.zeros(3, 'i')
    jkm = np.zeros(3, 'i')

    for blk in grid.blk:
        blk.fcl = []

    for iface in grid.iface:

        cell_indices = []

        for blk, side, dit in zip(iface.blk, iface.side, (iface.dit1,
                                                          iface.dit2)):
            jkmmax = grid.blk[blk-1].xyz.shape[:-1]

            # Normal
            di = (side + 1) // 2; di -= 1
            bit[di] = 1
            if np.mod(side, 2) == 1:
                jkms[di] = 1
                jkmf[di] = 2
            else:
                jkms[di] = jkmmax[di] - 1
                jkmf[di] = jkmmax[di]

            # Tangent 1
            it1 = np.abs(dit[0,0]); it1 -= 1
            bit[it1] = np.sign(dit[0,0])
            jkms[it1] = dit[0,1]
            if bit[it1] == - 1:
                jkms[it1] -= 1
            jkmf[it1] = jkms[it1] + bit[it1] * (dit[0,2] - 1)

            # Tangent 2
            it2 = np.abs(dit[1,0]); it2 -= 1
            bit[it2] = np.sign(dit[1,0])
            jkms[it2] = dit[1,1]
            if bit[it2] == - 1:
                jkms[it2] -= 1
            jkmf[it2] = jkms[it2] + bit[it2] * (dit[1,2] - 1)

            cell_index = []
            for jdi in xrange(jkms[di], jkmf[di], bit[di]):
                jkm[di] = jdi
                for jit1 in xrange(jkms[it1], jkmf[it1], bit[it1]):
                    jkm[it1] = jit1
                    for jit2 in xrange(jkms[it2], jkmf[it2], bit[it2]):
                        jkm[it2] = jit2
                        J, K, M = jkm
                        # Tecplot's cell index definition
                        ci = J + \
                            (K - 1) * jkmmax[0] + \
                            (M - 1) * jkmmax[0] * jkmmax[1]
                        cell_index.append(ci)
            cell_indices.append(cell_index)

        # Update face connection lists on either Block

        blk1, blk2 = iface.blk

        fcl = []
        for cz, cr in zip(*cell_indices):
            fz = iface.side[0]
            zr = blk2
            fcl += [cz, fz, zr, cr]
        grid.blk[blk1-1].fcl += fcl

        fcl = []
        for cr, cz in zip(*cell_indices):
            fz = iface.side[1]
            zr = blk1
            fcl += [cz, fz, zr, cr]
        grid.blk[blk2-1].fcl += fcl


def read_line_split(fh):
    line = fh.readline().split()
    return [int(i) for i in line]


# TODO


#def quick_reapprox_patches(self, n=10, m=10):
#
#    from nurbs.fit     import global_surf_approx_fixednm
#    from nurbs.surface import ControlNet
#
#    sidemap = {1: '0,:,:', 2: '-1,:,:',
#               3: ':,0,:', 4: ':,-1,:',
#               5: ':,:,0', 6: ':,:,-1'}
#
#    ptch = [bcface for bcface in self.bcface if bcface.type == 1]
#
#    ptchr = []
#    for p in ptch:
#        blk = self.blk[p.blk - 1]
#        ind = sidemap[p.side]
#        Q = eval('blk.xyz[' + ind + ']')
#
#        r, s = Q.shape[:2]
#        U, V, Pw = global_surf_approx_fixednm(r - 1, s - 1, Q, 3, 3, n, m)
#
#        pr = Surface(ControlNet(Pw=Pw), (3,3), (U, V))
#        ptchr.append(pr)
#
#    return ptchr
