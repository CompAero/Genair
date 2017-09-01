import time
import traceback

import numpy as np

import curve
import point
import surface


class Entity(object):

    def roll_section_counters(self):
        ParameterDataSection.sequence_number += ParameterData.current_line_count
        DirectoryEntrySection.sequence_number += 2

    def parse(self):
        self.DE.parse()
        self.PD.parse()
        self.roll_section_counters()

    def unparse(self):
        global directory_entry_lines, parameter_data_lines
        parameter_data_lines += self.PD.unparse()
        directory_entry_lines += self.DE.unparse()
        self.roll_section_counters()


class Entity116(Entity): # Point

    def __init__(self, point=None):
        self.PD = ParameterData116(point)
        self.DE = DirectoryEntry116()


class Entity126(Entity): # Rational B-Spline Curve

    def __init__(self, curve=None):
        self.PD = ParameterData126(curve)
        self.DE = DirectoryEntry126()


class Entity142(Entity): # Curve on a Parametric Surface

    def __init__(self, curve=None):
        self.PD = ParameterData142(curve)
        self.DE = DirectoryEntry142()


class Entity128(Entity): # Rational B-Spline Surface

    def __init__(self, surface=None):
        self.PD = ParameterData128(surface)
        self.DE = DirectoryEntry128()


class Entity144(Entity): # Trimmed (Parametric) Surface

    def __init__(self, surface=None):
        self.PD = ParameterData144(surface)
        self.DE = DirectoryEntry144()


class DirectoryEntry(object):

    def __init__(self):
        self.fields = {1:0,   # Entity type number
                       2:0,   # Parameter data pointer
                       3:'',  # Structure
                       4:'',  # Line font pattern
                       5:'',  # Level
                       6:'',  # View
                       7:'',  # Transformation matrix
                       8:'',  # Label display associativity
                       9:'',  # Status number
                       10:0,  # Sequence number
                       11:0,  # Entity type number
                       12:'', # Line weight number
                       13:'', # Color number
                       14:0,  # Parameter line count number
                       15:'', # Form number
                       16:'', # Reserved field
                       17:'', # Reserved field
                       18:'', # Entity label
                       19:'', # Entity subscript number
                       20:0}  # Sequence number

    def parse(self):
        l0, l1 = fh.readline(), fh.readline()
        ParameterDataSection.sequence_number = int(l0[8:16])
        ParameterData.current_line_count = int(l1[24:32])

    def unparse(self):
        self.fields[2]  = ParameterDataSection.sequence_number
        self.fields[14] = ParameterData.current_line_count
        self.fields[10] = DirectoryEntrySection.sequence_number
        self.fields[20] = DirectoryEntrySection.sequence_number + 1
        l0 = ('{0[1]:8}{0[2]:8}{0[3]:8}{0[4]:8}{0[5]:8}'
              '{0[6]:8}{0[7]:8}{0[8]:8}{0[9]:8}'.format(self.fields))
        l1 = ('{0[11]:8}{0[12]:8}{0[13]:8}{0[14]:8}{0[15]:8}'
              '{0[16]:8}{0[17]:8}{0[18]:>8}{0[19]:8}'.format(self.fields))
        return fill_sequence_field([l0, l1], 'D', self.fields[10])


class DirectoryEntry116(DirectoryEntry):

    def unparse(self):
        self.fields[1] = self.fields[11] = 116
        return super(DirectoryEntry116, self).unparse()


class DirectoryEntry126(DirectoryEntry):

    def unparse(self):
        self.fields[1] = self.fields[11] = 126
        return super(DirectoryEntry126, self).unparse()


class DirectoryEntry142(DirectoryEntry):

    def unparse(self):
        self.fields[1] = self.fields[11] = 142
        return super(DirectoryEntry142, self).unparse()


class DirectoryEntry128(DirectoryEntry):

    def unparse(self):
        self.fields[1] = self.fields[11] = 128
        return super(DirectoryEntry128, self).unparse()


class DirectoryEntry144(DirectoryEntry):

    def unparse(self):
        self.fields[1] = self.fields[11] = 144
        return super(DirectoryEntry144, self).unparse()


class ParameterData(object):

    current_line_count = 0

    def parse(self):
        fh.seek((ParameterDataSection.file_line_number +
                 ParameterDataSection.sequence_number - 1) * NCOL)
        lines = []
        for i in xrange(ParameterData.current_line_count):
            lines += [fh.readline().strip()]
        self.params = from_free_formatted_data(lines, last=64)

    def unparse(self, params):
        lines = to_free_formatted_data(params, last=64)
        ParameterData.current_line_count = len(lines)
        lines = [l.ljust(64) +
                 '{:8}'.format(DirectoryEntrySection.sequence_number)
                 for l in lines]
        return fill_sequence_field(lines, 'P',
                                   ParameterDataSection.sequence_number)


class ParameterData116(ParameterData):

    def __init__(self, point=None):
        self.point = point

    def parse(self):
        super(ParameterData116, self).parse()
        objs.append(point.Point(*self.params[1:4]))

    def unparse(self):
        params = self.point.xyz.tolist() + [0]
        return super(ParameterData116, self).unparse([116] + params)


class ParameterData126(ParameterData):

    def __init__(self, curve=None):
        self.curve = curve

    def parse(self):
        super(ParameterData126, self).parse()
        n, p = [int(p) for p in self.params[1:3]]
        U = self.params[7:9+n+p]
        w = self.params[9+n+p:10+2*n+p]
        P = self.params[10+2*n+p:13+5*n+p]
        w = np.resize(w, (n + 1, 1))
        P = np.resize(P, (n + 1, 3))
        P *= w
        Pw = np.concatenate((P, w), axis=1)
        c = curve.Curve(curve.ControlPolygon(Pw=Pw), (p,), (U,))
        objs.append(c)

    def unparse(self):
        n, p, U, Pw = self.curve.var()
        params = [n, p]
        params += [0, 0, 0 if self.curve.isrational else 1, 0]
        params += U.tolist()
        params += Pw[:,-1].tolist()
        for i in xrange(n + 1):
            params += (Pw[i,:3] / Pw[i,-1]).tolist()
        params += [U[0], U[-1], 0.0, 0.0, 0.0]
        return super(ParameterData126, self).unparse([126] + params)


class ParameterData142(ParameterData):

    def __init__(self, curve=None):
        self.curve = curve

    def unparse(self):
        params = [2, self.PTS]
        Entity126(self.curve).unparse()
        params += [DirectoryEntrySection.sequence_number - 2, 0, 1]
        return super(ParameterData142, self).unparse([142] + params)


class ParameterData128(ParameterData):

    def __init__(self, surface=None):
        self.surface = surface

    def parse(self):
        super(ParameterData128, self).parse()
        n, m, p, q = [int(p) for p in self.params[1:5]]
        U = self.params[10:10+n+p+2]
        V = self.params[10+n+p+2:10+n+m+p+q+4]
        w = self.params[10+n+m+p+q+4:10+n+m+p+q+4+(n+1)*(m+1)]
        P = self.params[10+n+m+p+q+4+(n+1)*(m+1):-4]
        w = np.resize(w, (m + 1, n + 1, 1)); w = w.transpose((1, 0, 2))
        P = np.resize(P, (m + 1, n + 1, 3)); P = P.transpose((1, 0, 2))
        P *= w
        Pw = np.concatenate((P, w), axis=2)
        s = surface.Surface(surface.ControlNet(Pw=Pw), (p,q), (U,V))
        objs.append(s)

    def unparse(self):
        n, p, U, m, q, V, Pw = self.surface.var()
        params = [n, m, p, q]
        params += [0, 0, 0 if self.surface.isrational else 1, 0, 0]
        params += U.tolist()
        params += V.tolist()
        for j in xrange(m + 1):
            params += Pw[:,j,-1].tolist()
        for j in xrange(m + 1):
            for i in xrange(n + 1):
                params += (Pw[i,j,:-1] / Pw[i,j,-1]).tolist()
        params += [U[0], U[-1]]
        params += [V[0], V[-1]]
        return super(ParameterData128, self).unparse([128] + params)


class ParameterData144(ParameterData):

    def __init__(self, surface=None):
        self.surface = surface

    def unparse(self):

        TCs = self.surface._trimcurves
        outer, inners = TCs[0], TCs[1:] if len(TCs) > 1 else []

        Entity128(self.surface).unparse()
        PTS = DirectoryEntrySection.sequence_number - 2
        params = [PTS, 1, len(inners)]

        E142 = Entity142(outer); E142.PD.PTS = PTS
        E142.unparse()
        params += [DirectoryEntrySection.sequence_number - 2]

        for inner in inners:
            E142 = Entity142(inner); E142.PD.PTS = PTS
            E142.unparse()
            params += [DirectoryEntrySection.sequence_number - 2]

        return super(ParameterData144, self).unparse([144] + params)


class Section(object):

    sequence_number = 1
    file_line_number = 1


class StartSection(Section):

    def parse(self):
        pass

    def unparse(self):
        return fill_sequence_field([72 * ' '], 'S')


class GlobalSection(Section):

    def __init__(self, params):
        params.setdefault(1, ',')       # Parameter delimiter character
        params.setdefault(2, ';')       # Record delimiter character
        params.setdefault(3, 'tmp')     # Product identification from sender
        params.setdefault(4, 'tmp.igs') # File name
        params.setdefault(5, 'gen1.0')  # Native system ID
        params.setdefault(6, 'gen1.0')  # Preprocessor version
        params.setdefault(7, 32)        # Number of binary bits
        params.setdefault(8, 38)        # Single-precision magnitude
        params.setdefault(9, 6)         # Single-precision significance
        params.setdefault(10, 308)      # Double-precision magnitude
        params.setdefault(11, 15)       # Double-precision significance
        params.setdefault(12, 'tmp')    # Product identification
        params.setdefault(13, 1.0)      # Ratio of model to real-world space
        params.setdefault(14, 6)        # Units flag
        params.setdefault(15, '1HM')    # Units name
        params.setdefault(16, 1)        # Max number of line weight gradations
        params.setdefault(17, 0.1)      # Width of maximum line weight in units
        params.setdefault(18, now())    # Date and time of file generation
        params.setdefault(19, 0.0001)   # Minimum user-intended resolution
        params.setdefault(20, 100)      # Approximate maximum coordinate value
        params.setdefault(21, 'user')   # Name of author
        params.setdefault(22, 'UTIAS')  # Author's organization
        params.setdefault(23, 11)       # Version flag
        params.setdefault(24, 0)        # Drafting standard flag
        params.setdefault(25, now())    # Date and time model was created
        params.setdefault(26, '')       # Application identifier
        self.params = params

    def parse(self):
        pass

    def unparse(self):
        params = [str(p) for i, p in sorted(self.params.items())]
        for i in (0, 1, 2, 3, 4, 5, 11, 14, 17, 20, 21, 24, 25):
            string = params[i]
            params[i] = to_hollerith(string)
        lines = to_free_formatted_data(params, last=72)
        GlobalSection.sequence_number = len(lines)
        return fill_sequence_field(lines, 'G')


class DataSection(Section):

    def __init__(self):
        DirectoryEntrySection.sequence_number = 1
        ParameterDataSection.sequence_number = 1

    def parse(self):

        def position_fo():
            fh.seek((DirectoryEntrySection.file_line_number +
                     DirectoryEntrySection.sequence_number - 1) * NCOL)

        for de in xrange((ParameterDataSection.file_line_number -
                          DirectoryEntrySection.file_line_number) // 2):
            position_fo()
            enum = int(fh.readline()[:8]); position_fo()
            if enum not in (116, 126, 128):
                print('IGES parser :: '
                      'Entity{} is not supported'.format(enum))
                DirectoryEntrySection.sequence_number += 2
                continue
            e = eval('Entity' + str(enum))()
            e.parse()

    def unparse(self):
        return directory_entry_lines + parameter_data_lines


class DirectoryEntrySection(DataSection):
    pass


class ParameterDataSection(DataSection):
    pass


class TerminateSection(Section):

    def parse(self, line):
        DirectoryEntrySection.file_line_number = (int(line[1:8]) +
                                                  int(line[9:16]))
        ParameterDataSection.file_line_number = \
            DirectoryEntrySection.file_line_number + int(line[17:24])

    def unparse(self):
        SGDP = (StartSection.sequence_number,
                GlobalSection.sequence_number,
                DirectoryEntrySection.sequence_number - 1,
                ParameterDataSection.sequence_number - 1)
        line = 'S{:7}G{:7}D{:7}P{:7}'.format(*SGDP).ljust(72)
        return fill_sequence_field([line], 'T')


class IGESFile(object):

    def __init__(self, fn):
        self.S = StartSection()
        self.G = GlobalSection({4: fn})
        self.D = DataSection()
        self.T = TerminateSection()

    def parse(self):
        global objs, NCOL, fh
        objs = []
        with open(self.G.params[4]) as fh:
            NCOL = len(fh.readline())
            fh.seek(-NCOL, 2)
            self.T.parse(fh.readline().strip())
            for s in self.S, self.G, self.D:
                s.parse()
        return objs

    def unparse(self, objs):
        global parameter_data_lines, directory_entry_lines
        parameter_data_lines, directory_entry_lines = [], []
        for o in objs:
            if isinstance(o, point.Point):
                Entity116(o).unparse()
            elif isinstance(o, curve.Curve):
                Entity126(o).unparse()
            elif isinstance(o, surface.Surface):
                if o.istrimmed:
                    Entity144(o).unparse()
                else:
                    Entity128(o).unparse()
            else:
                print('Unrecognized object {}, ignoring.'
                      .format(o))
        with open(self.G.params[4], 'w') as fh:
            for s in self.S, self.G, self.D, self.T:
                fh.writelines(s.unparse())
        return True


# UTILITIES


def now():
    return time.strftime('%y%m%d.%H%M%S')

def to_hollerith(string):
    return str(len(string)) + 'H' + string if string else ''

def to_string(hollerith):
    return hollerith.partition('H')[2]

def from_free_formatted_data(lines, last):
    params = ''
    for l in lines:
        i = l.find(';')
        if i == -1:
            params += l[:last]
        else:
            params += l[:i]
    return [float(p.replace('D', 'e')) for p in params.split(',')]

def to_free_formatted_data(params, last):
    for i, p in enumerate(params):
        if not isinstance(p, str):
            params[i] = repr(p).replace('e', 'D')
    si = 0
    lines = []
    while True:
        col = 0
        for i, p in enumerate(params[si:], start=si):
            col += len(p) + 1
            if col > last:
                break
        lines.append(','.join(params[si:i] + ['']))
        if i == len(params) - 1:
            last_param = params[-1] + ';'
            if col > last:
                lines.append(last_param)
            else:
                lines[-1] += last_param
            break
        si = i
    return lines

def fill_sequence_field(lines, S, start=1):
    return [l.ljust(72) + '{}{:7}\n'.format(S, i)
            for i, l in enumerate(lines, start)]
