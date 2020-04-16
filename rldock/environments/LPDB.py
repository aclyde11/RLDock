import inspect
import copy
from enum import Enum
from typing import List
import numpy as np
import scipy
import scipy.spatial
import scipy.spatial.transform


class JUSTIF(Enum):
    LEFT = 0
    RIGHT = 1


def minN(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return min(x, y)


def maxN(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)


'''
For ligands two important blocks. HETATM block which has coordinates and types, connect block.

Connect block save and pass through.
'''


class PDBTransformer:
    def __init__(self, pdb_file):
        pdb_file = list(map(lambda x: x.strip(), open(pdb_file, 'r').readlines()))
        connect_block = "\n".join(list(filter(lambda x: "CONNECT" in x, pdb_file)))

        self.get_atom_list = []


class PDBParseAttribute:
    justif: JUSTIF

    def __init__(self, start, end, justif, type, fmt=None):
        self.start = start  # adjust pdb number
        self.end = end
        self.justif = justif
        self.type = type
        self.fmt = fmt
        self.p = None

    def read(self, r):
        return self.type(r[self.start:(self.end + 1)])

    def write(self, p):
        if self.fmt is not None:
            p = str(self.fmt.format(p))
        p = str(p)
        return p.ljust(self.end - self.start + 1) if self.justif == JUSTIF.LEFT else p.rjust(
            self.end - self.start + 1)


class Parser:
    def __init__(self, i):
        self.__item_to_fill = i
        pass

    def parse(self, row):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if
                      not (a[0].startswith('__') or a[0].endswith('__') or a[0].startswith('_'))]

        for var, name in attributes:
            setattr(self.__item_to_fill, var, name.read(row))

    def write(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if
                      not (a[0].startswith('__') or a[0].endswith('__') or a[0].startswith('_'))]

        ## sort correct  name.start
        ids = np.argsort(list(map(lambda x: x[1].start, attributes)))
        attributes = [attributes[i] for i in ids]

        i_write = 0
        s = ""
        for var, name in attributes:

            while name.start > i_write:
                s += " "
                i_write += 1
            t = name.write(getattr(self.__item_to_fill, var))
            s += t
            i_write += len(t)
        return s


class LigandAtomParser(Parser):

    def __init__(self, i):
        super(LigandAtomParser, self).__init__(i)
        self.type = PDBParseAttribute(0, 5, JUSTIF.LEFT, str)
        self.serial_num = PDBParseAttribute(6, 10, JUSTIF.RIGHT, int)
        self.atom_name = PDBParseAttribute(12, 15, JUSTIF.LEFT, str)
        self.res_name = PDBParseAttribute(17, 19, JUSTIF.RIGHT, str)
        self.chain_id = PDBParseAttribute(21, 21, JUSTIF.RIGHT, str)
        self.res_seq_n = PDBParseAttribute(23, 25, JUSTIF.RIGHT, int)
        self.x_ortho_a = PDBParseAttribute(30, 37, JUSTIF.RIGHT, float, fmt="{:8.3f}")
        self.y_ortho_a = PDBParseAttribute(38, 45, JUSTIF.RIGHT, float, fmt="{:8.3f}")
        self.z_ortho_a = PDBParseAttribute(46, 53, JUSTIF.RIGHT, float, fmt="{:8.3f}")
        self.occupancy = PDBParseAttribute(56, 59, JUSTIF.RIGHT, float, fmt="{:1.2f}")
        self.temp_f = PDBParseAttribute(61, 65, JUSTIF.RIGHT, float, fmt="{:2.2f}")
        self.seg_id = PDBParseAttribute(72, 73, JUSTIF.LEFT, str)
        self.elem_sym = PDBParseAttribute(74, 77, JUSTIF.RIGHT, str)


class ConnectRowParser(Parser):

    def __init__(self, i):
        super(ConnectRowParser, self).__init__(i)
        self.type = PDBParseAttribute(1, 6, JUSTIF.LEFT, str)
        self.serial_num = PDBParseAttribute(7, 11, JUSTIF.RIGHT, int)
        self.bonded0 = PDBParseAttribute(12, 16, JUSTIF.RIGHT, int)
        self.bonded1 = PDBParseAttribute(17, 21, JUSTIF.RIGHT, int)
        self.bonded2 = PDBParseAttribute(22, 26, JUSTIF.RIGHT, int)
        self.bonded3 = PDBParseAttribute(27, 31, JUSTIF.RIGHT, int)


class ConnectRow:
    type: str
    serial_num: int
    bonded0: int
    bonded1: int
    bonded2: int
    bonded3: int

    __slots__ = ['type', 'serial_num', 'bonded0', 'bonded2', 'bonded1', 'bonded3']

    def __init__(self):
        self.type = 'CONECT'  # 1-6 left c
        self.serial_num = 0
        self.bonded0 = 0
        self.bonded1 = 0
        self.bonded2 = 0
        self.bonded3 = 0

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if
                      not (a[0].startswith('__') or a[0].endswith('__') or a[0].startswith('_'))]
        s = ""
        s += "----\n"
        for var, name in attributes:
            s += str(var) + "\t" + str(name) + "\n"
        s += '----\n'
        return s

    @classmethod
    def fromPDBRow(cls, line):
        atom = cls()
        ConnectRowParser(atom).parse(line)
        return atom


class LigandAtom:
    type: str
    serial_num: int
    atom_name: str
    res_name: str
    chain_id: str
    res_seq_n: int
    x_ortho_a: float
    y_ortho_a: float
    z_ortho_a: float
    occupancy: float
    temp_f: float
    seg_id: str
    elem_sym: str
    __slots__ = ['type', 'serial_num', 'atom_name', 'res_name', 'chain_id', 'res_seq_n',
                 'x_ortho_a', 'y_ortho_a', 'z_ortho_a', 'occupancy', 'temp_f',
                 'seg_id', 'elem_sym']

    def __init__(self):
        self.type = 'HETATM'  # 1-6 left c
        self.serial_num = 0  # 7-11  right i
        self.atom_name = ""  # 13-16 left  c
        self.res_name = ""  # 18-20 right c
        self.chain_id = ""  # 22          c
        self.res_seq_n = 0  # 23-26 right i
        self.x_ortho_a = 0.0  # 31-38 right real(8.3)
        self.y_ortho_a = 0.0  # 39-46 right real(8.3)
        self.z_ortho_a = 0.0  # 47-54 right real(8.3)
        self.occupancy = 0.0  # 55-60 right real(6.2)
        self.temp_f = 0.0  # 61-66 right real(6.2)
        self.seg_id = ""  # 73-76 left  c
        self.elem_sym = ""  # 77-78 right c

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if
                      not (a[0].startswith('__') or a[0].endswith('__') or a[0].startswith('_'))]
        s = ""
        s += "----\n"
        for var, name in attributes:
            s += str(var) + "\t" + str(name) + "\n"
        s += '----\n'
        return s

    def get_coords(self):
        return np.array([self.x_ortho_a, self.y_ortho_a, self.z_ortho_a], dtype=np.float32)

    def set_coords(self, x):
        self.x_ortho_a, self.y_ortho_a, self.z_ortho_a = x.flatten().ravel()

    @classmethod
    def toPDBRow(cls, ob):
        return LigandAtomParser(ob).write()

    @classmethod
    def fromPDBRow(cls, line):
        atom = cls()
        LigandAtomParser(atom).parse(line)
        return atom


class LigandPDB:
    hetatoms: List[LigandAtom]

    # connect: List[ConnectRow]

    def __init__(self):
        self.hetatoms = []
        self.connect = []

    # returns copy!!!
    def translate(self, x, y, z):
        pdb_c = copy.deepcopy(self)

        for atom in pdb_c.hetatoms:
            atom.x_ortho_a += x
            atom.y_ortho_a += y
            atom.z_ortho_a += z

        return pdb_c

    def get_center(self):
        max_x, max_y, max_z = None, None, None
        min_x, min_y, min_z = None, None, None
        for atom in self.hetatoms:
            max_x = maxN(max_x, atom.x_ortho_a)
            min_x = minN(min_x, atom.x_ortho_a)
            max_y = maxN(max_y, atom.y_ortho_a)
            min_y = minN(min_y, atom.y_ortho_a)
            max_z = maxN(max_z, atom.z_ortho_a)
            min_z = minN(min_z, atom.z_ortho_a)

        return (max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2

    def __translate(self, x, y, z):
        pdb_c = copy.deepcopy(self)

        for atom in pdb_c.hetatoms:
            atom.x_ortho_a += x
            atom.y_ortho_a += y
            atom.z_ortho_a += z

        return pdb_c

    # returns copy!!!
    def rotate(self, theta_x, theta_y, theta_z):
        pdb_c = copy.deepcopy(self)
        c = np.array(pdb_c.get_center()).flatten()
        rot_mat = scipy.spatial.transform.Rotation.from_euler('xyz', [theta_x, theta_y, theta_z])
        for atom in pdb_c.hetatoms:
            vec = atom.get_coords()
            vec -= c
            vec = rot_mat.apply(vec)
            vec += c
            atom.set_coords(vec)
        return pdb_c

    def rotateM(self, rot_mat):
        pdb_c = copy.deepcopy(self)
        c = np.array(pdb_c.get_center()).flatten()
        for atom in pdb_c.hetatoms:
            vec = atom.get_coords()
            vec -= c
            vec = np.dot(rot_mat, vec)
            vec += c
            atom.set_coords(vec)
        return pdb_c

    @classmethod
    def parse(cls, fname):
        pdb = LigandPDB()
        with open(fname, 'r') as f:
            pdb_file = list(map(lambda x: x.strip(), f.readlines()))

        for line in pdb_file:
            if "HETATM" in line:
                pdb.hetatoms.append(LigandAtom.fromPDBRow(line))
            elif "CONECT" in line:
                # pdb.connect.append(ConnectRow.fromPDBRow(line))
                pdb.connect.append(line)
        return pdb

    def dump_header(self):
        s = str(len(self.hetatoms)) + "\n"
        s += "\n"

        return s

    def dump_coords(self):
        rows = []
        for atom in self.hetatoms:
            rows.append(" ".join([str(atom.atom_name).strip(),
                                  "{:8.3f}".format(atom.x_ortho_a).strip(),
                                  "{:8.3f}".format(atom.y_ortho_a).strip(),
                                  "{:8.3f}".format(atom.z_ortho_a).strip()]))
        return "\n".join(rows) + "\n"

    def toPDB(self):
        lines = []
        for atom in self.hetatoms:
            lines.append(LigandAtom.toPDBRow(atom))
        return "\n".join(lines + self.connect)
