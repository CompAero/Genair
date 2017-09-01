#!/usr/bin/env python

import os
os.chdir('./play') # sandbox

from traitlets.config import Config
config = Config()

# InteractiveShell configuration
config.InteractiveShell.banner2 = 'Welcome to Genair!\n'

# InteractiveShellApp configuration
config.InteractiveShellApp.exec_lines = [

        'import nurbs',
        'from nurbs.point   import Point',
        'from nurbs.curve   import ControlPolygon, Curve',
        'from nurbs.surface import ControlNet, Surface',
        'from nurbs.volume  import ControlVolume, Volume',

        'import ffd',
        'from ffd.ffd       import FFDVolume',

        'import opti',
        'from opti.grid     import Grid',

        'import geom',
        'from geom.aircraft import Aircraft',
        'from geom.airfoil  import Airfoil',
        'from geom.fuselage import Fuselage',
        'from geom.io       import save, load',
        'from geom.misc     import Nacelle, Cabin',
        'from geom.wing     import Wing, WingMerger, HalfWingMerger, \
                                   BSplineFunctionCreator1, \
                                   BSplineFunctionCreator2, \
                                   BSplineFunctionCreator3',

        'from plot.figure   import Figure as draw',

        'import numpy as np',
        'np.set_printoptions(linewidth=79)'

]

# IPCompleter configuration
config.IPCompleter.limit_to__all__ = True # deprecated as of version 5.0

# get rid of the deprecated warning
import warnings
warnings.filterwarnings('ignore')

import IPython
IPython.start_ipython(config=config)
