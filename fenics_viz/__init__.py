__version__    = '1.0'
__author__     = 'Evan Cummings'
__license__    = 'LGPL-3'
__maintainer__ = 'Evan Cummings'
__email__      = 'cummings.evan@gmail.com'

__all__ = []

import pkgutil
import inspect
import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['font.family']          = 'serif'
#mpl.rcParams['legend.fontsize']      = 'medium'
mpl.rcParams['text.usetex']          = True
#mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']
#mpl.rcParams['contour.negative_linestyle']   = 'solid'

# conditional fix (issue #107) :
for loader, name, is_pkg in pkgutil.walk_packages(__path__):
  module = loader.find_module(name).load_module(name)
  for name, value in inspect.getmembers(module):
    if name.startswith('__'):
      continue

    globals()[name] = value
    __all__.append(name)

from fenics_viz import *
