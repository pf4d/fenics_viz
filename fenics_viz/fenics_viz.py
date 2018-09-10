from colored                 import fg, attr
from scipy.sparse            import spdiags
from ufl                     import indexed
from matplotlib              import colors, ticker
from matplotlib.ticker       import LogFormatter, ScalarFormatter
from matplotlib.colors       import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import numpy                     as np
import matplotlib.pyplot         as plt
import matplotlib.tri            as tri
import matplotlib                as mpl
import fenics                    as fe
import dolfin_adjoint            as da
import os
import sys


def print_text(text, color='white', atrb=0, cls=None):
  """
  Print text ``text`` from calling class ``cls`` to the screen.

  :param text: the text to print
  :param color: the color of the text to print
  :param atrb: attributes to send use by ``colored`` package
  :param cls: the calling class
  :type text: string
  :type color: string
  :type atrb: int
  :type cls: object
  """
  if cls is not None:
    color = cls.color()
  if fe.MPI.rank(fe.mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    print text


def print_min_max(u, title, color='97'):
  """
  Print the minimum and maximum values of ``u``, a Vector, Function, or array.

  :param u: the variable to print the min and max of
  :param title: the name of the function to print
  :param color: the color of printed text
  :type u: :class:`~fenics.GenericVector`, :class:`~numpy.ndarray`, :class:`~fenics.Function`, int, float, :class:`~fenics.Constant`
  :type title: string
  :type color: string
  """
  if isinstance(u, fe.GenericVector):
    uMin = MPI.min(fe.mpi_comm_world(), u.min())
    uMax = MPI.max(fe.mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, np.ndarray):
    if u.dtype != np.float64:
      u = u.astype(np.float64)
    uMin = fe.MPI.min(fe.mpi_comm_world(), u.min())
    uMax = fe.MPI.max(fe.mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, fe.Function):# \
    #   or isinstance(u, dolfin.functions.function.Function):
    uMin = MPI.min(fe.mpi_comm_world(), u.vector().min())
    uMax = MPI.max(fe.mpi_comm_world(), u.vector().max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, int) or isinstance(u, float):
    s    = title + ' : %.3e' % u
    print_text(s, color)
  elif isinstance(u, fe.Constant):
    s    = title + ' : %.3e' % u(0)
    print_text(s, color)
  else:
    er = title + ": print_min_max function requires a Vector, Function" \
         + ", array, int or float, not %s." % type(u)
    print_text(er, 'red', 1)


def plot_matrix(M, title, continuous=False, cmap='Greys'):
  """
  plot a matrix <M> with title <title> and a colorbar on subplot (axes object) 
  <ax>.
  """
  M    = M.array()
  m,n  = np.shape(M)
  M    = M.round(decimals=9)

  fig  = plt.figure()
  ax   = fig.add_subplot(111)
  cmap = mpl.cm.get_cmap(cmap)
  if not continuous:
    unq  = np.unique(M)
    num  = len(unq)
  im      = ax.imshow(M, cmap=cmap, interpolation='None')
  divider = make_axes_locatable(ax)
  cax     = divider.append_axes("right", size="5%", pad=0.05)
  dim     = r'$%i \times %i$ ' % (m,n)
  ax.set_title(dim + title)
  ax.axis('off')
  cb = plt.colorbar(im, cax=cax)
  if not continuous:
    cb.set_ticks(unq)
    cb.set_ticklabels(unq)


def mesh2triang(mesh):
  """
  create a matplotlib.tri.Triangulation object from a fenics mesh.
  """ 
  xy = mesh.coordinates()
  return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def plot_variable(u, name, direc, 
                  coords              = None,
                  cells               = None,
                  figsize             = (8,7),
                  cmap                = 'gist_yarg',
                  scale               = 'lin',
                  numLvls             = 10,
                  levels              = None,
                  levels_2            = None,
                  umin                = None,
                  umax                = None,
                  normalize_vec       = False,
                  plot_tp             = False,
                  tp_kwargs           = {'linestyle'      : '-',
                                         'lw'             : 1.0,
                                         'color'          : 'k',
                                         'alpha'          : 0.5},
                  show                = True,
                  hide_x_tick_labels  = False,
                  hide_y_tick_labels  = False,
                  vertical_y_labels   = False,
                  vertical_y_label    = False,
                  xlabel              = r'$x$',
                  ylabel              = r'$y$',
                  equal_axes          = True,
                  title               = '',
                  hide_axis           = False,
                  colorbar_loc        = 'right',
                  contour_type        = 'filled',
                  extend              = 'neither',
                  ext                 = '.pdf',
                  plot_quiver         = True,
                  quiver_skip         = 0,
                  quiver_kwargs       = {'pivot'          : 'middle',
                                         'color'          : 'k',
                                         'alpha'          : 0.8,
                                         'width'          : 0.004,
                                         'headwidth'      : 4.0, 
                                         'headlength'     : 4.0, 
                                         'headaxislength' : 4.0},
                  res                 = 150,
                  cb                  = True,
                  cb_format           = '%.1e'):
  """
  """
  vec = False  # assume initially that 'u' is not a vector

  # if 'u' is a NumPy array and the cell arrays and coordinates are supplied :
  if (type(u) == np.ndarray or type(u) == list or type(u) == tuple) \
    and len(coords) == 2:
    x = coords[0]
    y = coords[1]
    v = u
    
    if type(cells) == np.ndarray:
      t = cells
    else:
      t = []
   
    # if there are multiple components to 'u', it is a vector :
    if len(np.shape(u)) > len(np.shape(x)):
      vec = True  # used for plotting below
      v0  = u[0]
      v1  = u[1]
      # compute norm :
      v     = 0
      for k in u:
        v  += k**2
      v     = np.sqrt(v + 1e-16)

  # if 'u' is a NumPy array and cell/coordinate arrays are not supplied :
  if (type(u) == np.ndarray or type(u) == list or type(u) == tuple) \
    and coords is None:
    print_text(">>> numpy arrays require `coords' <<<", 'red', 1)
    sys.exit(1)

  # if 'u' is a FEniCS Function :
  elif    type(u) == indexed.Indexed \
       or type(u) == fe.dolfin.function.Function \
       or type(u) == fe.dolfin.functions.function.Function \
       or type(u) == da.function.Function:
    
    # if this is a scalar :
    if len(u.ufl_shape) == 0:
      mesh     = u.function_space().mesh()
      v        = u.compute_vertex_values(mesh)
    
    # otherwise it is a vector, so calculate the L^2 norm :
    else:
      vec = True  # used for plotting below
      # if the function is defined on a mixed space, deepcopy :
      # TODO: there is a way to do this without deepcopy
      if type(u[0]) == indexed.Indexed:
        out    = u.split(True)
      else:
        out    = u
      
      # extract the mesh :
      mesh = out[0].function_space().mesh()
    
      # compute norm :
      v     = 0
      for k in out:
        kv  = k.compute_vertex_values(mesh)
        v  += kv**2
      v     = np.sqrt(v + 1e-16)
      v0    = out[0].compute_vertex_values(mesh)
      v1    = out[1].compute_vertex_values(mesh)

    t    = mesh.cells()
    x    = mesh.coordinates()[:,0]
    y    = mesh.coordinates()[:,1]

  # if normalized vectors are desired :
  if vec and normalize_vec:
    v0 = v0 / v
    v1 = v1 / v

  if vec:  print_text("::: plotting vector variable :::", 'red')
  else:    print_text("::: plotting scalar variable :::", 'red')

  #=============================================================================
  # plotting :
  if umin != None and levels is None:
    vmin = umin
  elif levels is not None:
    vmin = levels.min()
  else:
    vmin = v.min()

  if umax != None and levels is None:
    vmax = umax
  elif levels is not None:
    vmax = levels.max()
  else:
    vmax = v.max()
  
  # set the extended colormap :  
  cmap = plt.get_cmap(cmap)
  
  # countour levels :
  if scale == 'log':
    if levels is None:
      levels    = np.logspace(np.log10(vmin), np.log10(vmax), numLvls)
    v[v < vmin] = vmin + 2e-16
    v[v > vmax] = vmax - 2e-16
    formatter   = LogFormatter(10, labelOnlyBase=False)
    norm        = colors.LogNorm()
  
  # countour levels :
  elif scale == 'sym_log':
    if levels is None:
      levels  = np.linspace(vmin, vmax, numLvls)
    v[v < vmin] = vmin + 2e-16
    v[v > vmax] = vmax - 2e-16
    formatter   = LogFormatter(e, labelOnlyBase=False)
    norm        = colors.SymLogNorm(vmin=vmin, vmax=vmax,
                                    linscale=0.001, linthresh=0.001)
  
  elif scale == 'lin':
    if levels is None:
      levels  = np.linspace(vmin, vmax, numLvls)
    norm = colors.BoundaryNorm(levels, cmap.N)
  
  elif scale == 'bool':
    v[v < 0.0] = 0.0
    levels  = [0, 1, 2]
    norm    = colors.BoundaryNorm(levels, cmap.N)

  fig = plt.figure(figsize=figsize)
  ax  = fig.add_subplot(111)
  ax.set_xlabel(xlabel)
  yl  = ax.set_ylabel(ylabel)
  if vertical_y_label:
    yl.set_rotation(0)
  if hide_x_tick_labels:
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none')
  if hide_y_tick_labels:
    ax.yaxis.set_ticks_position('none')
    ax.set_yticklabels([])
  if vertical_y_labels:
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=-90,
             verticalalignment='center')
    #for tick in ax.get_yticklabels():
    #  tick.set_rotation(-90)
  if hide_axis:
    ax.axis('off')
  if equal_axes:
    ax.axis('equal')

  # filled contours :
  if contour_type == 'filled':

    # if the number of degrees equal the number of cells, a DG space is used :
    if len(v) == len(t):
      cs = ax.tripcolor(mesh2triang(mesh), v, shading='flat',
                        cmap=cmap, norm=norm)

    # otherwise, a CG space is used :
    else:
      if len(np.shape(x)) > 1:  cs_ftn = ax.contourf
      else:                     cs_ftn = ax.tricontourf
      
      if   len(v) != len(t) and scale != 'log':
        cs = cs_ftn(x, y, v, triangles=t, levels=levels, 
                    cmap=cmap, norm=norm, extend=extend)
      elif len(v) != len(t)  and scale == 'log':
        cs = cs_ftn(x, y, v, triangles=t, levels=levels, 
                    cmap=cmap, norm=norm)

  # non-filled contours :
  elif contour_type == 'lines':
    
    if len(np.shape(x)) > 1:  cs_ftn = ax.contour
    else:                     cs_ftn = ax.tricontour

    cs = cs_ftn(x, y, v, triangles=t, linewidths=1.0,
                levels=levels, colors='k') 

    for line in cs.collections:
      if line.get_linestyle() != [(None, None)]:
        #line.set_linestyle([(None, None)])
        #line.set_color('red')
        # reduce the line size only if 'main' contours are needed : 
        if levels_2 is not None: line.set_linewidth(1.5)
    if levels_2 is not None:
      cs2 = ax.tricontour(x, y, v, triangles=t, levels=levels_2, colors='0.30') 
      for line in cs2.collections:
        if line.get_linestyle() != [(None, None)]:
          line.set_linestyle([(None, None)])
          line.set_color('#c1000e')
          line.set_linewidth(0.5)
    ax.clabel(cs, inline=1, fmt=cb_format)
  
  # plot triangles, if desired :
  if plot_tp == True:
    tp = ax.triplot(x, y, t, **tp_kwargs)

  # plot vectors, if desired :
  if vec and plot_quiver:
    # reduce the size of the dataset :
    if quiver_skip > 0:
     sav     = range(0, len(x), quiver_skip)
     v0_quiv = v0[sav]
     v1_quiv = v1[sav]
     x_quiv  = x[sav]
     y_quiv  = y[sav]
    else:
     v0_quiv = v0
     v1_quiv = v1
     x_quiv  = x
     y_quiv  = y
    q = ax.quiver(x_quiv, y_quiv, v0_quiv, v1_quiv, **quiver_kwargs)
  
  # this enforces equal axes no matter what (yeah, a hack) : 
  divider = make_axes_locatable(ax)

  # include colorbar :
  if cb and scale != 'bool' and contour_type != 'lines':
    cax  = divider.append_axes("right", "5%", pad="3%")
    cbar = fig.colorbar(cs, cax=cax, 
                        ticks=levels, format=cb_format) 
  
  ax.set_xlim([x.min(), x.max()])
  ax.set_ylim([y.min(), y.max()])
  #if title is None and cb is False:
  #  plt.tight_layout()#rect=[0,0,1,0.95])
  #else:
  #  plt.tight_layout(rect=[0,0,1,0.95])
  plt.tight_layout()
  
  #mpl.rcParams['axes.titlesize'] = 'small'
  #tit = plt.title(title)

  # title :
  if title is not None:
    tit = plt.title(title)
  #tit.set_fontsize(40)
  
  # create the output directory : 
  d     = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)

  # always save the figure to a file :
  plt.savefig(direc + name + ext, res=res)

  # show the figure too, if desired : 
  if show: plt.show()
  else:    plt.close(fig)



