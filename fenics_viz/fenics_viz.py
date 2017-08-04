from colored                 import fg, attr
from pylab                   import *
from fenics                  import *
from scipy.sparse            import spdiags
from ufl                     import indexed
from matplotlib              import colors, ticker
from matplotlib.ticker       import LogFormatter, ScalarFormatter
from matplotlib.colors       import from_levels_and_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
import matplotlib.pyplot         as plt
import matplotlib.tri            as tri
import os


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
  if MPI.rank(mpi_comm_world())==0:
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
  if isinstance(u, GenericVector):
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, ndarray):
    if u.dtype != float64:
      u = u.astype(float64)
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, Function):# \
    #   or isinstance(u, dolfin.functions.function.Function):
    uMin = MPI.min(mpi_comm_world(), u.vector().min())
    uMax = MPI.max(mpi_comm_world(), u.vector().max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, int) or isinstance(u, float):
    s    = title + ' : %.3e' % u
    print_text(s, color)
  elif isinstance(u, Constant):
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

  fig  = figure()
  ax   = fig.add_subplot(111)
  cmap = cm.get_cmap(cmap)
  if not continuous:
    unq  = unique(M)
    num  = len(unq)
  im      = ax.imshow(M, cmap=cmap, interpolation='None')
  divider = make_axes_locatable(ax)
  cax     = divider.append_axes("right", size="5%", pad=0.05)
  dim     = r'$%i \times %i$ ' % (m,n)
  ax.set_title(dim + title)
  ax.axis('off')
  cb = colorbar(im, cax=cax)
  if not continuous:
    cb.set_ticks(unq)
    cb.set_ticklabels(unq)

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def plot_variable(u, name, direc, 
                  figsize             = (8,7),
                  cmap                = 'gist_yarg',
                  scale               = 'lin',
                  numLvls             = 10,
                  levels              = None,
                  levels_2            = None,
                  umin                = None,
                  umax                = None,
                  normalize_vec       = False,
                  tp                  = False,
                  tpAlpha             = 0.5,
                  show                = True,
                  hide_ax_tick_labels = False,
                  xlabel              = r'$x$',
                  ylabel              = r'$y$',
                  equal_axes          = True,
                  title               = '',
                  hide_axis           = False,
                  colorbar_loc        = 'right',
                  contour_type        = 'filled',
                  extend              = 'neither',
                  ext                 = '.pdf',
                  quiver_kwargs       = None,
                  res                 = 150,
                  cb                  = True,
                  cb_format           = '%.1e'):
  """
  """
  vec  = False
  if len(u.ufl_shape) == 1:
    if type(u[0]) == indexed.Indexed:
      out    = u.split(True)
    else:
      out    = u
    vec      = True
    v        = 0
    mesh     = out[0].function_space().mesh()
    for k in out:
      kv  = k.compute_vertex_values(mesh)
      v  += kv**2
    v = np.sqrt(v + 1e-16)
    v0       = out[0].compute_vertex_values(mesh)
    v1       = out[1].compute_vertex_values(mesh)
    if normalize_vec:
      v2_mag   = np.sqrt(v0**2 + v1**2 + 1e-16)
      v0       = v0 / v2_mag
      v1       = v1 / v2_mag
  elif len(u.ufl_shape) == 0:
    mesh     = u.function_space().mesh()
    if u.vector().size() == mesh.num_cells():
      v       = u.vector().array()
    else:
      v       = u.compute_vertex_values(mesh)
  x    = mesh.coordinates()[:,0]
  y    = mesh.coordinates()[:,1]
  t    = mesh.cells()
  
  d    = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)
 
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
  cmap = get_cmap(cmap)
  
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
  ax.set_ylabel(ylabel)
  if hide_ax_tick_labels:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
  if hide_axis:
    ax.axis('off')
  if equal_axes:
    ax.axis('equal')

  if contour_type == 'filled':
    if u.vector().size() == mesh.num_cells():
      cs = ax.tripcolor(mesh2triang(mesh), v, shading='flat',
                        cmap=cmap, norm=norm)
    elif u.vector().size() != mesh.num_cells() and scale != 'log':
      cs = ax.tricontourf(x, y, t, v, levels=levels, 
                          cmap=cmap, norm=norm, extend=extend)
    elif u.vector().size() != mesh.num_cells() and scale == 'log':
      cs = ax.tricontourf(x, y, t, v, levels=levels, 
                          cmap=cmap, norm=norm)
  elif contour_type == 'lines':
    cs = ax.tricontour(x, y, t, v, linewidths=2.0,
                       levels=levels, colors='k') 
    for line in cs.collections:
      if line.get_linestyle() != [(None, None)]:
        line.set_linestyle([(None, None)])
        line.set_color('red')
        line.set_linewidth(1.5)
    if levels_2 is not None:
      cs2 = ax.tricontour(x, y, t, v, levels=levels_2, colors='0.30') 
      for line in cs2.collections:
        if line.get_linestyle() != [(None, None)]:
          line.set_linestyle([(None, None)])
          line.set_color('#c1000e')
          line.set_linewidth(0.5)
    ax.clabel(cs, inline=1, colors='k', fmt='%i')


  if vec:
    if quiver_kwargs is not None:
      q  = ax.quiver(x, y, v0, v1, **quiver_kwargs)
    else:
      q  = ax.quiver(x, y, v0, v1)
  
  # plot triangles :
  if tp == True:
    tp = ax.triplot(x, y, t, 'k-', lw=0.2, alpha=tpAlpha)
  
  # this enforces equal axes no matter what (yeah, a hack) : 
  divider = make_axes_locatable(ax)

  # include colorbar :
  if cb and scale != 'bool' and contour_type != 'lines':
    cax  = divider.append_axes("right", "5%", pad="3%")
    cbar = fig.colorbar(cs, cax=cax, 
                        ticks=levels, format=cb_format) 
  
  ax.set_xlim([x.min(), x.max()])
  ax.set_ylim([y.min(), y.max()])
  plt.tight_layout(rect=[0,0,1,0.95])
  
  #mpl.rcParams['axes.titlesize'] = 'small'
  #tit = plt.title(title)

  # title :
  tit = plt.title(title)
  #tit.set_fontsize(40)

  d     = os.path.dirname(direc)
  if not os.path.exists(d):
    os.makedirs(d)
  plt.savefig(direc + name + ext, res=res)
  if show:
    plt.show()
  plt.close(fig)

def plot_new(obj):
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')



