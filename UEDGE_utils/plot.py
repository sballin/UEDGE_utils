import os
import datetime
import glob
import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from uedge import bbb, com, api, grd
from uedge import __version__ as uedgeVersion
from UEDGE_utils import analysis, sparc


def zoom(factor):
    '''
    Zoom in/out of plot view, adjusting x and y axes by the same factor.
    '''
    x1, x2 = plt.gca().get_xlim()
    plt.xlim([(x1+x2)/2-factor*(x2-x1)/2, (x1+x2)/2+factor*(x2-x1)/2])
    y1, y2 = plt.gca().get_ylim()
    plt.ylim([(y1+y2)/2-factor*(y2-y1)/2, (y1+y2)/2+factor*(y2-y1)/2])


def getPatches():
    '''
    Create cell patches for 2D plotting.
    '''
    patches = []
    for iy in np.arange(0,com.ny+2):
        for ix in np.arange(0,com.nx+2):
            rcol=com.rm[ix,iy,[1,2,4,3]]
            zcol=com.zm[ix,iy,[1,2,4,3]]
            patches.append(np.column_stack((rcol,zcol)))
    return patches
    

def plotvar(var, title='', label=None, iso=True, rzlabels=True, stats=True, message=None,
            orientation='vertical', vmin=None, vmax=None, minratio=None, cmap=plt.cm.viridis, log=False,
            patches=None, show=True, sym=False, showGuards=False, colorbar=True, extend=None, norm=None,linscale=1):
    '''
    Plot a quantity on the grid in 2D. 
    
    Args:
        patches: supplying previously computed patches lowers execution time
        show: set this to False if you are calling this method to create a subplot
        minratio: set vmin to this fraction of vmax (useful for log plots with large range)
        sym: vmax=-vmin
    '''
    plt.rcParams['axes.axisbelow'] = True
    
    if not patches:
        patches = getPatches()

    # reorder value in 1-D array
    vals = var.T.flatten()
    
    # Set vmin and vmax disregarding guard cells
    if not vmax:
        vmax = np.max(analysis.nonGuard(var))
    if not vmin:
        vmin = np.min(analysis.nonGuard(var))
        
    if show:
        rextent = np.max(com.rm)-np.min(com.rm)
        zextent = np.max(com.zm)-np.min(com.zm)
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))
    else:
        ax = plt.gca()
        
    if sym:
        maxval = np.max(np.abs([vmax, vmin]))
        vmax = maxval
        vmin = -maxval
        cmap = plt.cm.bwr
        plt.gca().set_facecolor('lightgray')
    else:
        plt.gca().set_facecolor('gray')
        
    _extend = 'neither' # minratio related
        
    # Need to make a copy for set_bad
    cmap = copy.copy(cmap)
    
    if not np.any(var > 0):
        log = False

    if log:
        cmap.set_bad((1,0,0,1))
        if vmin > 0:
            if minratio:
                vmin = vmax/minratio
                _extend = 'min'
            _norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            if minratio:
                # linscale=np.log10(minratio)/linscale
                _norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=vmax/minratio, linscale=linscale, base=10)
            else:
                _norm = matplotlib.colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=(vmax-vmin)/1000, linscale=linscale, base=10)
    else:
        _norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if norm:
        _norm = norm
    p = PolyCollection(patches, array=np.array(vals), cmap=cmap, norm=_norm)
    
    if (vmin > np.min(analysis.nonGuard(var))) and (vmax < np.max(analysis.nonGuard(var))):
        _extend = 'both'
    elif vmin > np.min(analysis.nonGuard(var)):
        _extend = 'min'
    elif vmax < np.max(analysis.nonGuard(var)):
        _extend = 'max'
    
    if extend:
       _extend = extend
    
    if showGuards:
        plt.scatter(com.rm[0,:,0], com.zm[0,:,0], c=var[0,:], cmap=cmap)
        plt.scatter(com.rm[com.nx+1,:,0], com.zm[com.nx+1,:,0], c=var[com.nx+1,:], cmap=cmap)
        plt.scatter(com.rm[:,0,0], com.zm[:,0,0], c=var[:,0], cmap=cmap)
        plt.scatter(com.rm[:,com.ny+1,0], com.zm[:,com.ny+1,0], c=var[:,com.ny+1], cmap=cmap)
        if 'dnbot' in com.geometry[0].decode('UTF-8'):
            plt.scatter(com.rm[bbb.ixmp-1,:,0], com.zm[bbb.ixmp-1,:,0], c=var[bbb.ixmp-1,:], cmap=cmap)
            plt.scatter(com.rm[bbb.ixmp,:,0], com.zm[bbb.ixmp,:,0], c=var[bbb.ixmp,:], cmap=cmap)

    ax.grid(False)
    plt.title(title)
    if rzlabels:
        plt.xlabel(r'$R$ [m]')
        plt.ylabel(r'$Z$ [m]')
    else:
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        
    ax.add_collection(p)
    ax.autoscale_view()    

    if colorbar:
        if sym and log and minratio:
            # maxpow = int(np.log10(vmax))
            # minpow = int(np.log10(vmax/minratio)+0.5)
            # print(minpow, maxpow)
            # ticks = [-10**p for p in range(maxpow, minpow-1, -1)]
            # ticklabels = ['$-10^{%d}$' % p for p in range(maxpow, minpow-1, -1)]
            # ticks.append(0)
            # ticklabels.append('0')
            # ticks.extend([10**p for p in range(minpow, maxpow+1)])
            # ticklabels.extend(['$10^{%d}$' % p for p in range(minpow, maxpow+1)])
            
            # ticks = np.arange(1,10)
            # cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation, ticks=ticks)
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(ticklabels)
            # minticks = []
            # for p in range(maxpow, minpow-1, -1):
            #     minticks.extend([i*10**p for i in range(2, 10)])
            #cbar.set_ticks(minticks)
            cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation)
        else:
            cbar = plt.colorbar(p, label=label, extend=_extend, orientation=orientation)
    
    if iso:
        plt.axis('equal')  # regular aspect-ratio
        
    if stats:
        text =  '      max %.2g\n' % np.max(analysis.nonGuard(var))
        text += '      min %.2g\n' % np.min(analysis.nonGuard(var))
        text += ' min(abs) %.2g\n' % np.min(np.abs(analysis.nonGuard(var)))
        text += '     mean %.2g\n' % np.mean(analysis.nonGuard(var))
        text += 'mean(abs) %.2g' % np.mean(np.abs(analysis.nonGuard(var)))
    if message:
        text = message
    if stats or message:        
        plt.text(0.01, 0.01, text, fontsize=4, color='black', family='monospace',
                 horizontalalignment='left', verticalalignment='bottom', 
                 transform=plt.gca().transAxes)

    if show:
        plt.tight_layout()
        plt.show(block=False)
    
    
def plotDiffs(h5file):
    '''
    Plot differences between current variables and old ones.
    '''
    with h5py.File(h5file, 'r') as hf:
        hfb = hf.get('bbb')
        teold = np.array(hfb.get('tes'))
        tiold = np.array(hfb.get('tis'))
        niold = np.array(hfb.get('nis'))
    plt.figure(figsize=(14,5))
    plt.subplot(141)
    plotvar((bbb.ti-tiold)/tiold,sym=True,show=False,title=r'$(\Delta T_i)/T_i$')
    plt.subplot(142)
    plotvar((bbb.te-teold)/teold,sym=True,show=False,title=r'$(\Delta T_e)/T_e$')
    plt.subplot(143)
    plotvar((bbb.ni[:,:,0]-niold[:,:,0])/niold[:,:,0],sym=True,show=False,title=r'$(\Delta n_i)/n_i$')
    plt.subplot(144)
    plotvar((bbb.ni[:,:,1]-niold[:,:,1])/niold[:,:,1],sym=True,show=False,title=r'$(\Delta n_n)/n_n$')
    plt.tight_layout()
    plt.show()
    
    
def plotSources():
    kw = {'sym': True, 'show': False}
    rows = 2
    cols = 4
    plt.figure(figsize=(14,5))
    plt.subplot(rows, cols, 1)
    plotvar(bbb.fnix[:,:,0]/com.vol,title=r'fnix/vol', **kw)
    plt.subplot(rows, cols, 2)
    plotvar(bbb.fngx[:,:,0]/com.vol,title=r'fngx/vol', **kw)
    plt.subplot(rows, cols, 3)
    plotvar(bbb.fniy[:,:,0]/com.vol,title=r'fniy/vol', **kw)
    plt.subplot(rows, cols, 4)
    plotvar(bbb.fngy[:,:,0]/com.vol,title=r'fngy/vol', **kw)
    plt.subplot(rows, cols, 5)
    plotvar(bbb.psor[:,:,0]/com.vol,title=r'psor/vol', **kw)
    plt.subplot(rows, cols, 6)
    plotvar(bbb.psorrg[:,:,0]/com.vol,title=r'psorrg/vol', **kw)
    plt.subplot(rows, cols, 7)
    plotvar((bbb.psor[:,:,0]-bbb.psorrg[:,:,0]-bbb.fnix[:,:,0]-bbb.fniy[:,:,0])/com.vol,title=r'dni/dt', **kw)
    plt.subplot(rows, cols, 8)
    plotvar((-bbb.psor[:,:,0]+bbb.psorrg[:,:,0]-bbb.fngx[:,:,0]-bbb.fngy[:,:,0])/com.vol,title=r'dng/dt', **kw)
    plt.tight_layout()
    plt.show()
    
    
def readmesh(meshfile):
    """Return rm and zm of meshfile.
    
    Args:
        meshfile: (str) name of mesh file
        
    Source: https://github.com/LLNL/UEDGE/blob/master/pyscripts/uereadgrid.py
    """
    fh = open(meshfile, 'r')

    lns = fh.readlines()

    # Read the header information including metadata and grid shape
    ln1 = lns.pop(0).split()

    xxr = ln1[0]
    yyr = ln1[1]

    xsp1 = ln1[2]
    xsp2 = ln1[3]
    ysp = ln1[4]

    lns.pop(0)

    # Reshape the grid data to be linear
    data = np.zeros( (lns.__len__() * 3), np.float )
    print(data.shape)
    for i in range(0, lns.__len__()-1):
        ll = lns[i].split()
        print(ll)
        data[3*i  ] = float( ll[0].replace('D','E') )
        data[3*i+1] = float( ll[1].replace('D','E') )
        data[3*i+2] = float( ll[2].replace('D','E') )

    rml = 0
    rmh = (xxr+2) * (yyr+2) * 5
    rm = data[rml:rmh].reshape( (xxr+2, yyr+2, 5) )

    zml = rmh
    zmh = rmh + (xxr+2) * (yyr+2) * 5
    zm = data[zml:zmh].reshape( (xxr+2, yyr+2, 5) )
    return rm, zm


def plotmesh(meshfile=None, iso=True, xlim=None, ylim=None, wPlates=False, show=True, color=None, linewidth=0.2, fig=None, showBad=True, outlineOnly=False, zorder=1):
    """Plot current UEDGE mesh.
    
    Args:
        meshfile: (str) name of file containing mesh (default: {None} shows current UEDGE grid)
        iso: (bool) use equal x and y scale (default: {True})
        xlim: ([xmin, xmax]) x plotting limits (default: {None})
        ylim: ([ymin, ymax]) y plotting limits (default: {None})
        wPlates: (bool) show target plates as defined by user in UEDGE (default: {False})
        show: (bool) True = display to user immediately (default: {True})
        color: (string) desired color of grid (default: {None})
        linewidth: (float) desired line width of grid (default: {0.2})
        fig: (matplotlib figure) designate figure to plot on (default: {None})
        showBad: (bool) display twisted and overlapping cells (default: {True})
        outlineOnly: (bool) plot grid boundaries only
    """
    if meshfile:
        pass
    else:    
        rm = com.rm
        zm = com.zm
    
    if fig:
        ax = plt.gca()
    else:
        fig, ax = plt.subplots(1, figsize=(4.8, 6.4))
    
    if outlineOnly:
        lines = []
        lines.extend([list(zip(rm[0,iy,[1,2,4,3,1]],zm[0,iy,[1,2,4,3,1]])) 
                      for iy in np.arange(0,com.ny+2)])
        lines.extend([list(zip(rm[com.nx+1,iy,[1,2,4,3,1]],zm[com.nx+1,iy,[1,2,4,3,1]])) 
                      for iy in np.arange(0,com.ny+2)])
        lines.extend([list(zip(rm[ix,0,[1,2,4,3,1]],zm[ix,0,[1,2,4,3,1]])) 
                      for ix in np.arange(0,com.nx+2)])
        lines.extend([list(zip(rm[ix,com.ny+1,[1,2,4,3,1]],zm[ix,com.ny+1,[1,2,4,3,1]])) 
                      for ix in np.arange(0,com.nx+2)])
        lc = LineCollection(lines, linewidths=linewidth, color=color, zorder=1)
        ax.add_collection(lc)
        ax.autoscale()
    else:
        lines = [list(zip(rm[ix,iy,[1,2,4,3,1]],zm[ix,iy,[1,2,4,3,1]])) 
                 for ix in np.arange(0,com.nx+2) for iy in np.arange(0,com.ny+2)]
        lc = LineCollection(lines, linewidths=linewidth, color=color, zorder=zorder)
        ax.add_collection(lc)
        ax.autoscale()
    
    if showBad:
        rover = []
        zover = []
        for ix, iy in analysis.overlappingCells():
            rover.append(rm[ix,iy,0])
            zover.append(zm[ix,iy,0])
        if rover:
            plt.scatter(rover, zover, c='orange', marker='o', label='Overlapping (%d)' % len(rover), zorder=2)
        rbad = []
        zbad = []
        for ix, iy in analysis.badCells():
            rbad.append(rm[ix,iy,0])
            zbad.append(zm[ix,iy,0])
        if rbad:
            plt.scatter(rbad, zbad, c='red', marker='x', label='Invalid polygon (%d)' % len(rbad), zorder=3)
        if rbad or rover:
            plt.legend()
    
    if iso:
        plt.axis('equal')

    if wPlates:
        plt.plot(grd.rplate1, grd.zplate1, color='red')
        plt.plot(grd.rplate1, grd.zplate1, color='red', marker="o")
        plt.plot(grd.rplate2, grd.zplate2, color='red')
        plt.plot(grd.rplate2, grd.zplate2, color='red', marker="o")

    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.grid(False)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if show:
        plt.tight_layout()
        plt.show()
    
    
def showCell(ix, iy):
    '''
    Show the location of cell ix, iy in the mesh.
    '''
    plotmesh(show=False)
    corners = (1,3,4,2,1)
    plt.plot(com.rm[ix, iy, corners], com.zm[ix, iy, corners], c='red') 
    plt.show()
    
    
def plotAreas():
    '''
    Calculate signed area of all cells to make sure corners are in same order.
    Differences in signed area might indicate that some cells are flipped.
    Positive signed area = counterclockwise.
    '''
    def PolygonArea(corners):
        """
        https://stackoverflow.com/a/24468019
        """
        n = len(corners) # of corners
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corners[i][0] * corners[j][1]
            area -= corners[j][0] * corners[i][1]
        area = area / 2.0
        return area
    
    areas = np.zeros((com.nx+2, com.ny+2))
    for ix in range(0, com.nx+2):
        for iy in range(0, com.ny+2):
            corners = (1,3,4,2)
            rcenter = com.rm[ix, iy, 0]
            zcenter = com.zm[ix, iy, 0]
            # Subtract rcenter and zcenter to avoid numerical precision issues
            vs = list(zip(com.rm[ix, iy, corners]-rcenter, com.zm[ix, iy, corners]-zcenter))
            areas[ix, iy] = PolygonArea(vs)
            if areas[ix, iy] > 0:
                print(ix, iy, areas[ix, iy])
    
    print('min', np.min(areas))
    print('max', np.max(areas))
    plotvar(areas, cmap=plt.cm.viridis, label='Area [m^2]')
    
    
def plotCellRotated(ix, iy, edge=1):
    '''
    Plot specified cell to determine if it is healthy and not twisted.
    Cell is rotated to make it easier to see. Increment "edge" variable in case
    it's still difficult to see.
    '''
    corners = (1,3,4,2)
    rcenter = com.rm[ix, iy, 0]
    zcenter = com.zm[ix, iy, 0]
    rc = com.rm[ix,iy,corners]-rcenter
    zc = com.zm[ix,iy,corners]-zcenter
    theta = -np.arctan2(zc[1+edge]-zc[0+edge], rc[1+edge]-rc[0+edge])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    cRot = R.dot(np.array([rc, zc]))
    rc = cRot[0,:]
    zc = cRot[1,:]
    plt.plot(rc, zc)
    plt.scatter(rc[0], zc[0])
    plt.show()
    
    
def show_qpar():
    fig, ax = plt.subplots(1)

    #-parallel heat flux
    bbb.fetx=bbb.feex+bbb.feix

    #-radial profile of qpar below entrance to the outer leg
    qpar1=(bbb.fetx[com.ixpt2[0]+1,com.iysptrx:]/com.sx[com.ixpt2[0]+1,com.iysptrx:])/com.rr[com.ixpt2[0]+1,com.iysptrx:]

    #-radial profile of qpar below entrance to the inner leg
    qpar2=(bbb.fetx[com.ixpt1[0],com.iysptrx:]/com.sx[com.ixpt1[0],com.iysptrx:])/com.rr[com.ixpt1[0],com.iysptrx:]

    ###fig1 = plt.figure()
    plt.plot(com.yyc[com.iysptrx:], qpar1)
    plt.plot(com.yyc[com.iysptrx:], qpar2, linestyle="dashed")

    plt.xlabel('R-Rsep [m]')
    plt.ylabel('qpar [W/m^2]')
    fig.suptitle('qpar at inner & outer (dash) divertor entrance')
    plt.grid(True)

    plt.show()


def plotr(v, ix=1, 
          title="UEDGE data", 
          xtitle="R-Rsep [m]", 
          ytitle="", 
          linestyle="solid",
          overplot=False):


    if (overplot == False):
        ###print("Overplot=False")
        fig,ax = plt.subplots(1)
        fig.suptitle(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
 #   else:
 #       print("Overplot=True")


    plt.plot(com.yyc,v[ix,:], linestyle=linestyle)
    plt.grid(True)

    plt.show()


def showIndices():
    '''
    Plot grid and overlay cell indices as text.
    '''
    fig, ax = plt.subplots(1)

    plt.axes().set_aspect('equal', 'datalim')


    for iy in np.arange(0,com.ny+2):
        for ix in np.arange(0,com.nx+2):
            plt.plot(com.rm[ix,iy,[1,2,4,3,1]],
                     com.zm[ix,iy,[1,2,4,3,1]], 
                     color="b", linewidth=0.5)
            plt.text(com.rm[ix, iy, 0], com.zm[ix, iy, 0], '%d,%d' % (ix, iy), fontsize=8)


    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    fig.suptitle('UEDGE mesh')
    plt.grid(True)

    plt.show()


rlabel = r'$R_{omp}-R_{sep}$ [mm]'
c0 = 'C0'
c1 = 'tomato'
c2 = 'black'
    
    
def allSame(arr):
    """
    Return true if every element in the numpy array has the same value.
    """
    arrf = arr.flatten()
    return np.all(arrf[0] == arrf)
    
    
def getConfigText():
    # Label
    txt = r'$\bf{Run\ label}$ ' + bbb.label[0].decode('utf-8')
    # Path
    path = os.getcwd()
    spath = path.split('UEDGE_private/')
    if len(spath) > 1:
        path = spath[-1]
    txt += '\n' + r'$\bf{Path}$ ' + path
    # Date created
    date = datetime.datetime.now().strftime('%I:%M %p %a %d %b %Y')
    txt += '\n' + r'$\bf{Plots\ created}$ ' + date
    # UEDGE version
    txt += '\n' + r'$\bf{UEDGE\ version}$ ' + str(uedgeVersion)
    
    # Grid 
    numBad = len(analysis.badCells())
    txt += '\n\n' + r'$\bf{Grid}$ nx = %d, ny = %d, %d cells are invalid polygons' % (com.nx, com.ny, numBad)
    if numBad > 0:
        txt += r' $\bf{(!!!)}$'
    # Core ni
    txt += '\n' + r'$\bf{Core\ n_i}$ '
    d = {0: 'set flux to curcore/sy locally in ix',
         1: r'fixed uniform %.3g m$^{-3}$' % bbb.ncore[0],
         2: 'set flux & ni over range',
         3: 'set icur = curcore-recycc*fngy, const ni',
         4: 'use impur. source terms (impur only)',
         5: 'set d(ni)/dy = -ni/lynicore at midp & ni constant poloidally'}
    if bbb.isnicore[0] in d:
        txt += d[bbb.isnicore[0]]
    # Core ng
    txt += '\n' + r'$\bf{Core\ n_n}$ '
    d = {0: 'set loc flux = -(1-albedoc)*ng*vtg/4',
         1: r'fixed uniform %.3g /m$^3$' % bbb.ngcore[0],
         2: 'invalid option',
         3: 'extrapolation, but limited'}
    if bbb.isngcore[0] in d:
        txt += d[bbb.isngcore[0]]
    else:
        txt += 'set zero derivative'
    # Core Te,Ti or Pe,Pi
    txt += '\n' + r'$\bf{Core\ T_e,T_i\ or\ P_e,P_i}$ '
    if bbb.iflcore == 0:
        txt += r'fixed $T_e$ = %.3g eV, $T_i$ = %.3g eV' % (bbb.tcoree, bbb.tcorei)
    elif bbb.iflcore == 1:
        txt += r'fixed $P_e$ = %.3g MW, $P_i$ = %.3g MW' % (bbb.pcoree/1e6, bbb.pcorei/1e6)
    # Core ion vparallel
    txt += '\n' + r'$\bf{Core\ ion\ v_\parallel\ (up)}$ '
    d = {0: 'up = upcore at core boundary',
         1: 'd(up)/dy = 0 at core boundary',
         2: 'd^2(up)/dy^2 = 0',
         3: 'fmiy = 0',
         4: 'tor. ang mom flux = lzflux & n*up/R=const',
         5: 'ave tor vel = utorave & n*up/R=const'}
    if bbb.isupcore[0] in d:
        txt += d[bbb.isupcore[0]]
    # D,chi if constant
    txt += '\n' + r'$\bf{Uniform\ coeffs}$ $D$ = %.3g m$^2/$s, $\chi_e$ = %.3g m$^2/$s, $\chi_i$ = %.3g m$^2/$s' % (bbb.difni[0], bbb.kye, bbb.kyi)
    # CF wall Te
    txt += '\n' + r'$\bf{CF\ wall\ T_e}$ '
    if allSame(bbb.tewallo):
        tewallo = 'fixed %.3g eV' % bbb.tewallo[0]
    else: 
        tewallo = 'fixed to 1D spatially varying profile (bbb.tewallo)'
    d = {0: 'zero energy flux',
         1: tewallo,
         2: 'extrapolated',
         3: r'$L_{Te}$ = %.3g m' % bbb.lyte[1],
         4: 'feey = bceew*fniy*te'}
    if bbb.istewc in d:
        txt += d[bbb.istewc]
    # PF wall Te
    txt += '\n' + r'$\bf{PF\ wall\ T_e}$ '
    if allSame(bbb.tewalli):
        tewalli = 'fixed %.3g eV' % bbb.tewalli[0]
    else: 
        tewalli = 'fixed to 1D spatially varying profile (bbb.tewalli)'
    d = {0: 'zero energy flux',
         1: tewalli,
         2: 'extrapolated',
         3: r'$L_{Te}$ = %.3g m' % bbb.lyte[0],
         4: 'feey = bceew*fniy*te'}
    if bbb.istepfc in d:
        txt += d[bbb.istepfc]
    # CF wall Ti
    txt += '\n' + r'$\bf{CF\ wall\ T_i}$ '
    if allSame(bbb.tiwallo):
        tiwallo = 'fixed %.3g eV' % bbb.tiwallo[0]
    else:
        tiwallo = 'fixed to 1D spatially varying profile (bbb.tiwallo)'
    d = {0: 'zero energy flux',
         1: tiwallo,
         2: 'extrapolated',
         3: r'$L_{Ti}$ = %.3g m' % bbb.lyti[1],
         4: 'feiy = bceiw*fniy*ti'}
    if bbb.istiwc in d:
        txt += d[bbb.istiwc]
    # PF wall Ti
    txt += '\n' + r'$\bf{PF\ wall\ T_i}$ '
    if allSame(bbb.tiwalli):
        tiwalli = 'fixed %.3g eV' % bbb.tiwalli[0]
    else:
        tiwalli = 'fixed to 1D spatially varying profile (bbb.tiwalli)'
    d = {0: 'zero energy flux',
         1: tiwalli,
         2: 'extrapolated',
         3: r'$L_{Ti}$ = %.3g m' % bbb.lyti[0],
         4: 'feiy = bceiw*fniy*ti'}
    if bbb.istipfc in d:
        txt += d[bbb.istipfc]
    # CF wall ni
    txt += '\n' + r'$\bf{CF\ wall\ n_i}$ '
    if allSame(bbb.nwallo):
        nwallo = r'fixed %.3g m$^{-3}$' % bbb.nwallo[0]
    else:
        nwallo = 'fixed to 1D spatially varying profile (bbb.nwallo)'
    z = {0: 'dn/dy = 0', 1: 'fniy = 0'}
    d = {0: z[bbb.ifluxni],
         1: nwallo,
         2: 'extrapolated',
         3: r'$L_{ni}$ = %.3g m, $n_{wall\ min}$ = %.3g m$^{-3}$' % (bbb.lyni[1], bbb.nwomin[0])}
    if bbb.isnwcono[0] in d:
        txt += d[bbb.isnwcono[0]]
    # PF wall ni
    txt += '\n' + r'$\bf{PF\ wall\ n_i}$ '
    if allSame(bbb.nwalli):
        nwalli = r'fixed %.3g m$^{-3}$' % bbb.nwalli[0]
    else:
        nwalli = 'fixed to 1D spatially varying profile (bbb.nwalli)'
    z = {0: 'dn/dy = 0', 1: 'fniy = 0'}
    d = {0: z[bbb.ifluxni],
         1: nwalli,
         2: 'extrapolated',
         3: r'$L_{ni}$ = %.3g m, $n_{wall\ min}$ = %.3g m$^{-3}$' % (bbb.lyni[0], bbb.nwimin[0])}
    if bbb.isnwconi[0] in d:
        txt += d[bbb.isnwconi[0]]
    # Flux limits
    if bbb.flalfe == bbb.flalfi == 0.21 and bbb.flalfv == 1 and np.all(bbb.lgmax == 0.05) and np.all(bbb.lgtmax == 0.05):
        flim = 'on'
    elif bbb.flalfe == bbb.flalfi == 1e20 and bbb.flalfv == 1e10 and np.all(bbb.lgmax == 1e20) and np.all(bbb.lgtmax == 1e20):
        flim = 'off'
    else:
        flim = 'unknown'
    txt += '\n' + r'$\bf{Flux\ limits}$ %s' % flim
    # Plates H recycling coefficient
    txt += '\n' + r'$\bf{Recycling\ coefficient}$ %.5g (plates), %.5g (walls)' % (bbb.recycp[0], bbb.recycw[0])
    # Neutral model
    if bbb.isngon[0] == 1 and bbb.isupgon[0] == 0 and com.nhsp == 1:
        nmodel = 'diffusive neutrals'
    elif bbb.isngon[0] == 0 and bbb.isupgon[0] == 1 and com.nhsp == 2:
        nmodel = 'inertial neutrals'
    else:
        nmodel = 'unknown'
    txt += '\n' + r'$\bf{Neutral\ model}$ %s' % nmodel
    # Impurity
    if bbb.isimpon == 2:
        txt += '\n' + r'$\bf{Impurity\ Z}$ %i' % (api.atn)
    elif bbb.isimpon == 6:
        txt += '\n' + r'$\bf{Impurity\ Z}$ %i %s' % (bbb.znuclin[2], bbb.ziin[2:com.nzsp[0]+2])
    # Impurity model
    txt += '\n' + r'$\bf{Impurity\ model}$ '
    d = {0: 'no impurity',
         2: 'fixed-fraction model',
         3: 'average-impurity-ion model (disabled)',
         4: 'INEL multi-charge-state model (disabled)',
         5: "Hirshman's reduced-ion model",
         6: 'force-balance model or nusp_imp > 0; see also isofric for full-Z drag term',
         7: 'simultaneous fixed-fraction and multi-charge-state (isimpon=6) models'}
    if bbb.isimpon in d:
        txt += d[bbb.isimpon]
    # Impurity fraction
    if bbb.isimpon == 2:
        txt += '\n' + r'$\bf{Impurity\ fraction}$ '
        if allSame(bbb.afracs):
            txt += '%.3g (spatially uniform)' % bbb.afracs[0,0]
        else:
            txt += 'spatially varying (mean = %.3g, std = %.3g, min = %.3g, max = %.3g)' % (np.mean(bbb.afracs), np.std(bbb.afracs), np.min(bbb.afracs), np.max(bbb.afracs))
    # Potential equation
    txt += '\n' + r'$\bf{Potential\ equation}$ '
    d = {0: 'off',
         1: 'on, b0 = %.3g' % bbb.b0}
    if bbb.isphion in d:
        txt += d[bbb.isphion]
            
    # Converged
    if bbb.iterm == 1:
        converged = 'yes'
    else: 
        converged = 'NOOOOOOOOO' # just to catch viewer's attention
    txt += '\n\n' + r'$\bf{Converged}$ ' + converged + (', sim. time %.3g s' % bbb.dt_tot)
    # Field line angle
    flangs = analysis.fieldLineAngle()
    txt += '\n' + r'$\bf{Field\ line\ angle}$ %.3g$\degree$ inner target, %.3g$\degree$ outer target' % (flangs[0,com.iysptrx+1], flangs[com.nx+1,com.iysptrx+1])
    # Separatrix
    nisep = (bbb.ni[bbb.ixmp,com.iysptrx,0]+bbb.ni[bbb.ixmp,com.iysptrx+1,0])/2
    nnsep = (bbb.ng[bbb.ixmp,com.iysptrx,0]+bbb.ng[bbb.ixmp,com.iysptrx+1,0])/2
    tisep = (bbb.ti[bbb.ixmp,com.iysptrx]+bbb.ti[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    tesep = (bbb.te[bbb.ixmp,com.iysptrx]+bbb.te[bbb.ixmp,com.iysptrx+1])/2/bbb.ev
    txt += '\n' + r'$\bf{Separatrix}$ $n_i$ = %.2g m$^{-3}$, $n_n$ = %.2g m$^{-3}$, $T_i$ = %.3g eV, $T_e$ = %.3g eV' % (nisep, nnsep, tisep, tesep)
    # Corner neutral pressure
    txt += '\n' + r'$\bf{Outer\ PF\ corner\ p_n}$ %.3g Pa' % (bbb.ng[:,:,0]*bbb.ti)[com.nx,1]
    # Power sharing
    powcc = bbb.feey + bbb.feiy 
    ixilast = analysis.ixilast()
    powcci = np.sum(powcc[com.ixpt1[0]+1:ixilast+1,com.iysptrx])/1e6
    powcco = np.sum(powcc[ixilast+1:com.ixpt2[0]+1,com.iysptrx])/1e6
    txt += '\n' + r'$\bf{Power\ sharing}$ 1:%.2g, $P_{LCFS\ inboard}$ = %.2g MW, $P_{LCFS\ outboard}$ = %.2g MW' % (powcco/powcci, powcci, powcco)
    # Impurity densities if multi-species
    if bbb.isimpon == 6:
        txt += '\n' + r'$\bf{n_{imp}}$ ' + analysis.impStats()
    # Impurity radiation
    if bbb.isimpon != 0:
        irad = bbb.prad/1e6*com.vol
        iradXPoint = np.sum(irad[com.ixpt1[0]:com.ixpt1[0]+2,:])+np.sum(irad[com.ixpt2[0]:com.ixpt2[0]+2,:])
        iradInnerLeg = np.sum(irad[:com.ixpt1[0],:])
        iradOuterLeg = np.sum(irad[com.ixpt2[0]+2:,:])
        iradMainChamberSOL = np.sum(irad[com.ixpt1[0]+1:bbb.ixmp,com.iysptrx+1:])+np.sum(irad[bbb.ixmp:com.ixpt2[0]+1,com.iysptrx+1:])
        iradCore = np.sum(irad[com.ixpt1[0]+1:bbb.ixmp,:com.iysptrx+1])+np.sum(irad[bbb.ixmp:com.ixpt2[0]+1,:com.iysptrx+1])
        txt += '\n' + r'$\bf{P_{rad\ imp}}$ $P_{tot}$ = %.2g MW, $P_{xpt}$ = %.2g MW, $P_{ileg}$ = %.2g MW, $P_{oleg}$ = %.2g MW,' % (np.sum(irad), iradXPoint, iradInnerLeg, iradOuterLeg) + '\n' + r'             $P_{main\ chamber\ SOL}$ = %.2g MW, $P_{core}$ = %.2g MW' % (iradMainChamberSOL, iradCore)
    # Domain power balance
    pInnerTarget, pOuterTarget, pCFWall, pPFWallInner, pPFWallOuter, prad, irad = analysis.powerLostBreakdown()
    pLoss = analysis.powerLost()
    txt += '\n' + r'$\bf{Power\ balance}$ $P_{loss}$ = %.2g MW = $P_{core}$%+.2g%%' % (pLoss/1e6, 100*pLoss/(bbb.pcoree+bbb.pcorei)-100) + '\n' + r'              ($P_{IT}$ = %.2g MW, $P_{OT}$ = %.2g MW, $P_{CFW}$ = %.2g MW, $P_{PFW}$ = %.2g MW, $P_{H}$ = %.2g MW, $P_{I}$ = %.2g MW)' % (pInnerTarget/1e6, pOuterTarget/1e6, pCFWall/1e6, (pPFWallOuter+pPFWallInner)/1e6, prad/1e6, irad/1e6)
    # Density balance
    nbalAbs = np.sum(np.abs(analysis.nonGuard(analysis.gridParticleBalance())))/np.sum(analysis.nonGuard(analysis.gridParticleSumAbs()))
    txt += '\n' + r'$\bf{Density\ balance}$ $\Sigma_{xy}|\Sigma_s(\Delta n)_s^{xy}|\left/\Sigma_{xy}\Sigma_s|(\Delta n)_s^{xy}|\right.$ = %.2g%%' % (nbalAbs*100)
    # Angle factor
    angleDegs = 2
    fi = 1./com.rr[0,com.iysptrx+1]*np.sin(angleDegs*np.pi/180.)
    fo = 1./com.rr[com.nx,com.iysptrx+1]*np.sin(angleDegs*np.pi/180.)
    # txt += '\n' + r'$\bf{Tilted\ plate\ factor\ q_{2\degree} = Fq_{pol}}$ $F_{inboard}$ = %.2g, $F_{outboard}$ = %.2g' % (fi, fo)
    return txt
    
    
def plotTransportCoeffs(patches):
    plt.rcParams['font.size'] = 10
    yyc_mm = com.yyc*1000
    iximp = analysis.iximp()
    ixomp = bbb.ixmp
    kwargs = {}
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # D line plot
    if not np.alltrue(bbb.dif_use == 0):
        plt.subplot(4,3,7)
        plt.plot(yyc_mm, bbb.dif_use[:,:,0][ixomp], c=c0, label=r'$D_{omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.dif_use[:,:,0][iximp], c=c1, label=r'$D_{imp}$', **kwargs)
        plt.title(r"$D$ [m$^2$/s]")
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        plt.yscale('log')
        plt.legend()
        plt.xlabel(rlabel)
        # D 2D image
        plt.subplot(4,3,10)
        plotvar(bbb.dif_use[:,:,0],  title=r"$D$ [m$^2$/s]", log=True, patches=patches, show=False, 
                rzlabels=False, stats=False)
    # vconv line plot
    if not np.alltrue(bbb.vy_use == 0):
        plt.subplot(4,3,8)
        plt.plot(yyc_mm, bbb.vy_use[ixomp,:,0], c=c0, label=r'$v_{conv\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.vy_use[iximp,:,0], c=c1, label=r'$v_{conv\ imp}$', **kwargs)
        plt.title(r"$v_{conv}$ [m/s]")
        plt.legend()
        plt.xlabel(rlabel)
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        # vconv 2D image
        plt.subplot(4,3,11)
        plotvar(bbb.vy_use[:,:,0],  title=r"$v_{conv}$ [m/s]", log=False, patches=patches, show=False, 
                rzlabels=False, stats=False)
    # Chi line plot
    if not (np.alltrue(bbb.kye_use == 0) and np.alltrue(bbb.kyi_use == 0)):
        plt.subplot(4,3,9)
        plt.plot(yyc_mm, bbb.kye_use[ixomp], c=c0, lw=1, label=r'$\chi_{e\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kyi_use[ixomp], c=c0, lw=2, label=r'$\chi_{i\ omp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kye_use[iximp], c=c1, lw=1, label=r'$\chi_{e\ imp}$', **kwargs)
        plt.plot(yyc_mm, bbb.kyi_use[iximp], c=c1, lw=2, label=r'$\chi_{i\ imp}$', **kwargs)
        plt.title(r'$\chi$ [m$^2$/s]')
        plt.xlabel(rlabel)
        if np.any(bbb.kye_use > 0) and np.any(bbb.kyi_use > 0):
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
        for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
        # Chi 2D image
        plt.subplot(4,3,12)
        plotvar(bbb.kye_use,  title=r"$\chi$ [m$^2$/s]", log=True, patches=patches, show=False, 
                rzlabels=False, stats=False)
        plt.rcParams['font.size'] = 12
    
    
def plot2Dvars(patches):
    kwargs = {'log': True, 'patches': patches, 'show': False, 'rzlabels': False}
    kwargslin = {'log': False, 'patches': patches, 'show': False, 'rzlabels': False}
    plt.subplot(331)
    plotvar(bbb.te/bbb.ev,  title=r"$T_e$ [eV]", **kwargs)
    plt.subplot(332)
    plotvar(bbb.ti/bbb.ev,  title=r"$T_i$ [eV]", **kwargs)
    plt.subplot(334)
    plotvar(bbb.ni[:,:,0],  title=r"$n_{i}$ [m$^{-3}$]", **kwargs)
    plt.subplot(335)
    plotvar(bbb.ng[:,:,0],  title=r"$n_n$ [m$^{-3}$]", **kwargs)
    plt.subplot(337)
    plotvar(bbb.up[:,:,0],  title=r"$u_{pi}$ [m/s]", sym=True, **kwargs)
    if bbb.up.shape[2] > 1:
        plt.subplot(338)
        plotvar(bbb.up[:,:,1],  title=r"$u_{pn}$ [m/s]", sym=True, **kwargs)
    if bbb.isphion == 1:
        plt.subplot(339)
        plotvar(bbb.phi,  title=r"$\phi$ [V]", sym=True, **kwargslin)
    if bbb.isimpon == 6:
        plt.subplot(336)
        plotvar(np.sum(bbb.ni[:,:,2:], axis=2), title=r"$n_{imp}$ [m$^{-3}$]", **kwargs)
    

def plotnTprofiles(plotV0, h5):
    plt.rcParams['font.size'] = 10
    # plt.subplots_adjust(hspace = .01)
    yyc_mm = com.yyc*1000
    ixomp = bbb.ixmp
    iximp = analysis.iximp()
    df = sparc.getV0data()
    df = df[(df['rho [mm]'] > min(yyc_mm)) & (df['rho [mm]'] < max(yyc_mm))]
    niV0 = df[' ni [10^20 m^-3]']*1e20
    TiV0 = df[' Ti [keV]']*1000
    TeV0 = df[' Te [keV]']*1000
    rhoV0mm = df['rho [mm]']
    lineNiIn =  {'c': c1, 'ls': '-',  'lw': 2}
    lineNiOut = {'c': c0, 'ls': '-',  'lw': 2}
    lineNiV0 =  {'c': c2, 'ls': '-',  'lw': 2}
    lineNgIn =  {'c': c1, 'ls': '--', 'lw': 2}
    lineNgOut = {'c': c0, 'ls': '--', 'lw': 2}
    lineTiIn =  {'c': c1, 'ls': '-',  'lw': 2}
    lineTiOut = {'c': c0, 'ls': '-',  'lw': 2}
    lineJs1In = {'c': c1, 'ls': '-',  'lw': 2}
    lineJs3In = {'c': c1, 'ls': '-.',  'lw': 2}
    lineJs1Out = {'c': c0, 'ls': '-',  'lw': 2}
    lineJs2In = {'c': c1, 'ls': '-',  'lw': 1}
    lineJs2Out = {'c': c0, 'ls': '-',  'lw': 1}
    lineJs3Out = {'c': c0, 'ls': '-.',  'lw': 2}
    lineTiV0 =  {'c': c2, 'ls': '-',  'lw': 2}
    lineTeIn =  {'c': c1, 'ls': '-',  'lw': 1}
    lineTeOut = {'c': c0, 'ls': '-',  'lw': 1}
    lineTeV0 =  {'c': c2, 'ls': '-',  'lw': 1}
    #
    plt.subplot(521)
    # plt.title('Inner midplane')
    if plotV0:
        plt.plot(rhoV0mm, niV0, label=r'$n_i$ V0', **lineNiV0)
    plt.plot(yyc_mm, bbb.ni[:,:,0][iximp], label=r'$n_{i\,imp}$', **lineNiIn)
    # plt.plot(yyc_mm, bbb.ng[:,:,0][iximp], label=r'$n_{n\,imp}$', **lineNgIn)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.yscale('log')
    #
    plt.subplot(522)
    # plt.title('Outer midplane')
    plt.plot(yyc_mm, bbb.ni[:,:,0][ixomp], label=r'$n_{i\,omp}$', **lineNiOut)
    # plt.plot(yyc_mm, bbb.ng[:,:,0][ixomp], label=r'$n_{n\,omp}$', **lineNgOut)
    if h5:
        var = 'neomid'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            yerr = h5[var+'/value_err'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            #plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
            plt.errorbar(rho[mask]*1000, val[mask], c='skyblue', ms=2, fmt='o', yerr=yerr[mask], zorder=1, elinewidth=0.7)
        var = 'neomidfit'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            plt.plot(rho[mask]*1000, val[mask], c='k', zorder=100,ls='--')
    if plotV0:
        plt.plot(rhoV0mm, niV0, label=r'$n_i$ V0', **lineNiV0)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.legend()
    # ymax = np.max(bbb.ni[bbb.ixmp,:,0])
    # plt.ylim([-0.05*ymax,ymax*1.05])
    plt.yscale('log')
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(523)
    # plt.title('Inner midplane')
    if plotV0:
        plt.plot(rhoV0mm, TiV0, label=r'$T_i$ V0', **lineTiV0)
        plt.plot(rhoV0mm, TeV0, label=r'$T_e$ V0', **lineTeV0)
    plt.plot(yyc_mm, bbb.ti[iximp]/bbb.ev, label=r'$T_{i\,imp}$', **lineTiIn)
    plt.plot(yyc_mm, bbb.te[iximp]/bbb.ev, label=r'$T_{e\,imp}$', **lineTeIn)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(524)
    # plt.title('Outer midplane')
    if h5:
        var = 'teomid'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            yerr = h5[var+'/value_err'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            # plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
            plt.errorbar(rho[mask]*1000, val[mask], c='skyblue', ms=2, yerr=yerr[mask], fmt='o', zorder=1, elinewidth=0.7)
        var = 'teomidfit'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] <= rho) & (rho <= com.yyc[-1])
            plt.plot(rho[mask]*1000, val[mask], c='k', zorder=100, ls='--')
    if plotV0:
        plt.plot(rhoV0mm, TiV0, label=r'$T_i$ V0', **lineTiV0)
        plt.plot(rhoV0mm, TeV0, label=r'$T_e$ V0', **lineTeV0)
    plt.plot(yyc_mm, bbb.ti[ixomp]/bbb.ev, label=r'$T_{i\,omp}$', **lineTiOut)
    plt.plot(yyc_mm, bbb.te[ixomp]/bbb.ev, label=r'$T_{e\,omp}$', **lineTeOut)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    # ymax = np.max([np.max(bbb.ti[bbb.ixmp]),np.max(bbb.te[bbb.ixmp])])/bbb.ev
    # plt.ylim([-0.05*ymax,ymax*1.05])
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(525)
    # plt.title('Inner plate')
    plt.plot(yyc_mm, bbb.ni[:,:,0][0], label=r'$n_{i\,it}$', **lineNiIn)
    plt.plot(yyc_mm, bbb.ng[:,:,0][0], label=r'$n_{n\,it}$', **lineNgIn)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(526)
    # plt.title('Outer plate')
    if h5:
        var = 'neotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
    plt.plot(yyc_mm, bbb.ni[:,:,0][com.nx+1], label=r'$n_{i\,ot}$', **lineNiOut)
    plt.plot(yyc_mm, bbb.ng[:,:,0][com.nx+1], label=r'$n_{n\,ot}$', **lineNgOut)
    plt.ylabel(r'$n$ [m$^{-3}$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(527)
    # plt.title('Inner plate')
    plt.plot(yyc_mm, bbb.ti[0]/bbb.ev, label=r'$T_{i\,it}$', **lineTiIn)
    plt.plot(yyc_mm, bbb.te[0]/bbb.ev, label=r'$T_{e\,it}$', **lineTeIn)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    # plt.yscale('log')
    #
    plt.subplot(528)
    # plt.title('Outer plate')
    if h5:
        var = 'teotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask], c='skyblue', s=5)
    plt.plot(yyc_mm, bbb.ti[com.nx+1]/bbb.ev, label=r'$T_{i\,ot}$', **lineTiOut)
    plt.plot(yyc_mm, bbb.te[com.nx+1]/bbb.ev, label=r'$T_{e\,ot}$', **lineTeOut)
    plt.ylabel(r'$T$ [eV]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(529)
    jsat1 = bbb.qe*bbb.ni[:,:,0]*np.sqrt((bbb.zeff*bbb.te+bbb.ti)/bbb.mi[0])
    vpolce = -bbb.cf2ef*bbb.v2ce[:,:,0]*(1-com.rr**2)**.5
    vtorce = bbb.cf2ef*bbb.v2ce[:,:,0]*com.rr
    vpolcb = -bbb.cf2bf*bbb.v2cb[:,:,0]*(1-com.rr**2)**.5
    vtorcb = bbb.cf2bf*bbb.v2cb[:,:,0]*com.rr
    upol = bbb.up[:,:,0]*com.rr
    utor = bbb.up[:,:,0]*(1-com.rr**2)**.5
    vtot = ((vpolce+vpolcb+upol)**2+(vtorce+vtorcb+utor)**2)**.5
    #jsat2 = bbb.qe*bbb.ni[:,:,0]*vtot
    # jsat2 = bbb.qe*bbb.ni[:,:,0]*bbb.up[:,:,0]
    #jsat3 = bbb.qe*bbb.ni[:,:,0]*np.sqrt((bbb.zeff*bbb.te+3*bbb.ti)/bbb.mi[0])
    jscale = 1000
    # plt.title('Inner plate')
    plt.plot(yyc_mm, jsat1[0]/jscale, label=r'$j_{sat\,it}\ \gamma_i=1$', **lineJs1In)
    jsat2 = bbb.qe*bbb.ni[:,:,0]*bbb.up[:,:,0]
    # if not np.all(bbb.fqpsatlb==0):
    #     plt.plot(yyc_mm, bbb.fqpsatlb[:,0]/com.sx[0]/com.sxnp[0]/jscale, label=r'fqpsatlb', **lineJs2In)
    # plt.plot(yyc_mm, jsat1[1]/jscale, **lineJs3In)
    plt.ylabel(r'$j_{sat}$ [kA/m$^2$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    #
    plt.subplot(5,2,10)
    # plt.title('Outer plate')
    if h5:
        var = 'jsotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask]/jscale, c='skyblue', s=5)
    plt.plot(yyc_mm, jsat1[com.nx+1]/jscale, label=r'$j_{sat\,ot}\ \gamma_i=1$', **lineJs1Out)
    # if not np.all(bbb.fqpsatrb==0):
    #     fqpsatrb = bbb.qe*bbb.isfdiax*( bbb.ne[com.nx+1,:]*bbb.v2ce[com.nx,:,0]*bbb.rbfbt[com.nx+1,:]*com.sx[com.nx,:] + bbb.fdiaxrb[:,0] )
    #     fqpsatrb += bbb.qe*bbb.zi[0]*bbb.ni[com.nx+1,:,0]*bbb.up[com.nx,:,0]*com.sx[com.nx,:]*com.rrv[com.nx,:]  
    #     plt.plot(yyc_mm, fqpsatrb[:,0]/com.sx[com.nx+1]/com.sxnp[com.nx+1]/jscale, label=r'fqpsatrb', **lineJs2In)
    #plt.plot(yyc_mm, jsat1[com.nx]/jscale, **lineJs3In)
    plt.ylabel(r'$j_{sat}$ [kA/m$^2$]')
    plt.xlabel(rlabel)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.rcParams['font.size'] = 12
    plt.tight_layout(h_pad=0.01)
    
    
def plotAlongLegs():
    xpto = com.xfs[com.ixpt2[0]]
    xpti = com.xfs[com.ixpt1[0]]
    leg = list(range(0,com.ixpt1[0]+1))[::-1]
    xf = [(xpti-com.xfs[ix])*100 for ix in leg]
    legc = leg[:-1]
    xc = [(xpti-com.xcs[ix])*100 for ix in legc]
    xlim = [xc[0],xf[-1]+1]
    plt.figure(figsize=(8.5,11))
    plt.subplot(421)
    plt.plot(xf[1:], [np.sum(-(bbb.feex+bbb.feix)[ix,1:com.ny+1])/1e6 for ix in leg][1:], label='$P_{conv+cond}$', lw=2, c='k')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xf[1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW]')
    plt.title('$P_{conv+cond}$ along inner leg')
    plt.subplot(423)
    if np.mean(bbb.afracs) > 1e-20:
        plt.plot(xc, [np.sum((bbb.prad*com.vol)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ imp}$', c='C2')
    plt.plot(xc, [np.sum((bbb.erliz)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ ioniz}$', c='C3')
    plt.plot(xc, [np.sum((bbb.erlrc)[ix,1:com.ny+1])/1e6 for ix in legc], label='$P_{rad\ recomb}$', c='C4')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW/m$^3$]')
    plt.title('$P_{rad}$ along inner leg')
    plt.subplot(425)
    plt.plot(xc, [np.max(bbb.te[ix,1:com.ny+1])/bbb.ev for ix in legc], label='$T_e$', lw=1, c='C1')
    plt.plot(xc, [np.max(bbb.ti[ix,1:com.ny+1])/bbb.ev for ix in legc], label='$T_i$', lw=2, c='C1')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$T$ [eV]')
    plt.title('$T_{max}$ along inner leg')
    plt.subplot(427)
    plt.plot(xc, [np.mean(bbb.ni[ix,1:com.ny+1,0]) for ix in legc], label='$n_i$', lw=2, c='C0')
    plt.plot(xc, [np.mean(bbb.ng[ix,1:com.ny+1,0]) for ix in legc], label='$n_n$', c='C0', ls='--', lw=2)
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$n$ [m$^{-3}$]')
    plt.title('$n$ along inner leg')
    plt.legend()

    leg = range(com.ixpt2[0]+1,com.nx+1)
    xf = [(com.xfs[ix]-xpto)*100 for ix in leg]
    xc = [(com.xcs[ix]-xpto)*100 for ix in leg]
    xlim = [xc[0],xf[-1]+1]
    plt.subplot(422)
    plt.plot(xf, [np.sum((bbb.feex+bbb.feix)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{conv+cond}$', lw=2, c='k')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xf: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW]')
    plt.title('$P_{conv+cond}$ along outer leg')
    plt.subplot(424)
    if np.mean(bbb.afracs) > 1e-20:
        plt.plot(xc, [np.sum((bbb.prad)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ imp}$', c='C2')
    plt.plot(xc, [np.sum((bbb.erliz/com.vol)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ ioniz}$', c='C3')
    plt.plot(xc, [np.sum((bbb.erlrc/com.vol)[ix,1:com.ny+1])/1e6 for ix in leg], label='$P_{rad\ recomb}$', c='C4')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$P$ [MW/m$^3$]')
    plt.title('$P_{rad}$ along outer leg')
    plt.subplot(426)
    plt.plot(xc, [np.max(bbb.te[ix,1:com.ny+1])/bbb.ev for ix in leg], label='$T_e$', lw=1, c='C1')
    plt.plot(xc, [np.max(bbb.ti[ix,1:com.ny+1])/bbb.ev for ix in leg], label='$T_i$', lw=2, c='C1')
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$T$ [eV]')
    plt.title('$T_{max}$ along outer leg')
    plt.legend()
    plt.subplot(428)
    plt.plot(xc, [np.mean(bbb.ni[ix,1:com.ny+1,0]) for ix in leg], label='$n_i$', lw=2, c='C0')
    plt.plot(xc, [np.mean(bbb.ng[ix,1:com.ny+1,0]) for ix in leg], label='$n_n$', c='C0', ls='--', lw=2)
    plt.xlim(xlim)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in xc: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    plt.xlabel('Distance from X-point [cm]')
    plt.ylabel('$n$ [m$^{-3}$]')
    plt.title('$n$ along outer leg')
    plt.legend()
    
    
def plotPressures():
    yyc_mm = com.yyc*1000
    iximp = analysis.iximp()
    ixomp = bbb.ixmp
    # nT inboard calculation
    nT_imp = bbb.ni[:,:,0][iximp]*(bbb.te[iximp]+bbb.ti[iximp]) + bbb.ni[:,:,0][iximp]*bbb.mi[0]*bbb.up[:,:,0][iximp]**2
    nT_idiv = bbb.ni[:,:,0][1]*(bbb.te[1]+bbb.ti[1]) + bbb.ni[:,:,0][1]*bbb.mi[0]*bbb.up[:,:,0][1]**2
    # nT inboard plot
    plt.subplot(321)
    plt.title('Inboard thermal+ram pressure')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_imp[com.iysptrx+1:-1], ls='--', c=c1, label='Midplane')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_idiv[com.iysptrx+1:-1], c=c1, label='Divertor plate')
    plt.xlabel(rlabel)
    plt.ylabel(r'Pressure [Pa]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm[com.iysptrx+1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    # nT outboard calculation
    nT_omp = bbb.ni[:,:,0][ixomp]*(bbb.te[ixomp]+bbb.ti[ixomp]) + bbb.ni[:,:,0][ixomp]*bbb.mi[0]*bbb.up[:,:,0][ixomp]**2
    nT_odiv = bbb.ni[:,:,0][com.nx]*(bbb.te[com.nx]+bbb.ti[com.nx]) + bbb.ni[:,:,0][com.nx]*bbb.mi[0]*bbb.up[:,:,0][com.nx]**2
    # nT outboard plot
    plt.subplot(322)
    plt.title('Outboard thermal+ram pressure')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_omp[com.iysptrx+1:-1], ls='--', c=c0, label='Midplane')
    plt.plot(yyc_mm[com.iysptrx+1:-1], nT_odiv[com.iysptrx+1:-1], c=c0, label='Divertor plate')
    plt.xlabel(rlabel)
    plt.ylabel(r'Pressure [Pa]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in yyc_mm[com.iysptrx+1:]: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    
    
def plotqFits(h5):
    # qpar calculation
    ppar = analysis.Pparallel()
    rrf = analysis.getrrf()
    iy = com.iysptrx+1
    xq = com.yyc[iy:-1]
    ixo = com.ixpt2[0]
    ixi = com.ixpt1[0]
    #-radial profile of qpar at X-point outer
    qparo = ppar[ixo,iy:-1]/com.sx[ixo,iy:-1]/rrf[ixo,iy:-1]
    intqo = np.sum(ppar[ixo+1,iy:-1]) # integral along first set of edges that *enclose* the outer divertor
    #-radial profile of qpar at X-point inner
    qpari = -ppar[ixi,iy:-1]/com.sx[ixi,iy:-1]/rrf[ixi,iy:-1]
    intqi = np.sum(-ppar[ixi-1,iy:-1])
    # lamda_q fits
    expfun = lambda x, A, lamda_q_inv: A*np.exp(-x*lamda_q_inv) # needs to be in this form for curve_fit to work
    omax = np.argmax(qparo) # only fit stuff to right of max
    try:
        qofit, _ = curve_fit(expfun, xq[omax:], qparo[omax:], p0=[np.max(qparo),1000], bounds=(0, np.inf))
        lqo = 1000/qofit[1] # lamda_q in mm
        lqoGuess = lqo
    except Exception as e:
        print('q parallel outer fit failed:', e)
        qofit = None
        lqoGuess = 1.
    imax = np.argmax(qpari) # only fit stuff to right of max
    try:
        qifit, _ = curve_fit(expfun, xq[imax:], qpari[imax:], p0=[np.max(qpari),1000], bounds=(0, np.inf))
        lqi = 1000/qifit[1] # lamda_q in mm
        lqiGuess = lqi
    except Exception as e:
        print('q parallel inner fit failed:', e)
        qifit = None
        lqiGuess = 1.
    # qpar plotting
    plt.subplot(312)
    plt.title(r'$q_\parallel$ at divertor entrance ($P_{xpt\ in}:P_{xpt\ out}$ = 1:%.1f)' % (intqo/intqi))
    plt.plot(xq*1000, qparo/1e9, c=c0, label=r'X-point to outer wall, $P_{xpt}$ = %.3g MW' % (intqo/1e6))
    plt.plot(xq*1000, qpari/1e9, c=c1, label=r'X-point to inner wall, $P_{xpt}$ = %.3g MW' % (intqi/1e6))
    # ylim = plt.gca().get_ylim()
    if np.any(qofit):
        plt.plot(xq[omax:]*1000, expfun(xq, *qofit)[omax:]/1e9, c=c0, ls=':', 
                 label='Outboard exp. fit: $\lambda_q$ = %.3f mm' % lqo)
    if np.any(qifit):
        plt.plot(xq[imax:]*1000, expfun(xq, *qifit)[imax:]/1e9, c=c1, ls=':', 
                 label='Inboard exp. fit: $\lambda_q$ = %.3f mm' % lqi)
    try:
        ylim=[np.min([np.min(qparo[qparo>0]), np.min(qpari[qpari>0])])/1e9,np.max([np.max(qparo[qparo>0]), np.max(qpari[qpari>0])])/1e9]
        plt.ylim(ylim)
    except Exception as e:
        print('qpar ylim error:', e)
    plt.xlim([-0.1, com.yyc[-1]*1000])
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_\parallel$ [GW/m$^2$]')
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    
    # qsurf calculation
    #-radial profile of qpar below entrance to the outer leg
    psurfo = analysis.PsurfOuter()
    qsurfo = psurfo[1:-1]/com.sxnp[com.nx,1:-1]
    intqo = np.sum(psurfo)
    #-radial profile of qpar below entrance to the inner leg
    psurfi = analysis.PsurfInner()
    qsurfi = psurfi[1:-1]/com.sxnp[0,1:-1]
    intqi = np.sum(psurfi)
    # lamda_q fits
    def qEich(rho, q0, S, lqi, qbg, rho_0):
        rho = rho - rho_0
        # lqi is inverse lamda_q
        return q0/2*np.exp((S*lqi/2)**2-rho*lqi)*erfc(S*lqi/2-rho/S)+qbg
    bounds = ([0,0,0,0,com.yyc[0]], [np.inf,np.inf,np.inf,np.inf,com.yyc[-1]])
    oguess = (np.max(qsurfo)-np.min(qsurfo[qsurfo>0]), lqoGuess/1000/2, 1000/lqoGuess, np.min(qsurfo[qsurfo>0]), 0)
    try:
        qsofit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfo, p0=oguess, bounds=bounds)
        lqeo, So = 1000/qsofit[2], qsofit[1]*1000 # lamda_q and S in mm
    except Exception as e:
        print('qsurf outer fit failed:', e)
        qsofit = None
    iguess = (np.max(qsurfi)-np.min(qsurfi[qsurfi>0]), lqiGuess/1000/2, 1000/lqiGuess, np.min(qsurfi[qsurfi>0]), 0)
    try:
        qsifit, _ = curve_fit(qEich, com.yyc[1:-1], qsurfi, p0=iguess, bounds=bounds)
        lqei, Si = 1000/qsifit[2], qsifit[1]*1000 # lamda_q and S in mm 
    except Exception as e:
        print('qsurf inner fit failed:', e)
        qsifit = None
    # qsurf plotting
    plt.subplot(313)
    plt.title(r'$q_{surf\ tot}$ ($P_{surf\ in}:P_{surf\ out}$ = 1:%.1f)' % (intqo/intqi))
    if h5:
        var = 'qotarget'
        if var in h5.keys():
            rho = h5[var+'/rho'][()]
            val = h5[var+'/value'][()]
            mask = (com.yyc[0] < rho) & (rho < com.yyc[-1])
            plt.scatter(rho[mask]*1000, val[mask]/1e6, c='skyblue', s=5)
    plt.plot(com.yyc[1:-1]*1000, qsurfo/1e6, c=c0, label=r'Outboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqo/1e6, np.max(qsurfo)/1e6))
    plt.plot(com.yyc[1:-1]*1000, qsurfi/1e6, c=c1, label=r'Inboard plate, $P_{surf}$ = %.3g MW, $q_{peak}$ = %.3g MW/m$^2$' % (intqi/1e6, np.max(qsurfi)/1e6))
    plt.yscale('log')
    ylim = plt.gca().get_ylim()
    if np.any(qsofit):
        plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsofit)/1e6, c=c0, ls=':',
                 label=r'Outboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqeo, So))
    if np.any(qsifit):
        plt.plot(com.yyc[1:-1]*1000, qEich(com.yyc[1:-1], *qsifit)/1e6, c=c1, ls=':',
                 label=r'Inboard Eich fit: $\lambda_q$ = %.3f mm, $S$ = %.3g mm' % (lqei, Si))
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    plt.ylim(ylim)
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    
    
def plotPowerBreakdown():
    yyc_mm = com.yyc[1:-1]*1000
    pwrx = bbb.feex+bbb.feix

    # Inner target qsurf breakdown
    plt.subplot(221)
    plt.title('Inner target')
    plateIndex = 0
    xsign = -1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt1[0]
    xtarget = 0
    pwrxtot = xsign*np.sum(pwrx[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*pwrx[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Conv.+cond. e+i+n (%.2g MW)' % pwrxtot)
    fnixtot = xsign*np.sum(bbb.fnix[xtarget,1:-1,0])*bbb.ebind*bbb.ev/1e6
    plt.plot(yyc_mm, xsign*bbb.fnix[xtarget,1:-1,0]*bbb.ebind*bbb.ev/com.sxnp[xtarget,1:-1]/1e6, label='Surface recomb. (%.2g MW)' % fnixtot)
    ketot = np.sum(xsign*analysis.PionParallelKE()[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*analysis.PionParallelKE()[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Ion kinetic energy (%.2g MW)' % ketot)
    plthtot = np.sum(bbb.pwr_plth[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_plth[1:-1,plateIndex]/1e6, ls='--', label='H photons (%.2g MW)' % plthtot)
    pltztot = np.sum(bbb.pwr_pltz[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_pltz[1:-1,plateIndex]/1e6, ls='--', label='Imp. photons (%.2g MW)' % pltztot)
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    # Outer target qsurf breakdown
    plt.subplot(222)
    plt.title('Outer target')
    plateIndex = 1
    xsign = 1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt2[0]
    xlegEntrance = ixpt+1
    xtarget = com.nx
    xleg = slice(ixpt+1, xtarget+1)
    pwrxtot = xsign*np.sum(pwrx[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*pwrx[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Conv.+cond. e+i+n (%.2g MW)' % pwrxtot)
    fnixtot = xsign*np.sum(bbb.fnix[xtarget,1:-1,0])*bbb.ebind*bbb.ev/1e6
    plt.plot(yyc_mm, xsign*bbb.fnix[xtarget,1:-1,0]*bbb.ebind*bbb.ev/com.sxnp[xtarget,1:-1]/1e6, label='Surface recomb. (%.2g MW)' % fnixtot)
    ketot = np.sum(xsign*analysis.PionParallelKE()[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, xsign*analysis.PionParallelKE()[xtarget,1:-1]/com.sxnp[xtarget,1:-1]/1e6, label='Ion kinetic energy (%.2g MW)' % ketot)
    plthtot = np.sum(bbb.pwr_plth[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_plth[1:-1,plateIndex]/1e6, ls='--', label='H photons (%.2g MW)' % plthtot)
    pltztot = np.sum(bbb.pwr_pltz[1:-1,plateIndex]*com.sxnp[xtarget,1:-1])/1e6
    plt.plot(yyc_mm, bbb.pwr_pltz[1:-1,plateIndex]/1e6, ls='--', label='Imp. photons (%.2g MW)' % pltztot)
    plt.xlabel(rlabel)
    plt.ylabel(r'$q_{surf}$ [MW/m$^2$]')
    plt.yscale('log')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)


def plotPowerSurface():
    pwrx = bbb.feex+bbb.feix
    pwry = bbb.feey+bbb.feiy
    cmap = copy.copy(plt.cm.viridis)
    cmap.set_bad((1,0,0))
    fontsize = 8
    bdict = dict(boxstyle="round", alpha=0.5, color='lightgray')
    c = 'black'
    labelArgs = {}
    offset = .02

    # Inner target total qsurf
    ax = plt.subplot(223)
    labels = []
    plateIndex = 0
    xsign = -1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt1[0]
    xtarget = 0
    segments = []
    ptot = [] 
    pout = 0 # power lost in the volume of the divertor (with int prad dV rather than psurf dS)
    areas = []
    # Inner leg entrance
    pxs = []
    ix = ixpt-1
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        ptot.append(xsign*pwrx[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pent = sum(pxs)
    v1, v2 = analysis.cellFaceVertices('E', ix, com.iysptrx)
    rcenter, zcenter = v1
    rcenter += offset
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='bottom', size=fontsize, bbox=bdict))
    # Inner leg common flux
    pys = []
    pbinds = []
    iy = com.ny
    for ix in range(1, ixpt):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(pwry[ix, iy])
        pbinds.append(bbb.fniy[ix, com.ny,0]*bbb.ebind*bbb.ev)
        ptot.append(pwry[ix, iy]
                    +bbb.fniy[ix, com.ny,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_wallh[ix]*com.sy[ix, iy]
                    +bbb.pwr_wallz[ix]*com.sy[ix, iy])
        areas.append(com.sy[ix, iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[1,iy,0]+com.rm[ixpt-1,iy,0])/2
    zcenter = (com.zm[1,iy,0]+com.zm[ixpt-1,iy,0])/2+offset
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='bottom', size=fontsize, bbox=bdict))
    # Inner leg private flux
    pys = []
    pbinds = []
    iy = 0
    for ix in range(1, ixpt):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(-pwry[ix,iy])
        pbinds.append(-bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(-pwry[ix,iy]
                    -bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_pfwallh[ix,0]*com.sy[ix,iy]
                    +bbb.pwr_pfwallz[ix,0]*com.sy[ix,iy])
        areas.append(com.sy[ix, iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[1,iy,0]+com.rm[ixpt-1,iy,0])/2
    zcenter = (com.zm[1,iy,0]+com.zm[ixpt-1,iy,0])/2-offset
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='top', size=fontsize, bbox=bdict))
    # Inner target
    pxs = []
    pbinds = []
    ix = 0
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        pbinds.append(xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(xsign*pwrx[ix,iy]
                    +xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_plth[iy,plateIndex]*com.sxnp[ix,iy]
                    +bbb.pwr_pltz[iy,plateIndex]*com.sxnp[ix,iy])
        areas.append(com.sxnp[ix, iy])
    pout += sum(pxs) + sum(pbinds)
    rcenter = com.rm[ix,com.iysptrx,0]-offset
    zcenter = com.zm[ix,com.iysptrx,0]
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6) + '\n' + r'$P_{\parallel bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='top', size=fontsize, bbox=bdict))
    # Total power
    segments = np.array(segments)
    ptot = np.array(ptot)
    areas = np.array(areas)
    ptot = ptot/areas
    summ = np.sum(ptot*areas)-pent
    # Total radiation
    prad = np.sum((bbb.erliz+bbb.erlrc)[1:ixpt,1:com.ny+1])
    if bbb.isimpon != 0:
        irad = np.sum((bbb.prad*com.vol)[1:ixpt,1:com.ny+1])
    else:
        irad = 0
    pout += prad + irad
    text = '\n'.join([r'$\int p_{H\ rad}\ dV=$%.2g MW' % (prad/1e6), 
                      r'$\int p_{imp\ rad}\ dV=$%.2g MW' % (irad/1e6),
                      r'$P_{\parallel\ in}=$%.2g MW' % (pent/1e6),
                      r'$P_{out}=$%.2g MW' % (pout/1e6)])
    plt.text(0.03, 0.03, text, fontsize=fontsize, c=c, bbox=bdict,
                     ha='left', va='bottom', 
                     transform=ax.transAxes)
    # Plot line collection
    norm = matplotlib.colors.LogNorm()
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ptot/1e6)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)
    plt.colorbar(line, ax=ax, orientation='horizontal', pad=0, label=r'$q_{surf\ tot}$ [MW/m$^2$]')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('gray')
    ax.grid(False)
    plt.axis('equal')
    plt.title('Inner leg')
    plt.xlabel(r'$R$ [m]')
    plt.ylabel(r'$Z$ [m]')
    # Scale view to include labels that might otherwise be cut off
    plt.draw()
    for label in labels:
        bbox = label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bbox_data = bbox.transformed(ax.transData.inverted()).padded(.05)
        ax.update_datalim(bbox_data.corners())
    ax.autoscale_view()

    # Outer target total qsurf
    ax = plt.subplot(224)
    labels = []
    plateIndex = 1
    xsign = 1 # poloidal fluxes are measured on east face of cell
    ixpt = com.ixpt2[0]
    xlegEntrance = ixpt+1
    xtarget = com.nx
    xleg = slice(ixpt+1, xtarget+1)
    segments = []
    ptot = []
    pout = 0 # power lost in the volume of the divertor (with int prad dV rather than psurf dS)
    areas = []
    # Outer leg entrance
    pxs = []
    ix = ixpt+1
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        ptot.append(xsign*pwrx[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pent = sum(pxs)
    rcenter = (com.rm[ix,1,0]+com.rm[ix,com.ny,0])/2
    zcenter = (com.zm[ix,1,0]+com.zm[ix,com.ny,0])/2
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='center', va='bottom', size=fontsize, bbox=bdict))
    # Outer leg common flux
    pys = []
    pbinds = []
    iy = com.ny
    for ix in range(ixpt+2, com.nx+1):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(pwry[ix,iy])
        pbinds.append(bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(pwry[ix,iy]
                    +bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_wallh[ix]*com.sy[ix,iy]
                    +bbb.pwr_wallz[ix]*com.sy[ix,iy])
        areas.append(com.sy[ix,iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[ixpt+2,iy,0]+com.rm[com.nx,iy,0])/2+offset
    zcenter = (com.zm[ixpt+2,iy,0]+com.zm[com.nx,iy,0])/2
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='left', va='bottom', size=fontsize, bbox=bdict))
    # Outer leg private flux
    pys = []
    pbinds = []
    iy = 0
    for ix in range(ixpt+2, com.nx+1):
        segments.append(analysis.cellFaceVertices('N', ix, iy))
        pys.append(-pwry[ix,iy])
        pbinds.append(-bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(-pwry[ix,iy]
                    -bbb.fniy[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_pfwallh[ix,0]*com.sy[ix,iy]
                    +bbb.pwr_pfwallz[ix,0]*com.sy[ix,iy])
        areas.append(com.sy[ix,iy])
    pout += sum(pys) + sum(pbinds)
    rcenter = (com.rm[ixpt+2,iy,0]+com.rm[com.nx,iy,0])/2-offset
    zcenter = (com.zm[ixpt+2,iy,0]+com.zm[com.nx,iy,0])/2
    text = r'$P_{\perp ei}=$%.2g MW' % (sum(pys)/1e6) + '\n' + r'$P_{\perp bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='right', va='top', size=fontsize, bbox=bdict))
    # Outer target
    pxs = []
    pbinds = []
    ix = com.nx
    for iy in range(1, com.ny+1):
        segments.append(analysis.cellFaceVertices('E', ix, iy))
        pxs.append(xsign*pwrx[ix,iy])
        pbinds.append(xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev)
        ptot.append(xsign*pwrx[ix,iy]
                    +xsign*bbb.fnix[ix,iy,0]*bbb.ebind*bbb.ev
                    +bbb.pwr_plth[iy,plateIndex]*com.sxnp[ix,iy]
                    +bbb.pwr_pltz[iy,plateIndex]*com.sxnp[ix,iy])
        areas.append(com.sxnp[ix,iy])
    pout += sum(pxs) + sum(pbinds)
    rcenter = com.rm[ix+1,1,0]
    zcenter = com.zm[ix+1,1,0]-offset
    text = r'$P_{\parallel ei}=$%.2g MW' % (sum(pxs)/1e6) + '\n' + r'$P_{\parallel bind}=$%.2g MW' % (sum(pbinds)/1e6)
    labels.append(plt.annotate(text, (rcenter, zcenter), c=c, ha='center', va='top', size=fontsize, bbox=bdict))
    # Total power
    segments = np.array(segments)
    ptot = np.array(ptot)
    areas = np.array(areas)
    ptot = ptot/areas
    summ = np.sum(ptot*areas)-pent
    # Total radiation
    prad = np.sum((bbb.erliz+bbb.erlrc)[ixpt+2:com.nx+1,1:com.ny+1])
    if bbb.isimpon != 0:
        irad = np.sum((bbb.prad*com.vol)[ixpt+2:com.nx+1,1:com.ny+1])
    else:
        irad = 0
    pout += prad + irad
    text = '\n'.join([r'$\int p_{H\ rad}\ dV=$%.2g MW' % (prad/1e6), 
                      r'$\int p_{imp\ rad}\ dV=$%.2g MW' % (irad/1e6),
                      r'$P_{\parallel\ in}=$%.2g MW' % (pent/1e6),
                      r'$P_{out}=$%.2g MW' % (pout/1e6)])
    plt.text(0.03, 0.03, text, fontsize=fontsize, color=c, ha='left', va='bottom', transform=ax.transAxes, bbox=bdict)
    # Plot line collection
    norm = matplotlib.colors.LogNorm()
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ptot/1e6)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)
    plt.colorbar(line, ax=ax, orientation='horizontal', pad=0, label=r'$q_{surf\ tot}$ [MW/m$^2$]')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('gray')
    ax.grid(False)
    plt.axis('equal')
    plt.title('Outer leg')
    plt.xlabel(r'$R$ [m]')
    plt.ylabel(r'$Z$ [m]')
    # Scale view to include labels that might otherwise be cut off
    plt.draw()
    for label in labels:
        bbox = label.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bbox_data = bbox.transformed(ax.transData.inverted()).padded(0)
        ax.update_datalim(bbox_data.corners())
    ax.autoscale_view()
    
    
def plotPowerBalance(patches):
    args = {'patches': patches, 'rzlabels': False, 'show': False}
    argsBal = {'patches': patches, 'rzlabels': False, 'show': False, 'sym': True}
    plt.subplot(331)
    summ = np.sum(bbb.erliz/1e6)
    plotvar(bbb.erliz/com.vol/1e6, title=r'$P_{rad\ ioniz}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(332)
    summ = np.sum(bbb.erlrc/1e6)
    plotvar(bbb.erlrc/com.vol/1e6, title=r'$P_{rad\ recomb}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    plt.subplot(333)
    if bbb.isimpon != 0:
        summ = np.sum(bbb.prad/1e6*com.vol)
        plotvar(bbb.prad/1e6, title=r'$P_{rad\ imp}$ [MW/m$^3$]', message='$\int$dV = %.2g MW' % summ, log=True, minratio=1e3, **args)
    else:
        plt.axis('off')
    plt.subplot(334)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourcePoloidal(bbb.feex+bbb.feix, ix, iy))/com.vol/1e6, title=r'$P_{poloidal}$ [MW/m$^3$]', log=True, **argsBal)
    # sumAbs = analysis.gridPowerSumAbs()
    # plotvar(-bbb.erliz/sumAbs, title=r'$\frac{P_{rad\ ioniz}}{\Sigma_j|P_j|}$', **argsBal)
    plt.subplot(335)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourceRadial(bbb.feey+bbb.feiy, ix, iy))/com.vol/1e6, title=r'$P_{radial}$ [MW/m$^3$]', log=True, **argsBal)
    # plotvar(-bbb.erlrc/sumAbs, title=r'$\frac{P_{rad\ recomb}}{\Sigma_j|P_j|}$', **argsBal)
    # plt.subplot(336)
    # if bbb.isimpon != 0:
    #     plotvar(-bbb.prad/sumAbs, title=r'$\frac{P_{rad\ imp}}{\Sigma_j|P_j|}$', **argsBal)
    # else:
    #     plt.axis('off')
    # plt.subplot(337)
    # plt.subplot(338)
    plt.subplot(337)
    plotvar(analysis.PionParallelKE()/com.vol/1e6, title=r'$P_{ion\ KE}$ [MW/m$^3$]', log=True, **argsBal)
    plt.subplot(339)
    plotvar(analysis.gridPowerBalance()/com.vol/1e6, title=r'Power balance [MW/m$^3$]', log=True, **argsBal)
    
    
def plotDensityBalance(patches):
    args = {'patches': patches, 'rzlabels': False, 'show': False, 'sym': True}
    plt.subplot(321)
    #sumAbs = analysis.gridParticleSumAbs()
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourcePoloidal(bbb.fnix[:,:,0], ix, iy)), title=r'Poloidal source [s$^{-1}$]', log=True, **args)
    plt.subplot(322)
    plotvar(analysis.toGrid(lambda ix, iy: analysis.cellSourceRadial(bbb.fniy[:,:,0], ix, iy)), title=r'Radial source [s$^{-1}$]', log=True, **args)
    plt.subplot(323)
    plotvar(bbb.psor[:,:,0], title=r'Ionization source [s$^{-1}$]', log=True, **args)
    plt.subplot(324)
    plotvar(-bbb.psorrg[:,:,0], title=r'Recombination source [s$^{-1}$]', log=True, **args)
    plt.subplot(325)
    plotvar(analysis.gridParticleBalance(), title=r'Particle balance [s$^{-1}$]', log=True, **args)
    
    
def plotRadialFluxes():
    fniy = np.zeros((com.nx+2,com.ny+2))
    fniydd = np.zeros((com.nx+2,com.ny+2))
    fniydif = np.zeros((com.nx+2,com.ny+2))
    fniyconv = np.zeros((com.nx+2,com.ny+2))
    fniyef = np.zeros((com.nx+2,com.ny+2))
    fniybf = np.zeros((com.nx+2,com.ny+2))
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniy[ix,iy] = bbb.cnfy*bbb.vy[ix,iy,0]*com.sy[ix,iy]*t2
            fniydd[ix,iy] = bbb.vydd[ix,iy,0]*com.sy[ix,iy]*t2
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
            fniyconv[ix,iy] = vyconv[ix,iy]*com.sy[ix,iy]*t2
            fniyef[ix,iy] = bbb.cfyef*bbb.vyce[ix,iy,0]*com.sy[ix,iy]*t2
            fniybf[ix,iy] = bbb.cfybf*bbb.vycb[ix,iy,0]*com.sy[ix,iy]*t2
            if bbb.vy[ix,iy,0]*(bbb.ni[ix,iy,0]-bbb.ni[ix,iy+1,0]) < 0:
                fniy[ix,iy] = fniy[ix,iy]/(1+(bbb.nlimiy[0]/bbb.ni[ix,iy+1,0])**2+(bbb.nlimiy[0]/bbb.ni[ix,iy,0])**2) # nlimiy is 0 rn... might help with converg. problems in SPARC

    def upwind(f, p1, p2): 
        return max(f,0)*p1+min(f,0)*p2

    def upwindProxy(f, g, p1, p2):
        return max(f,0)/f*g*p1+min(f,0)/f*g*p2

    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fngy[ix,iy,0]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
    feey = econd+econv # should match bbb.feey
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy

    ix = com.isixcore == 1

    plt.subplot(311)
    plt.plot(com.yyc*1000, np.sum(bbb.fniy[ix,:,0],axis=0), c='k', label=r'Total ion flux');
    plt.plot(com.yyc*1000, np.sum(fniydif[ix,:],axis=0), label=r'Diffusion', c='C0')
    #plt.plot(com.yyc*1000, (-bbb.dif_use[:,:,0]*(bbb.niy1[:,:,0]-bbb.niy0[:,:,0])*com.gyf*com.sy)[ix,:], label=r'$-D(\nabla n) A_y$',c='C0',ls='--');
    plt.plot(com.yyc*1000, np.sum(fniyconv[ix,:],axis=0), label=r'Convection', c='C1')
    #plt.plot(com.yyc*1000, (vyconv*bbb.ni[:,:,0]*com.sy)[ix,:], label=r'$v_{conv}n_iA_y$',c='C1',ls='-');
    plt.plot(com.yyc*1000, np.sum(fniyef[ix,:],axis=0), label=r'$E\times B$ convection',c='C7',ls='--')
    plt.plot(com.yyc*1000, np.sum(fniybf[ix,:],axis=0), label=r'$\nabla B$ convection',c='C3',ls='--')
    plt.plot(com.yyc*1000, np.sum((bbb.fngy[:,:,0])[ix,:],axis=0), label='(Neutral flux)',c='C4',ls='-.')
    plt.plot(com.yyc*1000, np.sum((fniydif+fniyconv+fniyef+fniybf)[ix,:],axis=0), label='Sum of components',c='k',ls=':')
    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Flux [s$^{-1}$]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.title('Sum over core poloidal cells\n')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.subplot(312)
    mytot = icond+ncond+iconv+nconv
    plt.plot(com.yyc*1000, np.sum(bbb.feiy[ix,:]/1e6,axis=0), c='k', label='i+n conv.+cond.');
    plt.plot(com.yyc*1000, np.sum(icond[ix,:]/1e6,axis=0), ls='-', label='Ion conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(ncond[ix,:]/1e6,axis=0), ls='--', label='Neutral conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(iconv[ix,:]/1e6,axis=0), ls='-', label='Ion convection', c='C1');
    plt.plot(com.yyc*1000, np.sum(nconv[ix,:]/1e6,axis=0), ls='--', label='Neutral convection', c='C1');
    plt.plot(com.yyc*1000, np.sum(mytot[ix,:]/1e6,axis=0), c='k', ls=':', label='Sum of components')

    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Power [MW]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)

    plt.subplot(313)

    plt.plot(com.yyc*1000, np.sum(bbb.feey[ix,:]/1e6,axis=0), c='k', label='Electron conv.+cond.');
    plt.plot(com.yyc*1000, np.sum(econd[ix,:]/1e6,axis=0), ls='-', label='Electron conduction', c='C0');
    plt.plot(com.yyc*1000, np.sum(econv[ix,:]/1e6,axis=0), ls='-', label='Electron convection', c='C1');
    plt.plot(com.yyc*1000, np.sum((econd+econv)[ix,:]/1e6,axis=0), c='k', ls=':', label='Sum of components')

    ylim = plt.gca().get_ylim()
    maxabs = np.max(np.abs(ylim))
    plt.ylim([-maxabs, maxabs])
    plt.ylabel('Power [MW]')
    plt.xlabel(r'$R-R_{sep}$ [mm]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.grid(True, which='both', axis='y', color='#ddd'); plt.gca().set_axisbelow(True);
    for i in com.yyc*1000: plt.axvline(i, c='#ddd', lw=0.5, zorder=0)
    

def getThetaHat(ix, iy):
    p1R = com.rm[ix,iy,1]
    p1Z = com.zm[ix,iy,1]
    p2R = com.rm[ix,iy,2]
    p2Z = com.zm[ix,iy,2]
    dR = p2R-p1R
    dZ = p2Z-p1Z
    mag = (dR**2+dZ**2)**.5
    return dR/mag, dZ/mag
    
    
def getrHat(ix, iy):
    dR = com.rm[ix,iy,2]-com.rm[ix,iy,1]
    dZ = com.zm[ix,iy,2]-com.zm[ix,iy,1]
    mag = (dR**2+dZ**2)**.5
    return -dZ/mag, dR/mag    

    
def getDriftVector(v2, vy, ix, iy):
    v2 = -np.sign(bbb.b0)*v2
    return v2[ix,iy]*(1-com.rr[ix,iy]**2)**.5*np.array(getThetaHat(ix, iy))+vy[ix,iy]*np.array(getrHat(ix, iy))


def getDriftVectorLog(v2, vy, ix, iy):
    v = getDriftVector(v2, vy, ix, iy)
    vmag = (v[0]**2+v[1]**2)**.5
    v = np.log(vmag)*v/vmag
    if len(v[~np.isfinite(v)]) > 0:
        return np.array([0, 0])
    return v


def getDriftR(v2, vy, ix, iy):
    return getDriftVectorLog(v2, vy, ix, iy)[0]
    
    
def getDriftZ(v2, vy, ix, iy):
    return getDriftVectorLog(v2, vy, ix, iy)[1]
    
    
def plotImps(show=True):
    plt.figure(figsize=(10,10))
    imps = bbb.ni.shape[2]-2
    ncols = nrows = int((imps+1)**.5+.5)
    nimp = np.zeros((com.nx+2,com.ny+2,imps+1))
    nimp[:,:,0] = bbb.ng[:,:,1]
    nimp[:,:,1:] = bbb.ni[:,:,2:]
    plt.subplot(nrows, ncols, 1)
    vmin = np.min(analysis.nonGuard(nimp))
    vmax = np.max(analysis.nonGuard(nimp))
    for i in range(0, imps+1):
        plt.subplot(nrows, ncols, i+1)
        plotvar(nimp[:,:,i], log=True, rzlabels=False, show=False, title='Impurity +%d [m$^{-3}$]' % i, vmin=vmin, vmax=vmax)
    if show:
        plt.tight_layout()
        plt.show()
    
    
def plotDrift(v2, vy):
    args = {'width': .0015, 'alpha': 1}
    sepwidth = .5
    bdry = []
    bdry.extend([list(zip(com.rm[0,iy,[1,2,4,3,1]],com.zm[0,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[com.nx+1,iy,[1,2,4,3,1]],com.zm[com.nx+1,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[ix,0,[1,2,4,3,1]],com.zm[ix,0,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    bdry.extend([list(zip(com.rm[ix,com.ny+1,[1,2,4,3,1]],com.zm[ix,com.ny+1,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    dR = analysis.toGrid(lambda ix, iy: getDriftR(v2, vy, ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(v2, vy, ix, iy))
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim([np.min(com.zm), com.zm[com.ixpt1[0],com.iysptrx+1,0]/2])
    
    
def plotDrifts(patches):
    bdry = []
    bdry.extend([list(zip(com.rm[0,iy,[1,2,4,3,1]],com.zm[0,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[com.nx+1,iy,[1,2,4,3,1]],com.zm[com.nx+1,iy,[1,2,4,3,1]])) 
                  for iy in np.arange(0,com.ny+2)])
    bdry.extend([list(zip(com.rm[ix,0,[1,2,4,3,1]],com.zm[ix,0,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    bdry.extend([list(zip(com.rm[ix,com.ny+1,[1,2,4,3,1]],com.zm[ix,com.ny+1,[1,2,4,3,1]])) 
                  for ix in np.arange(0,com.nx+2)])
    args = {'width': .001, 'alpha': 1}
    ylim = [np.min(com.zm), com.zm[com.ixpt1[0],com.iysptrx+1,0]/2]
    sepwidth = .5
    vtot = (bbb.v2ce[:,:,0]**2+bbb.vyce[:,:,0]**2)**.5+(bbb.v2cb[:,:,0]**2+bbb.vycb[:,:,0]**2)**.5+(bbb.v2dd[:,:,0]**2+bbb.vydd[:,:,0]**2)**.5
    # Total drifts
    plt.subplot(221)
    plt.title('Sum of all drifts')
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2[:,:,0], bbb.vy[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2[:,:,0], bbb.vy[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # ExB
    plt.subplot(222)
    percent = np.mean(analysis.nonGuard((bbb.v2ce[:,:,0]**2+bbb.vyce[:,:,0]**2)**.5/vtot))*100
    plt.title(r'$\mathbf{E}\times \mathbf{B}$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2ce[:,:,0], bbb.vyce[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2ce[:,:,0], bbb.vyce[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # Grad B
    plt.subplot(223)
    percent = np.mean(analysis.nonGuard((bbb.v2cb[:,:,0]**2+bbb.vycb[:,:,0]**2)**.5/vtot))*100
    plt.title(r'$\mathbf{B}\times\nabla B$ drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.ve2cb[:,:], bbb.veycb[:,:], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.ve2cb[:,:], bbb.veycb[:,:], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='electrons', color='C1', **args)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cb[:,:,0], bbb.vycb[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cb[:,:,0], bbb.vycb[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    # plt.legend()
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # # Diamagnetic/grad P x B
    # plt.subplot(324)
    # percent = np.mean(analysis.nonGuard((bbb.v2cd[:,:,0]**2+bbb.vycp[:,:,0]**2)**.5/vtot))*100
    # plt.title(r'Diamagnetic drift ($\nabla P\times \mathbf{B}$) (%.2g%%)' % percent)
    # plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    # plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.ve2cd[:,:], bbb.veycp[:,:], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.ve2cd[:,:], bbb.veycp[:,:], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='electrons', color='C1', **args)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2cd[:,:,0], bbb.vycp[:,:,0], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2cd[:,:,0], bbb.vycp[:,:,0], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], label='ions', **args)
    # plt.legend()
    # plt.axis('equal')
    # plt.xlim([.45, .75])
    # plt.ylim([-.6, -.2])
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # # Resistive drift
    # plt.subplot(325)
    # percent = np.mean(analysis.nonGuard((bbb.v2rd[:,:,0]**2+bbb.vyrd[:,:,0]**2)**.5/vtot))*100
    # plt.title('Resistive drift (%.2g%%)' % percent)
    # plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    # plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    # dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2rd[:,:,0], bbb.vyrd[:,:,0], ix, iy))
    # dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2rd[:,:,0], bbb.vyrd[:,:,0], ix, iy))
    # plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    # plt.axis('equal')
    # plt.xlim([.45, .75])
    # plt.ylim([-.6, -.2])
    # plt.gca().axes.get_xaxis().set_visible(False)
    # plt.gca().axes.get_yaxis().set_visible(False)
    # Anomalous drift
    plt.subplot(224)
    percent = np.mean(analysis.nonGuard((bbb.v2dd[:,:,0]**2+bbb.vydd[:,:,0]**2)**.5/vtot))*100
    plt.title('Anomalous drift (%.2g%%)' % percent)
    plt.gca().add_collection(LineCollection(bdry, linewidths=.2, color='black', zorder=0))
    plt.plot(com.rm[:,com.iysptrx+1,2], com.zm[:,com.iysptrx+1,2], c='red', zorder=0, lw=sepwidth)
    dR = analysis.toGrid(lambda ix, iy: getDriftR(bbb.v2dd[:,:,0], bbb.vydd[:,:,0], ix, iy))
    dZ = analysis.toGrid(lambda ix, iy: getDriftZ(bbb.v2dd[:,:,0], bbb.vydd[:,:,0], ix, iy))
    plt.quiver(com.rm[1:-1,1:-1,0], com.zm[1:-1,1:-1,0], dR[1:-1,1:-1], dZ[1:-1,1:-1], **args)
    plt.axis('equal')
    # plt.xlim([.45, .75])
    plt.ylim(ylim)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    
    
def finishPage(pdf):
    plt.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
    
def toPlate(arr):
    """
    Poloidal flux in direction of closest divertor plate, through the face
    nearest that plate.
    
    Arguments
        arr: numpy array with dimension (nx, ny)
    """
    arr[:bbb.ixmp, :] *= -1
    if ix <= bbb.ixmp - 1:
        sign = -1
        ix = ix - 1
    else:
        sign = 1
    return sign*arr[ix, ]


def plotall(savefile='plots', plotV0=True):    
    savefile = savefile + '.pdf'
    plt.close()
    plt.rcParams['font.size'] = 12
    
    targetDataFiles = glob.glob("targetData_*")
    if targetDataFiles:
        h5 = h5py.File(targetDataFiles[0], 'r')
        plotV0 = False
    else:
        h5 = None
    
    # Calculate photon power fluxes (required for some plots)
    bbb.pradpltwl()
    
    patches = getPatches()
    
    with PdfPages(savefile) as pdf:
        # Page 1: input text and chi, D graphs
        fig = plt.figure(figsize=(8.5, 11))
        txt = getConfigText()
        plt.subplot(311)
        plt.axes(frameon=False)
        fig.text(0.1, 0.95, txt, transform=fig.transFigure, size=9, horizontalalignment='left', verticalalignment='top')
        plotTransportCoeffs(patches)
        finishPage(pdf)
        
        # Page 2: n and T 2D plots
        plt.figure(figsize=(8.5, 11))
        plot2Dvars(patches)
        finishPage(pdf)
        
        # Page 3: n and T line profiles
        plt.figure(figsize=(8.5, 11))
        plotnTprofiles(plotV0, h5)
        pdf.savefig()
        plt.close()
        # finishPage(pdf)
        
        # Page 4: n, T, Prad going down legs
        plt.figure(figsize=(8.5, 11))
        plotAlongLegs()
        finishPage(pdf)
        
        # Page 5: pressure, lamda_q
        plt.figure(figsize=(8.5, 11))
        plotPressures()
        plotqFits(h5)
        finishPage(pdf)

        # Page 6: power breakdown
        fig = plt.figure(figsize=(8.5, 11))
        plotPowerBreakdown()
        plotPowerSurface()
        finishPage(pdf)
        
        # Page 7: radiation balance 2D plots
        plt.figure(figsize=(8.5, 11))
        plotPowerBalance(patches)
        finishPage(pdf)
        
        # Page 8: density balance 2D plots
        plt.figure(figsize=(8.5, 11))
        plotDensityBalance(patches)
        finishPage(pdf)
        
        # Page 9: Density and power flux at midplane
        plt.figure(figsize=(8.5, 11))
        plotRadialFluxes()
        finishPage(pdf)
        
        # Page 10: drifts
        if bbb.cfjve+bbb.jhswitch+bbb.isfdiax+bbb.cfyef+bbb.cf2ef+bbb.cfybf+bbb.cf2bf+bbb.cfqybf+bbb.cfq2bf > 0:
            plt.figure(figsize=(8.5, 11))
            plotDrifts(patches)
            finishPage(pdf)

        d = pdf.infodict()
        d['Title'] = savefile
        d['CreationDate'] = datetime.datetime.today()
    
    plt.close('all')
