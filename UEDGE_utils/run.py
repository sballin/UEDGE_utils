from copy import copy, deepcopy
import shutil
import sys
import os
import inspect
import time
import datetime
import tempfile
import subprocess
import ast
import glob
from email.mime.text import MIMEText
from distutils.util import strtobool
import h5py
import numpy as np
from scipy.interpolate import griddata, bisplrep, bisplev, interp1d
import scipy.misc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import uedge
from uedge import bbb, com, flx, grd, svr, aph, api
import uedge.hdf5 
import UEDGE_utils.plot
import UEDGE_utils.analysis 

from copy import copy, deepcopy
from IPython.terminal.prompts import Prompts, Token
from IPython import get_ipython
from IPython.core.magic import register_line_magic
ipython = get_ipython()
import uedge
from uedge import bbb, com, flx, grd, svr, aph, api    
    

def myReprefix():
    pkgs = [bbb, com, flx, grd, svr, aph] # api.reprefix() causes segfault for me
    dictBefore = dict()
    for pkg in pkgs:
        dictBefore.update(pkg.getdict())
    dictBefore = deepcopy(dictBefore)
    for pkg in pkgs:
        pkg.reprefix()
    dictAfter = dict()
    for pkg in pkgs:
        dictAfter.update(pkg.getdict())
    compareDicts(dictBefore, dictAfter)


class MyPrompt(Prompts):
    def in_prompt_tokens(self, cli=None):
        try:
            myReprefix()
            uedge.deprefix()
        except Exception as e:
            print(e)
        return [(Token.Prompt, 'UEDGE>>> ')]
        
    def out_prompt_tokens(self, cli=None):
        return [(Token.Prompt, 'UEDGE>>> ')]


def prefixOff():
    uedge.deprefix()
    ipython = get_ipython()
    ipython.prompts = MyPrompt(ipython)
    
    
def split2D(valCore, valSol):
    '''Specify separate values inside/outside separatrix'''
    out = np.zeros((com.nx+2,com.ny+2))
    out[:,:com.iysptrx+1] = valSol
    out[com.isixcore==1,:com.iysptrx+1] = valCore
    out[:,com.iysptrx+1:] = valSol
    return out


def rise2D(vmin, vmax, exponent, startPoloidal):
    out = np.zeros((com.nx+2,com.ny+2))
    out[startPoloidal:,:] = rise1D(vmin, vmax, exponent)
    return out


def rise1D(vmin, vmax, exponent):
    out = np.zeros(com.ny+2)
    scale = (vmax-vmin)/com.yyc[-1]**exponent
    out[:com.iysptrx+1] = vmin
    out[com.iysptrx+1:] = vmin+scale*com.yyc[com.iysptrx+1:]**exponent
    return out


def toDchi2d():
    """Move from fixed D, chi to defining them over 2D grid."""
    d = bbb.difni[0].copy()
    kye = copy(bbb.kye)
    kyi = copy(bbb.kyi)
    bbb.difni = 0.0 # D for radial hydrogen diffusion
    bbb.kye = 0.0 # chi_e for radial elec energy diffusion
    bbb.kyi = 0.0 # chi_i for radial ion energy diffusion
    bbb.isbohmcalc = 0
    bbb.facbee = 1.0 # factor for Bohm Te diffusion coeff 
    bbb.facbei = 1.0 # factor for Bohm Ti diffusion coeff
    bbb.facbni = 1.0 # factor for Bohm ni diffusion coeff
    bbb.dif_use[:,:,0] = d
    bbb.kye_use = kye
    bbb.kyi_use = kyi


def toFixedMean():
    """Fix all boundary values to mean over CF/PF boundary."""
    bbb.isnwcono = bbb.isnwconi = 1
    bbb.nwallo = np.mean(bbb.ni[:,com.ny+1,0])
    bbb.nwalli = np.mean(np.concatenate([bbb.ni[:com.ixpt1[0]+1,0,0], bbb.ni[com.ixpt2[0]+1:,0,0]]))
    bbb.istewc = bbb.istepfc = bbb.istiwc = bbb.istipfc = 1
    bbb.tewallo = np.mean(bbb.te[:,com.ny+1])/bbb.ev
    bbb.tiwallo = np.mean(bbb.ti[:,com.ny+1])/bbb.ev
    bbb.tewalli = np.mean(np.concatenate([bbb.te[:com.ixpt1[0]+1,0], bbb.te[com.ixpt2[0]+1:,0]]))/bbb.ev
    bbb.tiwalli = np.mean(np.concatenate([bbb.ti[:com.ixpt1[0]+1,0], bbb.ti[com.ixpt2[0]+1:,0]]))/bbb.ev
    

def barrier2D(bmax, bmin, bwidth, bcenter, steepness):
    '''
    Return 2D profile with a transport barrier for D or chi. Minimum at separatrix.
    Max everywhere in divertor legs.
    
    bmax: barrier max (background value)
    bmin: barrier min (separatrix value)
    bwidth: barrier full width (in meters)
    bcenter: barrier center (in meters) relative to the separatrix
    steepness: barrier side steepness. Recommend one of [2, 4, 8, 16, 32], 32 being steepest,
               pretty much square.
    '''
    out = np.zeros((com.nx+2, com.ny+2))
    out[:com.ixpt1[0]+1,:] = bmax
    out[com.ixpt2[0]+1:,:] = bmax
    out[com.ixpt1[0]+1:com.ixpt2[0]+1,:] = barrier1D(bmax, bmin, bwidth, bcenter, steepness)
    return out
    
     
def barrier1D(bmax, bmin, bwidth, bcenter, steepness):
    ''' 
    Return midplane profile with a transport barrier for D or chi. Minimum at separatrix.
    
    bmax: barrier max (background value)
    bmin: barrier min (separatrix value)
    bwidth: barrier full width (in meters)
    bcenter: barrier center (in meters) relative to the separatrix
    steepness: barrier side steepness. Recommend one of [2, 4, 8, 16, 32], 32 being steepest,
               pretty much square.
    '''
    reldepth = 1-bmin/bmax
    return bmax - (bmax-bmin)*np.exp(-((com.yyc-bcenter)/(bwidth/2))**steepness)


def interpolateVar(rcOld, zcOld, varOld, rcNew, zcNew, method='nearest'):
    out = griddata((rcOld.flatten(), zcOld.flatten()), varOld.flatten(), (rcNew.flatten(), zcNew.flatten()),
             method=method, fill_value=0)
    return out.reshape((com.nx+2, com.ny+2))
    
    
def interpolateVarNew(varOld, xcsOld, psiOld, ixpt1Old, ixpt2Old):
    '''
    TODO: plates appear to be using nearest neighbor, investigate why
    '''
    out = np.zeros((com.nx+2,com.ny+2))
    if np.all(varOld == 0):
        return out
    xcsNew = com.xcs.copy()
    psiNew = com.psi[:,:,4].copy()

    nxOld = xcsOld.shape[0]-2
    nyOld = psiOld.shape[1]-2
    xcs2Dold = np.zeros((nxOld+2,nyOld+2))
    xcs2Dnew = np.zeros((com.nx+2,com.ny+2))
    for i in range(nyOld+2): xcs2Dold[:,i] = xcsOld.copy()
    for i in range(com.ny+2): xcs2Dnew[:,i] = xcsNew.copy()
    # Normalize axes to help convergence
    xcs2Dnew /= np.max(xcs2Dold)
    xcs2Dold /= np.max(xcs2Dold)
    psiNewNorm = psiNew/np.max(psiOld)
    psiOldNorm = psiOld/np.max(psiOld)
    varOldNorm = varOld/np.max(varOld)

    for ixso, ixeo, ixs, ixe in [(0,ixpt1Old+1,0,com.ixpt1[0]+1),
                                 (ixpt1Old+1,ixpt2Old+1,com.ixpt1[0]+1,com.ixpt2[0]+1),
                                 (ixpt2Old+1,nxOld+2,com.ixpt2[0]+1,com.nx+2)]:
        # Normalize poloidal distance in each leg but not in main chamber
        if ixeo == ixpt1Old+1:
            xcs2DnewNorm = xcs2Dnew-xcs2Dnew[0]
            xcs2DnewNorm /= np.max(xcs2Dnew[com.ixpt1[0]+1])
            xcs2DoldNorm = xcs2Dold-xcs2Dold[0]
            xcs2DoldNorm /= np.max(xcs2Dold[ixpt1Old+1])
        if ixeo == nxOld+2:
            xcs2DnewNorm = xcs2Dnew-xcs2Dnew[com.ixpt2[0]+1]
            xcs2DnewNorm /= np.max(xcs2DnewNorm[-1])
            xcs2DoldNorm = xcs2Dold-xcs2Dold[ixpt2Old+1]
            xcs2DoldNorm /= np.max(xcs2DoldNorm[-1])
        else:
            xcs2DnewNorm = xcs2Dnew-xcs2Dnew[com.ixpt1[0]+1]
            xcs2DoldNorm = xcs2Dold-xcs2Dold[ixpt1Old+1]
        out[ixs:ixe] = griddata((xcs2DoldNorm[ixso:ixeo].flatten(), psiOldNorm[ixso:ixeo].flatten()), varOldNorm[ixso:ixeo].flatten(), (xcs2DnewNorm[ixs:ixe].flatten(), psiNewNorm[ixs:ixe].flatten()), method='linear', fill_value=np.nan).reshape((ixe-ixs,com.ny+2))
        out[np.isnan(out)] = griddata((xcs2DoldNorm[ixso:ixeo].flatten(), psiOldNorm[ixso:ixeo].flatten()), varOldNorm[ixso:ixeo].flatten(), (xcs2DnewNorm[np.isnan(out)].flatten(), psiNewNorm[np.isnan(out)].flatten()), method='nearest')
        
    out *= np.max(varOld)
    return np.clip(out, np.min(varOld), np.max(varOld))
    

def setGridfile(filename, interpolate=False):
    if interpolate:
        xcsOld = com.xcs.copy()
        psiOld = com.psi[:,:,4].copy()
        ixpt1Old = com.ixpt1[0].copy()
        ixpt2Old = com.ixpt2[0].copy()
        saveVars = ['dif_use','vy_use','kye_use','kyi_use','afracs','ngs','ng','ni','nis','phi','phis','te','tes','ti','tis','up','ups','tg','tgs']
        saveVals = {s: getattr(bbb, s).copy() for s in saveVars}
    # Save fixed BC values because they can get messed up by a grid change
    tewallo = bbb.tewallo.copy()
    tiwallo = bbb.tiwallo.copy()
    tewalli = bbb.tewalli.copy()
    tiwalli = bbb.tiwalli.copy()
    # Remove existing gridue file if it exists and symlink gridue to new grid
    try:
        os.remove('gridue')
    except:
        pass
    os.symlink(filename, 'gridue')
    bbb.newgeo = 1 # read gridue on next exmain
    # Interpolate D, v, chi onto new grid
    if interpolate:
        init()
        args = [xcsOld, psiOld, ixpt1Old, ixpt2Old]
        for k in saveVals.keys():
            v = saveVals[k]
            if len(v.shape) == 3:
                newv = np.zeros((com.nx+2,com.ny+2,v.shape[2]))
                for i in range(v.shape[2]):
                    newv[:,:,i] = interpolateVarNew(v[:,:,i], *args)
            else:
                newv = interpolateVarNew(v, *args)
            setattr(bbb, k, newv)
    # Restore fixed BC values
    bbb.tewallo = np.mean(tewallo)
    bbb.tiwallo = np.mean(tiwallo)
    bbb.tewalli = np.mean(tewalli)
    bbb.tiwalli = np.mean(tiwalli)
    
    
def changeGrid(gridfile):
    '''Quickly change to a different grid and get com variables, useful for analysis'''
    setGridfile(gridfile, interpolate=False)
    bbb.newgeo = 1
    bbb.allocate()
    bbb.nphygeo()


def writeGrid():
    bbb.newgeo = 1 # 1 = calculate new grid
    bbb.gengrid = 1 # 1 = generate grid, 0 = read from file gridue
    flx.flxrun() #  main driver routine for the flx package
    grd.grdrun() # main driver routine for grd package
    bbb.nphygeo() # define geometry by reading grid info from file


def prepGrid(interpolate=True, method='nearest'):
    """
    Write new grid file and check that grid cells are valid polygons.
    
    Args
        interpolate: do 2D linear interpolation of D, v, and chi onto new grid
        method: 'nearest' can produce jagged results but is robust including for extrapolation,
                'linear' can be smoother but can't extrapolate and can mess up interpolation
    """
    if interpolate:
        rcOld = com.rm[:,:,0].copy()
        zcOld = com.zm[:,:,0].copy()
        DOld = bbb.dif_use[:,:,0].copy()
        vOld = bbb.vy_use[:,:,0].copy()
        kyeOld = bbb.kye_use.copy()
        kyiOld = bbb.kyi_use.copy()
        afracsOld = bbb.afracs.copy()
       
    # Generate new grid 
    bbb.isnintp = 1 # 1 = use new interpolation, 0 can result in negative density error at exmain()
    writeGrid()
    
    # Exmain to finish initialization of new grid
    bbb.icntnunk = 0
    bbb.itermx = 1
    bbb.dtreal = 1e-10
    bbb.ftol = 1e20
    bbb.exmain()
    
    # Interpolate D, v, chi onto new grid
    if interpolate:
        rc = com.rm[:,:,0].copy()
        zc = com.zm[:,:,0].copy()
        bbb.dif_use[:,:,0] = interpolateVar(rcOld, zcOld, DOld, rc, zc, method=method)
        bbb.vy_use[:,:,0] = interpolateVar(rcOld, zcOld, vOld, rc, zc, method=method)
        bbb.kye_use = interpolateVar(rcOld, zcOld, kyeOld, rc, zc, method=method)
        bbb.kyi_use = interpolateVar(rcOld, zcOld, kyiOld, rc, zc, method=method)
        bbb.afracs = interpolateVar(rcOld, zcOld, afracsOld, rc, zc, method=method)
        
    print('Bad cells:', len(UEDGE_utils.analysis.badCells()))
    print('Overlapping cells:', len(UEDGE_utils.analysis.overlappingCells()))


def setImpurity(impFrac=None):
    '''
    TODO
    Create the appropriate mist.dat symlink if necessary.
    '''
    if impFrac == None:
        bbb.isimpon = 0
        bbb.afracs = 0
    else:
        bbb.isimpon = 2
        bbb.afracs = impFrac
        bbb.allocate()


def setInertialNeutrals():
    com.nhsp = 2 # number of hydrogenic species
    bbb.ziin[1] = 0. # ion charge 
    bbb.isngon[0] = 0 # turn off neutral diffusion equation
    bbb.isupgon[0] = 1 # turn on parallel neutral velocity equation
    bbb.cngmom = 0 # mom. cx-loss coeff for diffusve-neut hydr only
    bbb.cmwall = 0 # mom. wall-loss coeff for diff-neut hydr only
    bbb.cngtgx = 0 # X-flux coef for gas comp. of Ti eqn.
    bbb.cngtgy = 0 # Y-flux coef for gas comp. of Ti eqn.
    bbb.cfbgt = 0 # Coef for the B x Grad(T) terms.
    bbb.kxn = 0 # poloidal cx-neutral heat diff. factor
    bbb.kyn = 0 # radial cx-neutral heat diff. factor
    bbb.allocate()
    bbb.nis[:,:,1] = bbb.ngs[:,:,0].copy()

    
def setDiffusiveNeutrals():
    bbb.isngon[0] = 1
    bbb.isupgon[0] = 0
    com.nhsp = 1


def setFluxLimits(on=True):
    if on:
        bbb.flalfe = 0.21 # electron parallel thermal conduct. coeff
        bbb.flalfi = 0.21 # ion parallel thermal conduct. coeff
        bbb.flalfv = 1.   # ion parallel viscosity coeff
        bbb.flalfgx = 1.  # neut. gas in poloidal direction
        bbb.flalfgy = 1.  # neut. gas in radial direction
        bbb.flalfgxy = 1. # nonorthog pol-face gas flux limit
        bbb.flalftgx = 1. # neut power in poloidal direction
        bbb.flalftgy = 1. # neut power in radial direction
        bbb.lgmax = 0.05  # max scale for gas particle diffusion
        bbb.lgtmax = 0.05 # max scale for gas thermal diffusion
    else:
        bbb.flalfe = 1e20 # electron parallel thermal conduct. coeff
        bbb.flalfi = 1e10 # ion parallel thermal conduct. coeff
        bbb.flalfv = 1e10 # ion parallel viscosity coeff
        bbb.flalfgx = 1.  # neut. gas in poloidal direction
        bbb.flalfgy = 1.  # neut. gas in radial direction
        bbb.flalfgxy = 1. # nonorthog pol-face gas flux limit
        bbb.flalftgx = 1. # neut power in poloidal direction
        bbb.flalftgy = 1. # neut power in radial direction 
        bbb.lgmax = 1e20  # max scale for gas particle diffusion
        bbb.lgtmax = 1e20 # max scale for gas ther
        
        
def setPhi():
    """Enable the potential equation with drifts off"""
    bbb.b0 = 1.         # =1 for normal direction of B-field
    bbb.isphion = 1     # turn on potential equation
    bbb.rsigpl = 1.e-8  # anomalous cross-field conductivity
    bbb.newbcl = 1      # Sheath BC [bee,i] from current equation
    bbb.newbcr = 1      # Sheath BC [bee,i] from current equation
    bbb.isnewpot = 1    # 1=new potential; J_r from tor. mom. bal, -2=phi constant on core boundary with total core current = icoreelec
    bbb.rnewpot = 1.    # mixture of fqy=(1-rnewpot)*fqy_old+rnewpot*fqy_new (fqy = net radial current north)
    bbb.iphibcc = 1     # set bbb.phi[,1] uniform poloidally 
    
    bbb.cfjpy = 0.      # diamag. cur. in flx.y-direction
    bbb.cfjp2 = 0.      # diamag. cur. in 2-direction
    bbb.cfydd = 0.      # Diamag. drift in flx.y-dir. [always=0]
    bbb.cf2dd = 0.      # Diamag. drift in 2-dir. [always=0]
    bbb.cfrd = 0.       # Resistive drift in flx.y and 2 dirs.
    bbb.cfbgt = 0.      # Diamag. energy drift [always=0]
    
    bbb.cfjhf = 0.      # turn-on heat flow from current [bbb.fqp]
    bbb.cfjve = 0.      # makes bbb.vex = vix - bbb.cfjve*bbb.fqx
    bbb.jhswitch = 0    # Joule Heating switch
    bbb.isfdiax = 0.    # Factor to turn on diamag. contrib. to sheath
    bbb.cfyef = 0       # ExB drift in flx.y-dir.
    bbb.cf2ef = 0       # ExB drift in 2-dir.
    bbb.cfybf = 0.      # turns on bbb.vycb - radial grad_B drift
    bbb.cf2bf = 0.      # turns on bbb.v2cb - perp grad_B drift [nearly pol]
    bbb.cfqybf = 0.     # turns on bbb.vycb contrib to radial current
    bbb.cfq2bf = 0.     # turns on bbb.v2cb contrib to perp["2"] current


def setDrifts(b0, c=1, rog=False):
    """Enable potential equation with drifts on"""
    bbb.b0 = b0         # =1 for normal direction of B-field, 100 for 1/100th of real B
    bbb.isphion = 1     # turn on potential equation
    bbb.rsigpl = 1.e-8  # anomalous cross-field conductivity
    bbb.newbcl = 1      # Sheath BC [bee,i] from current equation
    bbb.newbcr = 1      # Sheath BC [bee,i] from current equation
    bbb.isnewpot = 1    # 1=new potential; J_r from tor. mom. bal, -2=phi constant on core boundary with total core current = icoreelec
    bbb.rnewpot = 1.    # mixture of fqy=(1-rnewpot)*fqy_old+rnewpot*fqy_new (fqy = net radial current north)
    bbb.iphibcc = 1     # set bbb.phi[,1] uniform poloidally
    
    bbb.cfjpy = 0.      # diamag. cur. in flx.y-direction
    bbb.cfjp2 = 0.      # diamag. cur. in 2-direction
    bbb.cfydd = 0.      # Diamag. drift in flx.y-dir. [always=0]
    bbb.cf2dd = 0.      # Diamag. drift in 2-dir. [always=0]
    bbb.cfrd = 0.       # Resistive drift in flx.y and 2 dirs.
    bbb.cfbgt = 0.      # Diamag. energy drift [always=0]
    
    bbb.cfjhf = c      # turn-on heat flow from current [bbb.fqp]
    bbb.cfjve = c      # makes bbb.vex = vix - bbb.cfjve*bbb.fqx
    bbb.jhswitch = 1   # Joule Heating switch
    bbb.isfdiax = c    # Factor to turn on diamag. contrib. to sheath
    bbb.cfyef = c      # ExB drift in flx.y-dir.
    bbb.cf2ef = c      # ExB drift in 2-dir.
    bbb.cfybf = c      # turns on bbb.vycb - radial grad_B drift
    bbb.cf2bf = c      # turns on bbb.v2cb - perp grad_B drift [nearly pol]
    bbb.cfqybf = c     # turns on bbb.vycb contrib to radial current
    bbb.cfq2bf = c     # turns on bbb.v2cb contrib to perp["2"] current
    
    if rog:
        bbb.iphibcc = 3  # set phi core constant
        bbb.cfqydbo = c  # factor to includ. fqyd in core current B.C. only
        bbb.cfniybbo = c # factor to includ. vycb in fniy,feiy at iy=0 only
        bbb.cfeeybbo = c # factor to includ. vycb in feey at iy=0 only
        bbb.cfeixdbo = c # factor includ v2cdi & BxgradTi in BC at ix=0,nx 
        bbb.cfeexdbo = c # factor includ v2cde & BxgradTe in BC at ix=0,nx
        bbb.cftef = c    # Coef for ExB drift in toroidal direction
        bbb.cftdd = c    # Coef for diamagnetic drift in toroidal direction
        bbb.isutcore = 2 # Used for ix=ixcore phi BC ONLY IF iphibcc > 3
                         # >1, d^2(Ey)/dy^2=0 at outer midplane
                         
                         
def setDriftsRog():
    """Verbatim procedure that Tom sent me"""
    bbb.isphion=1
    bbb.b0 = 1.0      #=1 for normal direction B field
    bbb.rsigpl=1.e-8  #anomalous cross field conductivity
    bbb.cfjhf=1.      #turn on heat flow from current (fqp)
    bbb.cfjve=1.      #makes vex=vix-cfjve*fqx
    bbb.jhswitch=1    #Joule Heating switch
    bbb.newbcl=1      #Sheath boundary condition (bcee, i) from current equation
    bbb.newbcr=1
    bbb.isfdiax=1     #Factor to turn on diamagnetic contribution to sheath
    bbb.cfyef=1.0     #EXB drift in y direction
    bbb.cf2ef=1.0     #EXB drift in 2 direction
    bbb.cfybf=1.0     #turns on vycb - radial Grad B drift
    bbb.cf2bf=1.0     #turns on v2cb - perp. Grad B drift (nearly pol)
    bbb.cfqybf=1.0    #turns on vycb contrib to radial current
    bbb.cfq2bf=1.0    #turns on v2cb contrib to perp("2") current
    bbb.cfydd=0.0     #turns off divergence free diamagnetic current
    bbb.cf2dd=0.0     #turns off divergence free perp diagmatic current
    bbb.cfqybbo=0     #turn off Grad B current on boundary
    bbb.cfqydbo=1     #use full diagmagetic current on boundary to force j_r=0
    bbb.cfniybbo=1.   # use to avoid artificial source at core boundary
    bbb.cfniydbo=0.   # use to avoid artificial source at core boundary
    bbb.cfeeybbo=1.   # ditto
    bbb.cfeeydbo=0.   # ditto
    bbb.cfeixdbo=1.   # turn on BXgrad(T) drift in plate BC
    bbb.cfeexdbo=1.   # turn on diamagnetic drift in plate BC
    bbb.cftef=1.0     #turns on v2ce for toroidal velocity
    bbb.cftdd=1.0     #turns on v2dd (diamag vel) for toloidal velocity
    bbb.cfqym=1.0     #turns on inertial correction to fqy current
    bbb.iphibcc=3     # =3 gives ey=eycore on core bdry
    bbb.iphibcwi=0    # set ey=0 on inner wall if =0
                      # phi(PF)=phintewi*te(ix,0) on PF wall if =1
    bbb.iphibcwo=0    # same for outer wall
    bbb.isutcore=2    # =1, set dut/dy=0 on iy=0 (if iphibcc=0)
                      # =0, toroidal angular momentum=lzcore on iy=0 (iphibcc=0)
    bbb.isnewpot=1.0
    bbb.rnewpot=1.0
    
    
def unsetDriftsRog():
    """Return to non-drift values, at least those of CMod.py"""
    bbb.isphion=0
    bbb.b0 = 0      #=1 for normal direction B field
    bbb.rsigpl=0  #anomalous cross field conductivity
    bbb.cfjhf=1.      #turn on heat flow from current (fqp)
    bbb.cfjve=0      #makes vex=vix-cfjve*fqx
    bbb.jhswitch=0    #Joule Heating switch
    bbb.newbcl=0      #Sheath boundary condition (bcee, i) from current equation
    bbb.newbcr=0
    bbb.isfdiax=0     #Factor to turn on diamagnetic contribution to sheath
    bbb.cfyef=0     #EXB drift in y direction
    bbb.cf2ef=0     #EXB drift in 2 direction
    bbb.cfybf=0     #turns on vycb - radial Grad B drift
    bbb.cf2bf=0     #turns on v2cb - perp. Grad B drift (nearly pol)
    bbb.cfqybf=0    #turns on vycb contrib to radial current
    bbb.cfq2bf=0.0    #turns on v2cb contrib to perp("2") current
    bbb.cfydd=0.0     #turns off divergence free diamagnetic current
    bbb.cf2dd=0.0     #turns off divergence free perp diagmatic current
    bbb.cfqybbo=0     #turn off Grad B current on boundary
    bbb.cfqydbo=0     #use full diagmagetic current on boundary to force j_r=0
    bbb.cfniybbo=0.   # use to avoid artificial source at core boundary
    bbb.cfniydbo=0.   # use to avoid artificial source at core boundary
    bbb.cfeeybbo=0.   # ditto
    bbb.cfeeydbo=0.   # ditto
    bbb.cfeixdbo=0.   # turn on BXgrad(T) drift in plate BC
    bbb.cfeexdbo=0.   # turn on diamagnetic drift in plate BC
    bbb.cftef=0.0     #turns on v2ce for toroidal velocity
    bbb.cftdd=0.0     #turns on v2dd (diamag vel) for toloidal velocity
    bbb.cfqym=1.0     #turns on inertial correction to fqy current
    bbb.iphibcc=3     # =3 gives ey=eycore on core bdry
    bbb.iphibcwi=0    # set ey=0 on inner wall if =0
                      # phi(PF)=phintewi*te(ix,0) on PF wall if =1
    bbb.iphibcwo=0    # same for outer wall
    bbb.isutcore=0    # =1, set dut/dy=0 on iy=0 (if iphibcc=0)
                      # =0, toroidal angular momentum=lzcore on iy=0 (iphibcc=0)
    bbb.isnewpot=0.0
    bbb.rnewpot=0.0
    
    
def setCompatible():
    '''
    Make UEDGE 7.9.2 or higher compatible with 7.8.4 solutions. If this method is not run,
    a solution obtained with 7.8.4 will require further run time in 7.9.2 to restore.
    '''
    bbb.kxe=1.35 #-parallel conduction factor                                                    
    bbb.islnlamcon=1 #-Coulomb log                                                               
    bbb.lnlam=12
    bbb.isplflxlv=1 #=0, flalfv not active at ix=0 & nx;=1 active all ix                         
    bbb.isplflxlgx=1
    bbb.isplflxlgxy=1
    bbb.isplflxlvgx=1
    bbb.isplflxlvgxy=1
    bbb.iswflxlvgy=1
    bbb.isplflxltgx=1
    bbb.isplflxltgxy=1
    bbb.iswflxltgy=1 #-the main one, the only one that really matters                            
    bbb.isplflxl=1   # =0, flalfe,i not active at ix=0 & nx;=1 active all ix                     
    bbb.iswflxlvgy=1 # integer /1/ #=0, flalfvgy not active at iy=0 & ny;=1 active all iy        
    bbb.cngflox=1    # real /ngspmx*1./ #fac for x-flux from convection in ng-eqn.               
    bbb.isupwi=1     # integer /nispmx*1/ #=2 sets dup/dy=0 on inner wall                        
    bbb.isupwo=1     # integer /nispmx*1/ #=2 sets dup/dy=0 on outer wall           
    
    
def getUEDGEVar(varString):
    module, var = varString.split('.')
    return getattr(globals()[module], var)
    
    
def setUEDGEVar(varString, value):
    module, var = varString.split('.')
    setattr(globals()[module], var, value)
    

def init(dtreal=1e-12, ftol=1e-7, _retry=False):
    '''
    Proper initialization procedure before rundtp: need to begin with a successful small timestep.
    '''
    bbb.dtreal = dtreal
    bbb.icntnunk = 0
    bbb.itermx = 10
    bbb.ftol = ftol
    bbb.isbcwdt = 1
    bbb.rlx = 0.9 # fractional change allowed per iteration
    bbb.exmain()
    if bbb.iterm == 1:
        # Make sure the right ijactot/icntnunk situation is achieved
        if bbb.ijactot >= 2:
            bbb.icntnunk = 1 # continuation mode that reuses prev. precond. Jacobian
        else:
            print('init: ijactot = %d, lowering ftol' % bbb.ijactot)
            i = 0
            while bbb.ijactot < 2 and i < 5:
                bbb.ftol /= 10
                bbb.exmain()
                if bbb.iterm != 1:
                    print('init: trying to get ijactot >= 2 failed')
                    return
                i+= 1
    # Disabled 1/15/22 and put isbcwdt = 1 higher up
    # else:
    #     if not _retry:
    #         bbb.isbcwdt = 1 # make all BCs of the time-relaxation method
    #         init(dtreal, _retry=True)
            
            
def restoreBasic(name):
    '''
    Just restore the main ion and neutral variables.
    For when there is some shape mismatch due to e.g. turning on multi-impurity model.
    '''
    with h5py.File(name + '.h5', 'r') as h:
        bbb.nis[:,:,0] = h['bbb/nis'][()][:,:,0]
        bbb.nis[:,:,1] = h['bbb/nis'][()][:,:,1]
        bbb.ngs[:,:,0] = h['bbb/ngs'][()][:,:,0]
        bbb.ups[:,:,0] = h['bbb/ups'][()][:,:,0]
        bbb.ups[:,:,1] = h['bbb/ups'][()][:,:,1]
        bbb.tis = h['bbb/tis'][()]
        bbb.tes = h['bbb/tes'][()]
        bbb.tgs[:,:,0] = h['bbb/tgs'][()][:,:,0]
        
                
def readvar(filename, varname):
    '''
    E.g. readvar('bl3', 'bbb.ni') will return the full bbb.ni save data.
    '''
    varname = varname.replace('.', '/')
    with h5py.File(filename + '.h5', 'r') as h:
        return h[varname][()]


def rest(name='uerun', ftol=1e-8, changeGrid=False):
    '''
    Restore simulation variables and set label based on filename. Must be run after 
    completing corresponding simulation setup.
    '''
    if name.strip() == '':
        help(rest)
        return
    h5file = name + '.h5'
    if not os.path.isfile(h5file):
        raise OSError(2, 'No such file', h5file)
    bbb.label[0] = name.replace('/', '-').replace('..', '') 
    if changeGrid:
        with h5py.File(name + '.h5', 'r') as h:
            com.nxleg = h['com/nxleg'][()]
            com.nxcore = h['com/nxcore'][()]
            com.nycore = h['com/nycore'][()]
            com.nysol = h['com/nysol'][()]
        prepGrid(interpolate=False)
        print('Completed prepGrid')
    uedge.hdf5.hdf5_restore(name + '.h5')
    bbb.icntnunk = 0 # not a continuation call
    bbb.restart = 1
    bbb.dtreal = 1e20
    bbb.itermx = 5
    if ftol:
        bbb.ftol = ftol
    bbb.exmain() 
    if bbb.iterm != 1:
        raise Exception('iterm != 1')
    # Calculate impurity radiation if impurities are on
    if bbb.isimpon != 0:
        bbb.pradpltwl()
if ipython:
    @register_line_magic
    def restWrapper(*args, **kwargs):
        return rest(*args, **kwargs)
    ipython.run_line_magic('alias_magic', 'r restWrapper')       
         
                    
def loadVars(h, pkgs, verbose=True):
    """
    Load values from hdf5 file object for all the specified UEDGE packages.
    
    Args:
        h (hdf5 file object): the hdf5 file object from which to load values
        pkgs (list): e.g. ['bbb', 'com']
    """
    for pkg in pkgs:
        pkgObject = globals()[pkg]
        if pkg in h.keys():
            for var in h[pkg].keys():
                node = h[pkg + '/' + var]
                try:
                    o = pkgObject.getpyobject(var)
                    v = node[()]
                    if node.size > 1:
                        v = np.array(v)
                        if hasattr(o, 'shape') and o.shape == v.shape and np.any(o != v):
                            o[...] = v
                            if verbose:
                                if node.size < (com.nx+2)*(com.ny+2) or 'use' in var:
                                    print(pkg + '.' + var, v.shape, 'modified')
                    else:
                        if o != v:
                            setattr(pkgObject, var, v)
                            if verbose:
                                print(pkg + '.' + var, o, '->', v)
                except Exception as e:
                    print('(%s.%s) %s: %s' % (pkg, var, type(e).__name__, e))


def fpost(name='untitled', addvars=[]):
    '''
    Save full hdf5 file and generate plotall() pdf.
    
    Args:
        name: (str) filename excluding .h5
        addvars: (list of strings) additional variables to include
    '''
    bbb.label[0] = name
    fsave(name=name)
    UEDGE_utils.plot.plotall(name)
        
        
def frest(name='uerun'):
    '''
    Restore UEDGE case from full save.
    '''
    h5file = name + '.h5'
    if not os.path.isfile(h5file):
        raise OSError(2, 'No such file', h5file)
    with h5py.File(name + '.h5', 'r') as h:
        # Set up gridfile
        if os.path.isfile('gridue'):
            shutil.move('gridue', 'gridue_moved')
        with open('gridue', 'w') as f:
            f.write(str(h['gridue'][()]))
        # Restore essential variables before allocating
        loadVars(h, ['com', 'bbb'])
        bbb.allocate()
        # Restore more variables after proper allocation
        loadVars(h, ['bbb', 'com', 'flx', 'grd', 'svr', 'aph', 'api'])
    # Run
    bbb.icntnunk = 0 # not a continuation call
    bbb.restart = 1
    bbb.dtreal = 1e20
    bbb.itermx = 10
    bbb.ftol = 1e-8
    bbb.exmain() 
    if bbb.iterm != 1:
        raise Exception('iterm != 1')
    # Calculate impurity radiation if impurities are on
    if bbb.isimpon != 0:
        bbb.pradpltwl()
        
dontSaveGroups = ['bbb','Jacreorder','Jacobian','Jacobian_csc','Jacobian_part','Nonzero_diagonals','work_arrays','Aux','Wkspace'] 
dontShowGroups = ['bbb','Compla','Postproc','Comflo','Indexes','Stat','Gradients','Locflux','Rhsides','PNC_data','Decomp','Condition_number','Global_vars','Indices_domain_dcg','Indices_loc_glob_map','Indices_domain_dcl','Jacaux','RZ_cell_info','Impurity_source_flux','Bdy_indexlims']
# These are large work arrays 
dontsave = ['perm','qperm','levels','mask','jaci','jcsc','rwk1','rwk2','iwk1','iwk2','iwk3','rtol','atol','yl','yldot','delta','icnstr','constr','ylprevc','ylchng','suscal','sfscal','yloext','iseqalg','ylodt','dtoptv','dtuse','yldot_pert','yldot_unpt','ylold','yldot1','yldot0','fnormnw','ycor','vrsend','visend','vrsendl','visendl','ivloc2sdgl','ivloc2mdgl','v0zag','u0zag','t0zag','n0zag','stressg_mc','stressg_mc_rsd','stressg_ue','stressg_ue_rsd','igyl','ysave','radrate','avgz','avgz2','nzzag','vzzag','uzzag','vzparzag','sdod','test1','tist1','ngst1','phist1','nist1','upst1','ivloc2sdg','ivloc2mdg','iellast','jac','jacj','rcsc','icsc','wwp','iwwp','iwork','rwork']
# These are variables that show up as changing even when not set in code
dontsave2 = ['tend','tstart','ttjstor','ttotjf','csh','msh','ncrhs','qsh','t1','t2','ijac','a','t0','ydt_max','ydt_max0','ubw','qfl','mfl','cs','iv','iv1','iv2','iv3','iy','ix','ix1','ix2','ix3','ix4','ix5','ix6','tv','lbw','lenplumx','liw','liwp','lrw','lwp','mmaxu','neq','neqmx','neqp1','nnzmx','numvar','numvarl','ttotfe']
# These are not user settings
dontshow = []

pkgs = []


def dictDifferences(a, b):
    differences = []
    for k, v in a.items():
        if type(v) == np.ndarray:
            if b[k].shape != v.shape or np.any(b[k] != v):
                differences.append(k)
        else:
            if b[k] != v:
                differences.append(k)
    return differences
    
    
# def getuedict():
    
                

def compareDicts(a, b):
    for k in dictDifferences(a, b):
        print(k, 'modified')
        
        
def fsave(name='uerun'):
    """
    Save all currently allocated UEDGE variables and grid to hdf5 file. (~10 MB 60x20)
    """
    skip = dontsave
    if bbb.isimpon != 0:
        bbb.pradpltwl()        
    with h5py.File(name + '.h5','w') as h:
        for pkg in ['bbb', 'com', 'flx', 'grd', 'svr', 'aph', 'api']:
            pkgObject = globals()[pkg]
            currDict = pkgObject.getdict()
            for var in currDict.keys():
                if var not in skip:
                    try:
                        obj = currDict[var]
                        if type(obj) == np.ndarray:
                            h.create_dataset(pkg + '/' + var, data=obj, compression='gzip') 
                        else:
                            h[pkg + '/' + var] = obj
                    except Exception as e:
                        print('(%s.%s) %s: %s' % (pkg, var, type(e).__name__, e))
        # with open('gridue', 'r') as f:
        #     h.create_dataset('gridue', data=f.read()) 
            
            
def saveh5Extended(name='untitled', addvars=[]):
    """Save hdf5 file with extended variable list for easier analysis.
    
    Args:
        name: (str) filename excluding .h5
        addvars: (list of strings) additional variables to include
    """
    # List of useful extra variables
    extendedVarList = ['bbb.b0','bbb.isteon','bbb.istion','bbb.isnion','bbb.isngon','bbb.isupon','bbb.isupgon','bbb.istgon','bbb.isphion','bbb.isphiofft','bbb.isnewpot','bbb.dif_use','bbb.vy_use','bbb.kye_use','bbb.kyi_use','bbb.difni','bbb.kye','bbb.kyi','bbb.travis','bbb.isbohmcalc','bbb.difniv','bbb.travisv','bbb.kyev','bbb.kyiv','bbb.vconyv','bbb.inbpdif','bbb.inbtdif','bbb.pcoree','bbb.pcorei','bbb.iflcore','bbb.tcoree','bbb.tcorei','bbb.afracs','bbb.feex','bbb.feix','bbb.feey','bbb.feiy','bbb.fnix','bbb.fniy','bbb.pwr_plth','bbb.pwr_pltz','bbb.recycp','bbb.recycw','bbb.isnicore','bbb.isngcore','bbb.curcore','bbb.ncore','bbb.ngcore','bbb.istewc','bbb.istiwc','bbb.isnwcono','bbb.ifluxni','bbb.nwimin','bbb.nwomin','bbb.istepfc','bbb.istipfc','bbb.isnwconi','bbb.tewallo','bbb.tiwallo','bbb.nwallo','bbb.tewalli','bbb.tiwalli','bbb.nwalli','bbb.lyte','bbb.lyti','bbb.lyni','com.nxleg','com.nxcore','com.nycore','com.nysol','com.sx','com.sxnp','com.sy','com.rr','com.yyc','com.iysptrx','bbb.ixmp','com.ixpt1','com.ixpt2','com.vol']
    # Add user-requested variables to save list
    extendedVarList.extend(addvars)
    # Remove duplicates
    extendedVarList = list(set(extendedVarList))
    # Postprocess so radiation-to-walls and other variables are computed
    bbb.pradpltwl()
    uedge.hdf5.hdf5_save(name + '.h5', addvarlist=extendedVarList)
    
    
def post(name='untitled', addvars=[], interactive=True):
    '''
    Save hdf5 file and generate plotall() pdf.
    
    Args:
        name: (str) filename excluding .h5
        addvars: (list of strings) additional variables to include
    '''
    if name.strip() == '':
        help(post)
        return
    if interactive:
        if os.path.isfile(name+'.h5') or os.path.isfile(name+'.pdf'):
            if not yesno(name+'.h5/pdf exists. Overwrite?'):
                return
    bbb.label[0] = name
    saveh5Extended(name=name, addvars=addvars)
    UEDGE_utils.plot.plotall(name)
if ipython:
    @register_line_magic    
    def postWrapper(*args, **kwargs):
        return post(*args, **kwargs)
    ipython.run_line_magic('alias_magic', 'p postWrapper')
    
    
def notify(message='Run complete'):
    pass
    # message += ', iterm=%d' % bbb.iterm
    # msg = MIMEText(message)
    # msg["To"] = "user@email.com"
    # msg["Subject"] = "UEDGE"
    # # -t: extract recipients from message headers
    # # -oi: don't treat a line with only "." as the end of input
    # p = subprocess.Popen(["sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
    # p.communicate(msg.as_bytes())
    
    
def tanh(psi, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    """
    Function used to fit TS profiles. Written based on MDSplus tree comment.
    """
    c = np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9])
    z = 2.*(c[0]-psi)/c[1]
    pz2 = 1 + ( c[7]*z + (c[8]*z*z) ) # depending on whether there are 7,8,or 9 coefficients specified
    pz1 = 1.+ c[4]*z + c[5]*z*z + c[6]*z*z*z # if param=None
    return 0.5*(c[2]-c[3])* ( pz1*np.exp(z) - pz2*np.exp(-z) )/(np.exp(z) + np.exp(-z) ) + 0.5*(c[2]+c[3])


def fitDensityTS():
    with h5py.File('targetData_1160718025.h5','r') as h5:
        nerhos = h5['neomid/rho'][()]
        nes = h5['neomid/value'][()]
        neerrs = h5['neomid/value_err'][()]
        mask = (-.1 < nerhos) & (nerhos < .1)
        nes = nes[mask]
        nerhos = nerhos[mask]
        neerrs = neerrs[mask]
    popt, pcov = curve_fit(tanh, nerhos, nes/1e20, maxfev=100000,sigma=neerrs/1e20)
    return tanh, popt    


def densityTS(shift=0.0002):
    fun, popt = fitDensityTS()
    prof_interp = np.array([fun(rho+shift, *popt) for rho in com.yyc])*1e20
    return np.clip(prof_interp,0,np.inf)

    
def gradDensityTS(shift=0.0002):
    fun, popt = fitDensityTS()
    funp = lambda rho: fun(rho+shift, *popt)*1e20
    return np.array([scipy.misc.derivative(funp, y, dx=1e-5) for y in com.yyc])
    

def temperatureTS(shift=0.0002):
    fun, popt = fitTemperatureTS()
    prof_interp = np.array([fun(rho+shift, *popt) for rho in com.yyc])
    return np.clip(prof_interp,0,np.inf)
    
    
def TsepTS(shift=0.0002):
    fun, popt = fitTemperatureTS()
    funp = lambda rho: fun(rho+shift, *popt)
    return funp(0)
    
    
def TwalloTS(shift=0.0002):
    fun, popt = fitTemperatureTS()
    funp = lambda rho: fun(rho+shift, *popt)
    return funp(com.yyc[-1])
    
    
def fitTemperatureTS():
    with h5py.File('targetData_1160718025.h5','r') as h5:
        terhos = h5['teomid/rho'][()]
        tes = h5['teomid/value'][()]
        teerrs = h5['teomid/value_err'][()]
        mask = (-.1 < terhos) & (terhos < .1)
        tes = tes[mask]
        teerrs = teerrs[mask]
        terhos = terhos[mask]
    # Modified tanh fitting to produce better shape... fixed instead by expanding fitting region
    #tanhmod = lambda psi, c2, c3, c4, c5, c7: tanh(psi, 0, c2, c3, c4, c5, 0, c7, 0, 0)
    popt, pcov = curve_fit(tanh, terhos, tes, maxfev=100000, sigma=teerrs)
    return tanh, popt
    
    
def gradTemperatureTS(shift=0.0002):
    fun, popt = fitTemperatureTS()
    funp = lambda rho: fun(rho+shift, *popt)
    return np.array([scipy.misc.derivative(funp, y, dx=1e-5) for y in com.yyc])
    
    
def to2D(profile):
    out = np.zeros((com.nx+2,com.ny+2))
    out[com.ixpt1[0]+1:com.ixpt2[0]+1,:] = profile
    out[:com.ixpt1[0]+1,com.iysptrx+1:] = profile[com.iysptrx+1:]
    out[com.ixpt2[0]+1:,com.iysptrx+1:] = profile[com.iysptrx+1:]
    out[:com.ixpt1[0]+1,:com.iysptrx+1] = profile[com.iysptrx+1]
    out[com.ixpt2[0]+1:,:com.iysptrx+1] = profile[com.iysptrx+1]
    return out
    
    
def clip(profile, cmin=0.1, cmax=1):
    return np.clip(profile, cmin, cmax)    
    
    
def upwind(f, p1, p2): 
    return max(f,0)*p1+min(f,0)*p2


def upwindProxy(f, g, p1, p2):
    return max(f,0)/f*g*p1+min(f,0)/f*g*p2
    
    
def newD(cmin=-np.inf, cmax=np.inf, shift=0.0002,div=False): # was .001 and 1
    vyconv = bbb.vcony[0] + bbb.vy_use[:,:,0] + bbb.vy_cft[:,:,0]
    vydif = bbb.vydd[:,:,0]-vyconv
    fniydif = np.zeros((com.nx+2,com.ny+2))
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            # This is for upwind scheme (methn=33)
            if bbb.vy[ix,iy,0] > 0:
                t2 = bbb.niy0[ix,iy,0] # outside sep in case I developed this with
            else:
                t2 = bbb.niy1[ix,iy,0] # inside sep in case I developed this with
            fniydif[ix,iy] = vydif[ix,iy]*com.sy[ix,iy]*t2
    gradn = gradDensityTS(shift=shift)
    d = -fniydif[bbb.ixmp,:]/(gradn*com.sy[bbb.ixmp,:]+1e-30)
    d[-1] = d[-2]
    d2d = to2D(clip(d, cmin=cmin, cmax=cmax))
        
    n = densityTS(shift=shift)
    print('%.3g should be core boundary density (currently %.3g)' % (n[0], bbb.ncore[0]))
    print('%.3g should be edge grad len' % (-n[-2]/gradn[-2]))
    return d2d
    

def newkye(cmin=-np.inf, cmax=np.inf, shift=0.0002): # was .001, 100
    feey = np.zeros((com.nx+2,com.ny+2))
    econv = np.zeros((com.nx+2,com.ny+2))
    econd = np.zeros((com.nx+2,com.ny+2))
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            econd[ix,iy]=-bbb.conye[ix,iy]*(bbb.te[ix,iy+1]-bbb.te[ix,iy])
            econv[ix,iy]=upwind(bbb.floye[ix,iy],bbb.te[ix,iy],bbb.te[ix,iy+1])
    feey = econd+econv # should match bbb.feey
    gradt = gradTemperatureTS(shift=shift)
    k = -(econd[bbb.ixmp])/(densityTS(shift=shift)*gradt*bbb.ev*com.sy[bbb.ixmp]-1e-100)
    k[-1] = k[-2]
    t = temperatureTS(shift=shift)
    print('%.3g should be edge grad len' % (-t[-2]/gradt[-2]))
    return to2D(clip(k, cmin=cmin, cmax=cmax))
        
    
def newkyi(cmin=-np.inf, cmax=np.inf, shift=0.0002):
    feiy = np.zeros((com.nx+2,com.ny+2))
    iconv = np.zeros((com.nx+2,com.ny+2))
    nconv = np.zeros((com.nx+2,com.ny+2))
    icond = np.zeros((com.nx+2,com.ny+2))
    ncond = np.zeros((com.nx+2,com.ny+2))
    conyn = com.sy*bbb.hcyn/com.dynog
    for ix in range(0,com.nx+2):
        for iy in range(0,com.ny+1):
            ncond[ix,iy]=-conyn[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])
            icond[ix,iy]=-bbb.conyi[ix,iy]*(bbb.ti[ix,iy+1]-bbb.ti[ix,iy])-ncond[ix,iy]
            floyn = bbb.cfneut*bbb.cfneutsor_ei*2.5*bbb.fniy[ix,iy,1]
            floyi = bbb.floyi[ix,iy]-floyn # ions only, unlike bbb.floyi
            iconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyi,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
            nconv[ix,iy]=upwindProxy(bbb.floyi[ix,iy],floyn,bbb.ti[ix,iy],bbb.ti[ix,iy+1])
    feiy = icond+iconv+ncond+nconv # should match bbb.feiy
    gradt = gradTemperatureTS(shift=shift)
    k = -(icond[bbb.ixmp])/(densityTS(shift=shift)*gradt*bbb.ev*com.sy[bbb.ixmp]-1e-100)
    k[-1] = k[-2]
    t = temperatureTS(shift=shift)
    print('%.3g should be edge grad len' % (-t[-2]/gradt[-2]))
    return to2D(clip(k, cmin=cmin, cmax=cmax))
    
    
def step(targets, timeoutMins=10, dfrac=0.1, minDfrac=0.06, killDfrac=1e-10, changeGrid=False, plot=True, notifyWhenDone=True, refit=False, rlx=0.9):
    '''
    Change variable from original to target value with adaptive steps in value
    using rundtp with timeout.
    Arguments:
        targets: (dict) e.g. {'bbb.tiwallo': 1, 'bbb.tiwalli': 1}
        timeoutMins: (int) minutes to run rundtp before killing and trying a smaller step
        dfrac: (float between 0 and 1) initial step to take (default 0.1 = 10%)
        minDfrac: (float) if dfrac gets this small, let rundtp run as long as it needs to succeed.
                  If minDfrac=None, always do rundtp with execution time limit.
        killDfrac: (float) stop stepping if dfrac gets this small
        notifyWhenDone: (bool) send notification email when done
    ''' 
    stepStartTime = time.time()
    
    # Create temporary file to store intermediate solutions (deleted if step() succeeds in the end)
    tempDir = 'step'
    if not os.path.isdir(tempDir):
        os.mkdir(tempDir)
    _, savePath = tempfile.mkstemp(prefix='', suffix='.h5', dir=tempDir)
    saveFile = os.path.join(tempDir, os.path.basename(savePath))
    saveh5Extended(saveFile.split('.h5')[0], addvars=list(targets.keys()))
    
    starts = {v: np.copy(getUEDGEVar(v)) for v in targets.keys()}
    errCount = 0
    frac = 0
    while frac <= 1:
        dfrac = min(dfrac, 1-frac)
        if killDfrac and dfrac < killDfrac:
            print('Step: dfrac < killDfrac')
            bbb.iterm = 2
            break
        frac = frac + dfrac
        status = 'step %.5g%% + %.5g%% -> %.5g%%, savefile %s' % ((frac-dfrac)*100, dfrac*100, frac*100, saveFile)
        print('Step: starting ' + str(status))
        for var in targets.keys():
            newVal = (1-frac)*starts[var]+frac*targets[var]
            setUEDGEVar(var, newVal)
        if changeGrid:
            prepGrid()
            if len(UEDGE_utils.analysis.badCells()) > 0:
                print('Step: quitting due to grid problems')
                bbb.iterm = 2
                break
        rundtpStart = time.time()
        try:
            if refit:
                bbb.dif_use[:,:,0] = newD()
                bbb.kye_use = newkye()
                bbb.kyi_use = newkyi()
            if minDfrac and dfrac < minDfrac:
                rundtp(message=status, notifyWhenDone=False, rlx=rlx) 
            else:
                rundtp(timeoutMins=timeoutMins, message=status, notifyWhenDone=False, rlx=rlx) 
        except Exception as e:
            print('rundtp error ' + str(e))
            errCount += 1
            bbb.iterm = 2
        rundtpEnd = time.time()
        # if refit and bbb.iterm == 1:
        #     bbb.dif_use[:,:,0] = newD()
        #     bbb.kye_use = newkye()
        #     bbb.kyi_use = newkyi()
        #     rundtp(timeoutMins=timeoutMins, message=status + ' refit', notifyWhenDone=False)
        if bbb.iterm == 1:
            # Step succeeded: save to temp file and increase step size
            saveh5Extended(saveFile.split('.h5')[0], addvars=list(targets.keys()))
            if plot:
                UEDGE_utils.plot.plotall(saveFile.split('.h5')[0])
            print('Step: step complete, saved h5 file.')
            if frac == 1:
                # Stepping complete: remove temp file and break out of loop
                print('Step: stepping complete.')
                os.remove(saveFile)
                break
            # Calculate new step size based on rundtp() completion time relative to timeoutMins: 
            # fast = much bigger step, slow = slightly bigger step
            dfracNew = dfrac*(1+min(0.5, 0.1*timeoutMins*60/(rundtpEnd-rundtpStart)))
            #dfracNew = dfrac*timeoutMins*60/(rundtpEnd-rundtpStart)/2
        else:
            # If this step was unsuccessful, rewind to last successful case and set step /= 3
            frac = frac-dfrac
            dfracNew = dfrac/3
            print('Step: restoring %.5g%% of (target - original)' % (frac*100))
            # if changeGrid:
            #     prepGrid()
            bbb.icntnunk = 0
            rest(saveFile.split('.h5')[0], changeGrid=changeGrid)
            if bbb.iterm != 1:
                print('Step: restore %s failed' % saveFile)
                break
        dfrac = dfracNew
    stepTime = datetime.timedelta(seconds=int(time.time()-stepStartTime))
    print('Step: finished in %s with %d errors' % (str(stepTime), errCount))
    if notifyWhenDone and time.time()-stepStartTime > 5*60:
        notify('Step complete %s %s' % (stepTime, ', '.join(targets)))


def rundtp(dtreal=1e-10, nfe_tot=0, savedir='solutions', dt_tot=0, ii1max=50000, 
           ii2max=5, ftol_dt=1e-5, itermx=7, rlx=0.9, n_stor=0, 
           tstor=(1e-3, 4e-2), incpset=7, dtmfnk3=1e-4, timeoutMins=None,
           saveIntermediates=False, saveAll=False, debug=False, storeExt=False, storeName='storeExt.h5', initdt=1e-12,
           message=None, notifyWhenDone=True):
    ''' 
    Args
        saveIntermediates: (False) legacy system of saving ..._last_ii2.h5 after each success
        timeoutMins: (None) if rundtp takes longer than timeoutMins, give up
        initdt: (1e-10) small timestep for an initial run which must succeed before we launch into the main rundtp logic.
        storeExt: (False) store important variables after each successful timestep
        storeName: ('storeExt.h5') file name under which to store variables at each timestep
    
    Function advancing case time-dependently: increasing time-stepping is the default to attain SS solution
    rdrundt(dtreal,**keys)

    Variables
    dtreal                  The inital time step time

    Keyword parameters:
    nfe_tot[0]              Number of function evaluations
    savedir[savedt]         Directory where hdf5 savefile is written
    dt_tot[0]               Total time accummulated: default option resets time between runs    
    ii1max[500]             Outer loop (dt-changing) iterations
    ii2max[5]               Inner loop (steps at dt) iterations
    ftol_dt[1e-5]           Time-dependent fnrm tolerance 
    itermx[7]               Max. number of linear iterations allowed
    rlx[0.9]                Max. allowed change in variable at each iteration
    n_stor[0]               Number of linearly spaced hdf5 dumps 
    tstor_s[(1e-3,4e-2)]    Tuple with start and stop times for storing snapshots to HDF5
    incpset[7]              Iterations until Jacobian is recomputed
    dtmfnk[1e-4]            dtreal for mfnksol signchange if ismfnkauto=1 (default)
    The above defaults are based on rdinitdt.

    Additional UEDGE parameters used in the function, assuming their default values are:
    bbb.rdtphidtr[1e20]     # Ratio dtphi/dtreal
    bbb.ismfnkauto[1]       # If =1, mfnksol=3 for dtreal<dtmfnk3, otherwise=-3
    bbb.mult_dt[3.4]        # Factor expanding dtreal after each successful inner loop
    bbb.itermxrdc[7]        # Itermx used by the script
    bbb.ftol_min[1e-9]      # Value of fnrm where time advance will stop
    bbb.t_stop[100]         # Value of dt_tot (sec) where calculation will stop
    bbb.dt_max[100]         # Max. time step for dtreal
    bbb.dt_kill[1e-14]      # Min. allowed time step; rdcontdt stops if reached
    bbb.deldt_min[0.04]     # Minimum relative change allowed for model_dt > 0
    bbb.numrevjmax[2]       # Number of dt reductions before Jac recalculated
    bbb.numfwdjmax[1]       # Number of dt increases before Jac recalculated
    bbb.ismmaxuc[1]         # =1 for intern calc mmaxu; =0,set mmaxu & dont chng
    bbb.irev[-1]            # Flag to allow reduced dt advance after cutback
    bbb.initjac[0]          # If=1, calc initial Jac upon reading rdcontdt
    bbb.ipt[1]              # Index of variable; value printed at step
                            # If ipt not reset from unity, ipt=idxte(nx,iysptrx+1)
   
    Additional comments (from rdcontdt):
    This file runs a time-dependent case using dtreal.  First, a converged solution for a (usually small) dtreal is obtained:
    UEDGE must report iterm=1 at the end. Then the control parameters are adjusted. If a mistake is made, to restart this file 
    without a Jacobian evaluation, be sure to reset iterm=1 (=> last step was successful)
    '''
    start_time = time.time()
    
    if storeExt:
        storeExtVars = ['fnix', 'fngx', 'fniy', 'fngy', 'psor', 'psorrg', 'psorbgg', 'ni', 'up', 'te', 'ti', 'tg', 'ng', 'dt_tot', 'dtreal']
        h5storeExt = h5py.File(storeName, 'w')
        iStoreExt = 1
    
    if debug:
        nmaxes = []
        nmins = []
        ngmaxes = []
        ngmins = []
        temaxes = []
        temins = []
        timaxes = []
        timins = []
        times = []
        
    # Give simulation a generic label if it doesn't have one.
    # Otherwise intermediate savefile names start with 30 spaces.
    if bbb.label[0].isspace():
        bbb.label[0] = 'uerun'
        
    # Store the original values
    dt_tot_o=bbb.dt_tot
    ii1max_o=bbb.ii1max
    ii2max_o=bbb.ii2max
    ftol_dt_o=bbb.ftol_dt 
    itermx_o=bbb.itermx   
    rlx_o=bbb.rlx    
    n_stor_o=bbb.n_stor   
    tstor_s_o=bbb.tstor_s  
    tstor_e_o=bbb.tstor_e 
    incpset_o=bbb.incpset 
    dtmfnk3_o=bbb.dtmfnk3
    icntnunk_o=bbb.icntnunk
    ftol_o=bbb.ftol
    
    # Initialize with small timestep
    if initdt:
        init(initdt)
        bbb.rlx = rlx_o
        bbb.ftol = ftol_o
        bbb.itermx = itermx_o

    # Set inital time-step to dtreal
    bbb.dtreal = dtreal
    
    # Check if successful time-step exists (bbb.iterm=1)
    if (bbb.iterm == 1 and bbb.ijactot>1):
        print("Initial successful time-step exists")    
    else:
        print("*---------------------------------------------------------*")
        print("Init failed. Trying with smaller timestep 1e-15")
        print("*---------------------------------------------------------*")
        init(1e-15)
    bbb.dtreal = bbb.dtreal*bbb.mult_dt #compensates dtreal divided by mult_dt below

    if (bbb.iterm != 1):
        print("*--------------------------------------------------------------*")
        print("Error: converge an initial time-step first; then retry rdcontdt")
        print("*--------------------------------------------------------------*")
        return
    
    # Set UEDGE variables to the prescribed values
    bbb.dt_tot=dt_tot
    bbb.ii1max=ii1max
    bbb.ii2max=ii2max
    bbb.ftol_dt=ftol_dt 
    bbb.itermx=itermx   
    bbb.rlx=rlx    
    bbb.n_stor=n_stor   
    bbb.tstor_s=tstor[0]  
    bbb.tstor_e=tstor[1] 
    bbb.incpset=incpset 
    bbb.dtmfnk3=dtmfnk3

    # Saved intermediates counter
    i_stor=0

    # Helper variables
    nfe_tot = max(nfe_tot,0)
    deldt_0 = bbb.deldt

    # Empty dictionary for writing
    data=dict() 
    storevar=   [   ['ni',      bbb.ni],
                    ['up',      bbb.up],
                    ['te',      bbb.te],
                    ['ti',      bbb.ti],
                    ['tg',      bbb.tg],
                    ['ng',      bbb.ng],
                    ['phi',     bbb.phi],
                    ['dt_tot',  bbb.dt_tot],
                    ['nfe',     None],
                    ['dtreal',  bbb.dtreal]     ]
    # Linearly spaced time slices for writing 
    dt_stor = (bbb.tstor_e - bbb.tstor_s)/(bbb.n_stor - 1)

    isdtsf_sav = bbb.isdtsfscal

    if(bbb.ipt==1):  # No index requested
        # Check for first variable solved: order is defined as Te,Ti,ni,ng,Tg,phi
        for eq in [bbb.idxte, bbb.idxti, bbb.idxn, bbb.idxg, bbb.idxtg, bbb.idxu]:
            # If multi-species:
            if len(eq.shape)==3:
                # Loop through all species to find first solved
                for index in range(eq.shape[2]):
                    # See if equation is solved
                    if eq[:,:,index].min()!=0:
                        ipt=eq[com.nx-1,com.iysptrx+1,index]
                        break
            # If not, see if equation is solved
            else:
                if eq.min()!=0:
                    ipt=eq[com.nx-1,com.iysptrx+1]
                    break


    bbb.irev = -1         # forces second branch of irev in ii1 loop below
    if (bbb.iterm == 1):  # successful initial run with dtreal
        bbb.dtreal = bbb.dtreal/bbb.mult_dt     # gives same dtreal after irev loop
    else:                 # unsuccessful initial run; reduce dtreal
        bbb.dtreal = bbb.dtreal/(3*bbb.mult_dt) # causes dt=dt/mult_dt after irev loop
       
    if (bbb.initjac == 0): bbb.newgeo=0
    dtreal_sav = bbb.dtreal
    bbb.itermx = bbb.itermxrdc
    bbb.dtreal = bbb.dtreal/bbb.mult_dt	#adjust for mult. to follow; mult_dt in rdinitdt
    bbb.dtphi = bbb.rdtphidtr*bbb.dtreal
    bbb.ylodt = bbb.yl
    bbb.pandf1 (-1, -1, 0, bbb.neq, 1., bbb.yl, bbb.yldot)
    fnrm_old = np.sqrt(sum((bbb.yldot[0:bbb.neq]*bbb.sfscal[0:bbb.neq])**2))
    if (bbb.initjac == 1): fnrm_old=1.e20
    print("initial fnrm ={:.4E}".format(fnrm_old))

    for ii1 in range( 1, bbb.ii1max+1):
        if (bbb.ismfnkauto==1): bbb.mfnksol = 3
        # adjust the time-step
        if (bbb.irev == 0):
            # Only used after a dt reduc. success. completes loop ii2 for fixed dt
            bbb.dtreal = min(3*bbb.dtreal,bbb.t_stop)	#first move forward after reduction
            bbb.dtphi = bbb.rdtphidtr*bbb.dtreal
            if (bbb.ismfnkauto==1 and bbb.dtreal > bbb.dtmfnk3): bbb.mfnksol = -3
            bbb.deldt =  3*bbb.deldt
        else:
            # either increase or decrease dtreal; depends on mult_dt
            bbb.dtreal = min(bbb.mult_dt*bbb.dtreal,bbb.t_stop)
            bbb.dtphi = bbb.rdtphidtr*bbb.dtreal
            if (bbb.ismfnkauto==1 and bbb.dtreal > bbb.dtmfnk3): bbb.mfnksol = -3
            bbb.deldt =  bbb.mult_dt*bbb.deldt
          
        bbb.dtreal = min(bbb.dtreal,bbb.dt_max)
        bbb.dtphi = bbb.rdtphidtr*bbb.dtreal
        if (bbb.ismfnkauto==1 and bbb.dtreal > bbb.dtmfnk3): bbb.mfnksol = -3
        bbb.deldt = min(bbb.deldt,deldt_0)
        bbb.deldt = max(bbb.deldt,bbb.deldt_min)
        nsteps_nk=1
        print('--------------------------------------------------------------------')
        print('*** Number time-step changes = {} New time-step = {:.4E}'.format(ii1, bbb.dtreal))
        walltime = datetime.timedelta(seconds=int(time.time()-start_time))
        print('Wall time:', walltime, '    Simulation time: %.3g' % bbb.dt_tot)
        if message:
            print(message)
        print('--------------------------------------------------------------------')

        bbb.itermx = bbb.itermxrdc
        if (ii1>1  or  bbb.initjac==1):	# first time calc Jac if initjac=1
            if (bbb.irev == 1):      # decrease in bbb.dtreal
                if (bbb.numrev < bbb.numrevjmax and \
                    bbb.numrfcum < bbb.numrevjmax+bbb.numfwdjmax): #dont recom bbb.jac
                    bbb.icntnunk = 1	
                    bbb.numrfcum = bbb.numrfcum + 1
                else:                          # force bbb.jac calc, reset numrev
                    bbb.icntnunk = 0
                    bbb.numrev = -1		      # yields api.zero in next statement
                    bbb.numrfcum = 0
                bbb.numrev = bbb.numrev + 1
                bbb.numfwd = 0
            else:  # increase in bbb.dtreal
                if (bbb.numfwd < bbb.numfwdjmax and \
                    bbb.numrfcum < bbb.numrevjmax+bbb.numfwdjmax): 	#dont recomp bbb.jac
                    bbb.icntnunk = 1
                    bbb.numrfcum = bbb.numrfcum + 1
                else:
                    bbb.icntnunk = 0			#recompute jacobian for increase dt
                    bbb.numfwd = -1
                    bbb.numrfcum = 0
                bbb.numfwd = bbb.numfwd + 1
                bbb.numrev = 0			#bbb.restart counter for dt reversals
            bbb.isdtsfscal = isdtsf_sav
            bbb.ftol = max(min(bbb.ftol_dt, 0.01*fnrm_old),bbb.ftol_min)
            bbb.exmain() # take a single step at the present bbb.dtreal
            if (bbb.iterm == 1):
                bbb.dt_tot += bbb.dtreal
                nfe_tot += bbb.nfe[0,0]
                bbb.ylodt = bbb.yl
                bbb.pandf1 (-1, -1, 0, bbb.neq, 1., bbb.yl, bbb.yldot)
                fnrm_old = np.sqrt(sum((bbb.yldot[0:bbb.neq-1]*bbb.sfscal[0:bbb.neq-1])**2))
                if storeExt:
                        for v in storeExtVars:
                            h5storeExt[str(iStoreExt) + '/' + v] = np.copy(getattr(bbb, v))
                        iStoreExt += 1
                if (bbb.dt_tot>=0.9999999*bbb.t_stop  or  fnrm_old<bbb.ftol_min):
                    print(' ')
                    print('*****************************************************')
                    print('**  SUCCESS: frnm < bbb.ftol; or dt_tot >= t_stop  **')
                    print('*****************************************************')
                    # Remove last intermediate solution file
                    if saveIntermediates:
                        os.remove('{}_last_ii2.h5'.format(bbb.label[0].decode('UTF-8')))
                    break

        bbb.icntnunk = 1
        bbb.isdtsfscal = 0
        for ii2 in range( 1, bbb.ii2max+1): #take ii2max steps at the present time-step
            if (bbb.iterm == 1):
                if debug:
                    nmaxes.append(np.max(bbb.ni[:,:,0]))
                    nmins.append(np.min(bbb.ni[:,:,0]))
                    ngmaxes.append(np.max(bbb.ng))
                    ngmins.append(np.min(bbb.ng))
                    temaxes.append(np.max(bbb.te/bbb.ev))
                    temins.append(np.min(bbb.te/bbb.ev))
                    timaxes.append(np.max(bbb.ti/bbb.ev))
                    timins.append(np.min(bbb.ti/bbb.ev))
                    times.append(bbb.dt_tot)
                bbb.itermx = bbb.itermxrdc
                bbb.ftol = max(min(bbb.ftol_dt, 0.01*fnrm_old),bbb.ftol_min)
                bbb.exmain()
                if (bbb.iterm == 1):
                    bbb.ylodt = bbb.yl
                    bbb.pandf1 (-1, -1, 0, bbb.neq, 1., bbb.yl, bbb.yldot)
                    fnrm_old = np.sqrt(sum((bbb.yldot[0:bbb.neq-1]*bbb.sfscal[0:bbb.neq-1])**2))
                    print("Total time = {:.4E}; Timestep = {:.4E}".format(bbb.dt_tot,bbb.dtreal))
                    print("variable index ipt = {} bbb.yl[ipt] = {:.4E}".format(ipt,bbb.yl[ipt]))
                    dtreal_sav = bbb.dtreal
                    bbb.dt_tot += bbb.dtreal
                    nfe_tot += bbb.nfe[0,0]
                    if storeExt:
                        for v in storeExtVars:
                            h5storeExt[str(iStoreExt) + '/' + v] = np.copy(getattr(bbb, v))
                        iStoreExt += 1
                    if saveIntermediates:
                        if os.path.exists(savedir):        
                            uedge.hdf5.hdf5_save('{}/{}_last_ii2.h5'.format(savedir,bbb.label[0].decode('UTF-8')))
                        else:
                            # print('Folder {} not found, saving output to cwd...'.format(savedir))
                            uedge.hdf5.hdf5_save('{}_last_ii2.h5'.format(bbb.label[0].decode('UTF-8')))
                    if saveAll:
                        uedge.hdf5.hdf5_save('{}/{}_dt_tot{}.h5'.format(savedir, bbb.label[0].decode('UTF-8'), bbb.dt_tot))
                        
                        
                    if (bbb.dt_tot>=0.999999999999*bbb.t_stop  or  fnrm_old<bbb.ftol_min):
                        print(' ')
                        print('*****************************************************')
                        print('**  SUCCESS: frnm < bbb.ftol; or dt_tot >= t_stop  **')
                        print('*****************************************************')
                        break
                    print(" ")
    ##       Store variables if a storage time has been crossed
                    if (bbb.dt_tot >= tstor[0]+dt_stor*i_stor and i_stor<bbb.n_stor):
                        # Check if variables are already present
                        for var in storevar:
                            if var[0]=='nfe':       var[1]=nfe_tot          # Update non-pointer variable
                            if var[0]=='dtreal':    var[1]=copy(bbb.dtreal) # Update variable: unsure why pointer does not update
                            if var[0]=='dt_tot':    var[1]=copy(bbb.dt_tot) # Update variable: unsure why pointer does not update
                            # Check if array initialized
                            if var[0] in data.keys():
                                data[var[0]]=np.append(data[var[0]],np.expand_dims(np.array(copy(var[1])),axis=0),axis=0)
                            else:
                                data[var[0]]=np.expand_dims(np.array(copy(var[1])),axis=0)
                                                

                        i_stor = i_stor + 1
       ##          End of storage section
        if timeoutMins:
            if walltime > datetime.timedelta(minutes=timeoutMins):
                print('Case exceeded rundtp timeout. Terminating...')
                bbb.iterm = 2
                return
          
        if (bbb.dt_tot>=bbb.t_stop  or  fnrm_old<bbb.ftol_min): break   # need for both loops
        bbb.irev = bbb.irev-1
        if (bbb.iterm != 1):	#print bad eqn, cut dtreal by 3, set irev flag
            itroub()
            
            if (bbb.dtreal < bbb.dt_kill):
                print('\n*************************************')
                print('**  FAILURE: time-step < dt_kill   **')
                print('*************************************')
                break
            bbb.irev = 1
            print('*** Converg. fails for bbb.dtreal; reduce time-step by 3, try again')
            print('-------------------------------------------------------------------- ')
            bbb.dtreal = bbb.dtreal/(3*bbb.mult_dt)
            bbb.dtphi = bbb.rdtphidtr*bbb.dtreal
            if (bbb.ismfnkauto==1 and bbb.dtreal > bbb.dtmfnk3): bbb.mfnksol = -3
            bbb.deldt =  bbb.deldt/(3*bbb.mult_dt) 
            bbb.iterm = 1
    
    if ii1 == ii1max:
        print('rundtp reached ii1max. Terminating...')
        bbb.iterm = 2
                
    walltime = datetime.timedelta(seconds=int(time.time()-start_time))
    print('Wall time:', walltime)
    print('Simulation time:', bbb.dt_tot)
    
    if storeExt:
        h5storeExt.close()
    
    if debug:
        try:
            with h5py.File('debug' + str(bbb.label[0]) + '.h5', 'w') as hf: 
                hf.create_dataset('nmaxes', data=nmaxes)
                hf.create_dataset('nmins', data=nmins)
                hf.create_dataset('ngmaxes', data=ngmaxes)
                hf.create_dataset('ngmins', data=ngmins)
                hf.create_dataset('temaxes', data=temaxes)
                hf.create_dataset('temins', data=temins)
                hf.create_dataset('timaxes', data=timaxes)
                hf.create_dataset('timins', data=timins)
                hf.create_dataset('times', data=times)
        except:
            pass
        plt.figure(figsize=(10,10))
        plt.subplot(421)
        plt.plot(times, nmaxes)
        plt.yscale('log')
        plt.ylabel('ni max')
        plt.subplot(422)
        plt.plot(times, nmins)
        plt.yscale('log')
        plt.ylabel('ni min')
        plt.subplot(423)
        plt.plot(times, ngmaxes)
        plt.yscale('log')
        plt.ylabel('ng max')
        plt.subplot(424)
        plt.plot(times, ngmins)
        plt.yscale('log')
        plt.ylabel('ng min')
        plt.subplot(425)
        plt.plot(times, temaxes)
        plt.yscale('log')
        plt.ylabel('te max')
        plt.subplot(426)
        plt.plot(times, temins)
        plt.yscale('log')
        plt.ylabel('te min')
        plt.subplot(427)
        plt.plot(times, timaxes)
        plt.yscale('log')
        plt.ylabel('ti max')
        plt.subplot(428)
        plt.plot(times, timins)
        plt.yscale('log')
        plt.ylabel('ti min')
        plt.tight_layout()
        plt.show()

    if bbb.iterm!=1:
        print('Unconverged case dropped out of loop: try again! Terminating...')
        if notifyWhenDone and time.time()-start_time > 5*60:
            notify('Run complete %s %.2g s' % (walltime, bbb.dt_tot))
        return

    # Save the data to HDF5
    if n_stor>0:
        if os.path.exists(savedir):        
            save_dt('{}/dt_{}.h5'.format(savedir,bbb.label[0].decode('UTF-8')),data)
        else:
            print('Folder {} not found, saving output to cwd...'.format(savedir))
            save_dt('dt_{}.h5'.format(bbb.label[0].decode('UTF-8')),data)
    
    # Restore the original values
    # bbb.dt_tot=dt_tot_o
    bbb.ii1max=ii1max_o
    bbb.ii2max=ii2max_o
    bbb.ftol_dt=ftol_dt_o 
    bbb.itermx=itermx_o   
    bbb.rlx=rlx_o    
    bbb.n_stor=n_stor_o   
    bbb.tstor_s=tstor_s_o  
    bbb.tstor_e=tstor_e_o 
    bbb.incpset=incpset_o 
    bbb.dtmfnk3=dtmfnk3_o
    bbb.icntnunk=icntnunk_o
    bbb.ftol=ftol_o
    bbb.dtreal=1e20
    
    # Calculate impurity radiation if impurities are on
    if bbb.isimpon != 0:
        bbb.pradpltwl()
        
    if notifyWhenDone and time.time()-start_time > 5*60:
        notify('Run complete %s %.2g s' % (walltime, bbb.dt_tot))


def itroub():
    ''' Function that displays information on the problematic equation '''
    from numpy import mod,argmax
    from uedge import bbb
    # Set scaling factor
    scalfac = bbb.sfscal
    if (bbb.svrpkg[0].decode('UTF-8').strip() != "nksol"): scalfac = 1/(bbb.yl + 1.e-30)  # for time-dep calc.

    # Find the fortran index of the troublemaking equation
    itrouble=argmax(abs(bbb.yldot[:bbb.neq]))+1
    print("** Fortran index of trouble making equation is:")
    print(itrouble)

    # Print equation information
    print("** Number of equations solved per cell:")
    print("numvar = {}\n".format(bbb.numvar))
    iv_t = mod(itrouble-1,bbb.numvar) + 1 # Use basis indexing for equation number
    print("** Troublemaker equation is:")
    # Verbose troublemaker equation
    if abs(bbb.idxte-itrouble).min()==0:
        print('Electron energy equation: iv_t={}'.format(iv_t))           
    elif abs(bbb.idxti-itrouble).min()==0:
        print('Ion energy equation: iv_t={}'.format(iv_t))   
    elif abs(bbb.idxphi-itrouble).min()==0:
        print('Potential equation: iv_t={}'.format(iv_t))   
    elif abs(bbb.idxu-itrouble).min()==0:
        for species in range(bbb.idxu.shape[2]):
            if abs(bbb.idxu[:,:,species]-itrouble).min()==0:
                print('Ion momentum equation of species {}: iv_t={}'.format(species, iv_t))   
    elif abs(bbb.idxn-itrouble).min()==0:
        for species in range(bbb.idxn.shape[2]):
            if abs(bbb.idxn[:,:,species]-itrouble).min()==0:
                print('Ion density equation of species {}: iv_t={}'.format(species, iv_t))   
    elif abs(bbb.idxg-itrouble).min()==0:
        for species in range(bbb.idxg.shape[2]):
            if abs(bbb.idxg[:,:,species]-itrouble).min()==0:
                print('Gas density equation of species {}: iv_t={}'.format(species, iv_t))   
    elif abs(bbb.idxtg-itrouble).min()==0:
        for species in range(bbb.idxtg.shape[2]):
            if abs(bbb.idxtg[:,:,species]-itrouble).min()==0:
                print('Gas temperature equation of species {}: iv_t={}'.format(species, iv_t))   
    # Display additional information about troublemaker cell
    print("\n** Troublemaker cell (ix,iy) is:")
    print(bbb.igyl[itrouble-1,])
    print("\n** Timestep for troublemaker equation:")
    print(bbb.dtuse[itrouble-1])
    print("\n** yl for troublemaker equation:")
    print(bbb.yl[itrouble-1], '\n')


def save_dt(file,data):
    ''' 
    Save HDF5 file containing time-evolution of restore parameters and time
    Created by holm10 based on meyer8's hdf5.py
    '''
    from time import ctime
    from uedge import bbb

    # Open file for writing
    try:
        hf = h5py.File(file,'w')        # Open file for writing
        hfb = hf.create_group('globals')# Create group for dt data
        hfb.attrs['date'] = ctime()
        hfb.attrs['code'] = 'UEDGE'
        hfb.attrs['ver'] = bbb.uedge_ver
    except ValueError as error:
        print("HDF5 file open failed to {}".format(file))
        print(error)
    except:
        print("HDF5 file open failed to {}".format(file))
        raise

    # Store variables from dictionary data
    for var in ['dt_tot','dtreal','nfe','ni','up','te','ti','tg','ng','phi']:
        try:
            hfb.create_dataset(var, data=data[var])
        except ValueError as error:
            print("{} HDF5 file write failed to {}".format(var,file))
            print(error)
        except:
            print("{} HDF5 file write failed to {}".format(var,file))


def restore_dt(file,ReturnDict=True):
    ''' 
    Restore HDF5 file containing time-evolution of restore parameters and time
    Created by holm10 based on meyer8's hdf5.py
    '''
    from numpy import array    

    data=dict() # Empty dictionary for storage
    
    # Open file for reading
    try:
        hf = h5py.File(file,'r')        # Open file for reading
    except ValueError as error:
        print("HDF5 file open failed to {}".format(file))
        print(error)
        return
    except:
        print("HDF5 file open failed to {}".format(file))
        return

    try:
        dummy = hf['globals']    # Force exception if group not found
        hfb=hf.get('globals')


    except ValueError as error:
        print("HDF5 file could not find group 'globals' in {}".format(file))
        print(error)
        return
    except:
        print("HDF5 file could not find group 'globals' in {}".format(file))
        return
    # Print information on save
    print('Restored time-dependent data for {} case written {} using version {}'.format(hfb.attrs['code'], hfb.attrs['date'],hfb.attrs['ver'][0].decode('UTF-8').strip()[7:-1].replace('_','.')))
    # Loop over all variables
    for var in ['dt_tot','dtreal','nfe','ni','up','te','ti','tg','ng','phi']:
        try:
            data[var] = np.array(hfb.get(var))
        except ValueError as error:
            print("Couldn't read {} from {}".format(var,file))
            print(error)
        except:
            print("Couldn't read {} from {}".format(var,file))     
 
    if ReturnDict:
        return data
    else:
        return data['dt_tot'],data['dtreal'],data['nfe'],data['ni'],data['up'],data['te'],data['ti'],data['tg'],data['ng'],data['phi']

    
def paws(message=""):
    print(message)
    input("Press the <ENTER> key to continue...")
    

def yesno(question):
    '''
    Prompt the user to answer a question with y or n, returning True/False.
    Args:
        question: (str) text to display before asking for y/n input
    '''
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return bool(strtobool(input().lower()))
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')
