import numpy as np
import matplotlib.pyplot as plt

import random
from priors import *
import fnmatch
import pdb

def plotinit():
    fig1 = plt.figure('posteriors',figsize=(8,12))
    fig2 = plt.figure('hrd',figsize=(12,12))
    plt.figure('posteriors')
 
def plotposterior(x,y,res,err1,err2,names,j,ix,iy):
    #fig = plt.figure(figsize=(8,12))
    plt.subplot(len(names),2,ix)
    plt.plot(x,np.cumsum(y))
    plt.plot([res,res],[0,1],'r')
    plt.plot([res+err1,res+err1],[0,1],'--r')
    plt.plot([res-err2,res-err2],[0,1],'--r')
    plt.ylim([0,1])
    plt.title(names[j])
    if fnmatch.fnmatch(names[j],'*rho*'):
        plt.xscale('log')
    if fnmatch.fnmatch(names[j],'*lum*'):
        plt.xscale('log')


    plt.subplot(len(names),2,iy)
    plt.plot(x,y)
    plt.plot([res,res],[0,1],'r')
    plt.plot([res+err1,res+err1],[0,1],'--r')
    plt.plot([res-err2,res-err2],[0,1],'--r')
    plt.ylim([0,np.max(y)+np.max(y)*0.1])
    plt.title(names[j])
    if fnmatch.fnmatch(names[j],'*rho*'):
        plt.xscale('log')
    if fnmatch.fnmatch(names[j],'*lum*'):
        plt.xscale('log')

    if fnmatch.fnmatch(names[j],'*feh*'):
        xt=np.arange(-2.,1.,0.01)
        yt=fehprior(xt)
        plt.plot(xt,yt*np.max(y)/np.max(yt),'--g')

def plothrd(model,mod,modsel,input,mabs,mabse,ix,iy):
    plt.subplots_adjust(
        left=0.08, bottom=0.05, right=0.96, top=0.96, wspace=0.31, hspace=0.26
    )

    plt.figure('hrd')
    plt.subplot(2,4,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    ### Sloan color-color
    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.plot(mod['gmag'][modsel]-mod['rmag'][modsel],\
             mod['rmag'][modsel]-mod['imag'][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.gmag > -99) & (input.rmag > -99) & (input.imag > -99)):
        plt.errorbar([input.gmag-input.rmag], [input.rmag-input.imag],xerr=np.sqrt(input.gmage**2+input.rmage**2),yerr=np.sqrt(input.rmage**2+input.image**2),color='green',elinewidth=5)

    plt.xlabel('g-r')
    plt.ylabel('r-i')
    #plt.xlim([-0.5,2.5])
    #plt.ylim([-0.5,2])
    plt.autoscale()


    '''### Sloan color-magnitude
    plt.subplot(2,3,2)
    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['imag'][ran[d]]-model['zmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['imag'][ran[g]]-model['zmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    if ((input.gmag > -99) & (input.rmag > -99) & (input.imag > -99) & (input.zmag > -99)):
        plt.errorbar([input.gmag-input.rmag], [input.imag-input.zmag], \
                 xerr=np.sqrt(input.gmage**2+input.rmage**2), \
                 yerr=np.sqrt(input.image**2+input.zmage**2),color='green',elinewidth=5)

    plt.xlabel('g-r')
    plt.ylabel('i-z')
    plt.xlim([-0.5,2.5])
    plt.ylim([-0.5,2])'''
    
    ### BtVtJH color-color
    plt.subplot(2,4,2)
    plt.plot(model['btmag'][ran[d]]-model['vtmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    plt.xlim([-0.1,2.5])
    plt.ylim([-0.2,1.2])
    plt.plot(model['btmag'][ran[g]]-model['vtmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.plot(mod['btmag'][modsel]-mod['vtmag'][modsel],\
             mod['jmag'][modsel]-mod['hmag'][modsel],\
    '.',color='black',markersize=1,zorder=-32)
    
    if ((input.jmag > -99) & (input.hmag > -99) & (input.vtmag > -99) & (input.btmag > -99)):
        plt.errorbar([input.btmag-input.vtmag], [input.jmag-input.hmag], \
                 xerr=np.sqrt(input.btmage**2+input.vtmage**2), \
                 yerr=np.sqrt(input.jmage**2+input.hmage**2),color='green',elinewidth=5)

    plt.xlabel('Bt-Vt')
    plt.ylabel('J-H')
    plt.autoscale()

    ### 2MASS JHK color-color
    plt.subplot(2,4,3)
    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.plot(mod['hmag'][modsel]-mod['kmag'][modsel],\
             mod['jmag'][modsel]-mod['hmag'][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.jmag > -99) & (input.hmag > -99) & (input.kmag > -99)):
        plt.errorbar([input.hmag-input.kmag], [input.jmag-input.hmag], \
                 xerr=np.sqrt(input.hmage**2+input.kmage**2), \
                 yerr=np.sqrt(input.jmage**2+input.hmage**2),color='green',elinewidth=5)

    plt.xlabel('H-K')
    plt.ylabel('J-H')
    plt.autoscale()
    #plt.xlim([-0.1,0.5])
    #plt.ylim([-0.3,1.3])

    # 2MASS K, g-r CMD
    plt.subplot(2,4,4)
    mag1='gmag'
    mag2='rmag'
    absmag='kmag'
    col=input.gmag - input.rmag
    cole=np.sqrt(input.gmage**2+input.rmage**2)

    plt.plot(model[mag1][ran[d]]-model[mag2][ran[d]],\
             model[absmag][ran[d]],'.',color='blue',markersize=1,zorder=-32)

    plt.plot(model[mag1][ran[g]]-model[mag2][ran[g]], \
             model[absmag][ran[g]],'.',color='red',markersize=1,zorder=-32)

    plt.plot(mod[mag1][modsel]-mod[mag2][modsel],\
             mod[absmag][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.plx > 0.) & (input.kmag > -99) & (input.gmag > -99) & (input.rmag > -99)):
        plt.errorbar([col], [mabs], xerr=cole, yerr=mabse,color='green',elinewidth=5)

    #plt.xlim([np.min(model[absmag]),np.max(model[absmag])])
    #plt.ylim([np.max(model[absmag]),np.min(model[absmag])])
    plt.autoscale()
    plt.gca().invert_yaxis()
    plt.xlabel(mag1+'-'+mag2)
    plt.ylabel(absmag)

    # 2MASS k, g-K CMD
    plt.subplot(2,4,5)
    mag1='gmag'
    mag2='kmag'
    absmag='kmag'
    col=input.gmag - input.kmag
    cole=np.sqrt(input.gmage**2+input.kmage**2)

    plt.plot(model[mag1][ran[d]]-model[mag2][ran[d]],\
             model[absmag][ran[d]],'.',color='blue',markersize=1,zorder=-32)

    plt.plot(model[mag1][ran[g]]-model[mag2][ran[g]], \
             model[absmag][ran[g]],'.',color='red',markersize=1,zorder=-32)

    plt.plot(mod[mag1][modsel]-mod[mag2][modsel],\
             mod[absmag][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.plx > 0.) & (input.kmag > -99) & (input.gmag > -99)):
        plt.errorbar([col], [mabs], xerr=cole, yerr=mabse,color='green',elinewidth=5)

    #plt.xlim([np.min(model[absmag]),np.max(model[absmag])])
    #plt.ylim([np.max(model[absmag]),np.min(model[absmag])])
    plt.autoscale()
    plt.gca().invert_yaxis()
    plt.xlabel(mag1+'-'+mag2)
    plt.ylabel(absmag)
    
    # 2MASS k, r-K CMD
    plt.subplot(2,4,6)
    
    mag1='rmag'
    mag2='kmag'
    absmag='kmag'
    col=input.rmag - input.kmag
    cole=np.sqrt(input.rmage**2+input.kmage**2)

    plt.plot(model[mag1][ran[d]]-model[mag2][ran[d]],\
             model[absmag][ran[d]],'.',color='blue',markersize=1,zorder=-32)

    plt.plot(model[mag1][ran[g]]-model[mag2][ran[g]], \
             model[absmag][ran[g]],'.',color='red',markersize=1,zorder=-32)

    plt.plot(mod[mag1][modsel]-mod[mag2][modsel],\
             mod[absmag][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.plx > 0.) & (input.kmag > -99) & (input.rmag > -99)):
        plt.errorbar([col], [mabs], xerr=cole, yerr=mabse,color='green',elinewidth=5)

    #plt.xlim([np.min(model[absmag]),np.max(model[absmag])])
    #plt.ylim([np.max(model[absmag]),np.min(model[absmag])])
    plt.autoscale()
    plt.gca().invert_yaxis()
    plt.xlabel(mag1+'-'+mag2)
    plt.ylabel(absmag)

    # 2MASS k, Vt-K CMD
    plt.subplot(2,4,7)
    mag1='vtmag'
    mag2='kmag'
    absmag='kmag'
    col=input.vtmag - input.kmag
    cole=np.sqrt(input.vtmage**2+input.kmage**2)

    plt.plot(model[mag1][ran[d]]-model[mag2][ran[d]],\
             model[absmag][ran[d]],'.',color='blue',markersize=1,zorder=-32)

    plt.plot(model[mag1][ran[g]]-model[mag2][ran[g]], \
             model[absmag][ran[g]],'.',color='red',markersize=1,zorder=-32)

    plt.plot(mod[mag1][modsel]-mod[mag2][modsel],\
             mod[absmag][modsel],\
    '.',color='black',markersize=1,zorder=-32)

    if ((input.plx > 0.) & (input.kmag > -99) & (input.vtmag > -99)):
        plt.errorbar([col], [mabs], xerr=cole, yerr=mabse,color='green',elinewidth=5)

    #plt.xlim([np.min(model[absmag]),np.max(model[absmag])])
    #plt.ylim([np.max(model[absmag]),np.min(model[absmag])])
    plt.autoscale()
    plt.gca().invert_yaxis()
    plt.xlabel(mag1+'-'+mag2)
    plt.ylabel(absmag)

    # HRD
    plt.subplot(2,4,8)

    if (input.numax == -99):
        plt.plot(model['teff'][ran[d]],model['logg'][ran[d]],\
                 '.',color='blue',markersize=1,zorder=-32)
        plt.xlim([10000,2000])
        plt.ylim([6,0])
	plt.yscale('linear')
        plt.plot(model['teff'][ran[g]],model['logg'][ran[g]],\
                 '.',color='red',markersize=1,zorder=-32)

	plt.plot(mod['teff'][modsel],\
             mod['logg'][modsel],\
    '.',color='black',markersize=1,zorder=-32)

        plt.errorbar([input.teff], [input.logg], xerr=input.teffe, yerr=input.logge, \
                 color='green',elinewidth=5)
	plt.xlabel('teff')
	plt.ylabel('logg')

    else:
        mod_numax=3090*(10**model['logg']/27420.)*(model['teff']/5777.)**(-0.5)
        plt.semilogy(model['teff'][ran[d]],mod_numax[ran[d]],\
                 '.',color='blue',markersize=1,zorder=-32)
	#plt.semilogy(modelsel['teff'],modelsel['numax'],'.',color='black',markersize=1,zorder=-32)
        plt.xlim([10000,2000])
        plt.ylim([100000,0.1])
        plt.plot(model['teff'][ran[g]],mod_numax[ran[g]],\
                 '.',color='red',markersize=1,zorder=-32)

        plt.errorbar([input.teff], [input.numax], xerr=input.teffe, yerr=input.numaxe, \
                 color='green',elinewidth=5)
    plt.tight_layout()

def plothrdold(model,grcol,ricol,grcole,ricole,Mg,Mge,ix,iy):

    plt.figure('hrd')
    plt.subplot(3,1,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.errorbar([grcol], [ricol], xerr=grcole, yerr=ricole,color='green',elinewidth=5)
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.xlim([-0.5,2.5])
    plt.ylim([-0.5,2])


    plt.subplot(3,1,2)
    plt.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    plt.xlim([-0.1,0.4])
    plt.ylim([-0.2,1.2])

    plt.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    plt.xlabel('H-K')
    plt.ylabel('J-H')
    
    #ran=np.array(random.sample(range(len(model_red['teff'])),\
    #int(len(model_red['teff'])*frac)))

    '''
    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['rmag'][ran]-model_red['imag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.max(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.min(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
   '''

    plt.subplot(3,1,3)
    #plt.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=15)

    #ran=np.array(random.sample(range(len(model['teff'])),\
    #int(len(model['teff'])*frac)))
    
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['gmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['gmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    '''
    ran=np.array(random.sample(range(len(model_red['teff'])),\
    int(len(model_red['teff'])*frac)))

    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['gmag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.max(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.min(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
    '''

    plt.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=5)

    plt.xlim([-0.5,2])
    plt.ylim([15,-5])
    plt.xlabel('g-r')
    plt.ylabel('Mg')


def plothrd2(x,y,res,err1,err2,avs,model,model_red,names,j,medav,stdav,grcol,ricol,grcole,ricole,plx,plxe,ix,iy,model_plx):

    plt.figure('hrd')
    plt.subplot(3,1,1)
    frac=0.01

    ran=np.array(random.sample(range(len(model['teff'])),\
    int(len(model['teff'])*frac)))

    d=np.where(model['logg'][ran] > 3.5)[0]
    plt.plot(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model['rmag'][ran[d]]-model['imag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    g=np.where(model['logg'][ran] < 3.5)[0]
    plt.plot(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model['rmag'][ran[g]]-model['imag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)
    plt.errorbar([grcol], [ricol], xerr=grcole, yerr=ricole,color='green',elinewidth=5)
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.xlim([-0.5,2.5])
    plt.ylim([-0.5,2])


    plt.subplot(3,1,2)
    plt.plot(model['hmag'][ran[d]]-model['kmag'][ran[d]],\
             model['jmag'][ran[d]]-model['hmag'][ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)
    plt.xlim([-0.1,0.4])
    plt.ylim([-0.2,1.2])

    plt.plot(model['hmag'][ran[g]]-model['kmag'][ran[g]],\
             model['jmag'][ran[g]]-model['hmag'][ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    plt.xlabel('H-K')
    plt.ylabel('J-H')
    
    #ran=np.array(random.sample(range(len(model_red['teff'])),\
    #int(len(model_red['teff'])*frac)))

    '''
    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['rmag'][ran]-model_red['imag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.max(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.min(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['rmag'][ran[um]]-model_red['imag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
   '''

    plt.subplot(3,1,3)
    #plt.errorbar([grcol], [Mg], xerr=grcole, yerr=Mge,color='green',elinewidth=15)

    #ran=np.array(random.sample(range(len(model['teff'])),\
    #int(len(model['teff'])*frac)))
    
    plt.semilogy(model['gmag'][ran[d]]-model['rmag'][ran[d]],\
             model_plx[ran[d]],\
    '.',color='blue',markersize=1,zorder=-32)

    plt.semilogy(model['gmag'][ran[g]]-model['rmag'][ran[g]],\
             model_plx[ran[g]],\
    '.',color='red',markersize=1,zorder=-32)

    '''
    ran=np.array(random.sample(range(len(model_red['teff'])),\
    int(len(model_red['teff'])*frac)))

    plt.plot(model_red['gmag'][ran]-model_red['rmag'][ran],\
    model_red['gmag'][ran]\
    ,'.',color='red',markersize=3,zorder=-32)
    '''
    '''
    um=np.where(model_red['avs'][ran] == np.max(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='blue',markersize=3,zorder=-32)

    um=np.where(model_red['avs'][ran] == np.min(avs))
    plt.plot(model_red['gmag'][ran[um]]-model_red['rmag'][ran[um]],\
    model_red['gmag'][ran[um]]\
    ,'.',color='yellow',markersize=3,zorder=-32)
    '''

    plt.errorbar([grcol], [plx], xerr=grcole, yerr=plxe,color='green',elinewidth=5)

    plt.xlim([-0.5,2])
    plt.ylim([np.max(model_plx),np.min(model_plx)])
    plt.xlabel('g-r')
    plt.ylabel('plx')
