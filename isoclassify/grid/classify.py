import copy
import time

import ephem
import pandas as pd
import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d

from pdf import *  # part of isoclassify package (to do make explicit import) 
from priors import * # part of isoclassify package (to do make explicit import) 
from .plot import * # part of isoclassify package (to do make explicit import)

class obsdata():
    def __init__(self):
        self.plx = -99.0
        self.plxe = -99.0
    
        self.teff = -99.0
        self.teffe = -99.0
        self.logg = -99.0
        self.logge = -99.0
        self.feh = -99.0
        self.fehe = -99.0
        
        self.bmag = -99.0
        self.bmage = -99.0
        self.vmag = -99.0
        self.vmage = -99.0

        self.btmag = -99.0
        self.btmage = -99.0
        self.vtmag = -99.0
        self.vtmage = -99.0

	self.umag = -99.0
	self.umage = -99.0
        self.gmag = -99.0
        self.gmage = -99.0
        self.rmag = -99.0
        self.rmage = -99.0
        self.imag = -99.0
        self.image = -99.0
        self.zmag = -99.0
        self.zmage = -99.0

        self.jmag = -99.0
        self.jmage = -99.0
        self.hmag = -99.0
        self.hmage = -99.0
        self.kmag = -99.0
        self.kmage = -99.0

	self.gamag = -99.0
	self.bpmag = -99.0
	self.rpmag = -99.0
        
        self.numax = -99.0
        self.numaxe = -99.0
        self.dnu = -99.0
        self.dnue = -99.0

	self.evstate = -99.0
	self.mdwarfbool = -99.0
	self.outdir = ''

	self.reddenmap = -99.0
                   
    def addspec(self,value,sigma):
        self.teff = value[0]
        self.teffe = sigma[0]
        self.logg = value[1]
        self.logge = sigma[1]
        self.feh = value[2]
        self.fehe = sigma[2]
               
    def addbv(self,value,sigma):
        self.bmag = value[0]
        self.bmage = sigma[0]
        self.vmag = value[1]
        self.vmage = sigma[1]

    def addbvt(self,value,sigma):
        self.btmag = value[0]
        self.btmage = sigma[0]
        self.vtmag = value[1]
        self.vtmage = sigma[1]
        
    def addugriz(self,value,sigma):
	self.umag = value[0]
	self.umage = sigma[0]
        self.gmag = value[1]
        self.gmage = sigma[1]
        self.rmag = value[2]
        self.rmage = sigma[2]
        self.imag = value[3]
        self.image = sigma[3]
        self.zmag = value[4]
        self.zmage = sigma[4]
        
    def addjhk(self,value,sigma):
        self.jmag = value[0]
        self.jmage = sigma[0]
        self.hmag = value[1]
        self.hmage = sigma[1]
        self.kmag = value[2]
        self.kmage = sigma[2]

    def addgabprp(self,value,sigma):
        self.gamag = value[0]
        self.gamage = sigma[0]
        self.bpmag = value[1]
        self.bpmage = sigma[1]
        self.rpmag = value[2]
        self.rpmage = sigma[2]
        
    def addplx(self,value,sigma):
        self.plx = value
        self.plxe = sigma
        
    def addseismo(self,value,sigma):
        self.numax = value[0]
        self.numaxe = sigma[0]
        self.dnu = value[1]
        self.dnue = sigma[1]
	
    def addcoords(self,value1,value2):
        self.ra = value1
        self.dec = value2

    def addevstate(self,value):
	self.evstate = value

    def addmdwarfbool(self,value):
	self.mdwarfbool = value

    def addoutdir(self,value):
	self.outdir = value

class resdata():
    def __init__(self):
        self.teff = 0.0
        self.teffep = 0.0
        self.teffem = 0.0
        self.teffpx = 0.0
        self.teffpy = 0.0
        self.logg = 0.0
        self.loggep = 0.0
        self.loggem = 0.0
        self.loggpx = 0.0
        self.loggpy = 0.0
        self.feh = 0.0
        self.fehep = 0.0
        self.fehem = 0.0
        self.fehpx = 0.0
        self.fehpy = 0.0
        self.rad = 0.0
        self.radep = 0.0
        self.radem = 0.0
        self.radpx = 0.0
        self.radpy = 0.0
        self.mass = 0.0
        self.massep = 0.0
        self.massem = 0.0
        self.masspx = 0.0
        self.masspy = 0.0
        self.rho = 0.0
        self.rhoep = 0.0
        self.rhoem = 0.0
        self.rhopx = 0.0
        self.rhopy = 0.0
        self.lum = 0.0
        self.lumep = 0.0
        self.lumem = 0.0
        self.lumpx = 0.0
        self.lumpy = 0.0
        self.age = 0.0
        self.ageep = 0.0
        self.ageem = 0.0
        self.agepx = 0.0
        self.agepy = 0.0
        self.avs = 0.0
        self.avsep = 0.0
        self.avsem = 0.0
        self.avspx = 0.0
        self.avspy = 0.0
        self.dis = 0.0
        self.disep = 0.0
        self.disem = 0.0
        self.dispx = 0.0
        self.dispy = 0.0

	self.gof = 0.0
	self.gofep = 0.0
	self.gofem = 0.0
	self.gofpx = 0.0
	self.gofpy = 0.0

class extinction():
    def __init__(self):
        self.ab = 1.3454449
        self.av = 1.00

        self.abt = 1.3986523
        self.avt = 1.0602271

        self.ag = 1.2348743
        self.ar = 0.88343449
        self.ai = 0.68095687
        self.az = 0.48308430

        self.aj = 0.28814896
        self.ah = 0.18152716
        self.ak = 0.11505195

        self.aga=1.2348743


def classify(input, model, dustmodel=0, plot=1, useav=-99.0, ext=-99.0):
    """
    Run grid based classifier

    Args:
        input (object): input object
        model (dict): dictionary of arrays
        dustmodel (Optional[DataFrame]): extinction model
        useav (float):
        ext (float):
    """

    ## constants
    gsun = 27420.010
    numaxsun = 3090.0
    dnusun = 135.1
    teffsun = 5772.0

    # bolometric correction error; kinda needs to be motivated better ...
    bcerr = 0.03

    ## extinction coefficients
    extfactors = ext
    
    ## class containing output results
    result = resdata()

    # calculate colors + errors
    bvcol = input.bmag - input.vmag
    bvtcol = input.btmag - input.vtmag
    grcol = input.gmag - input.rmag
    ricol = input.rmag - input.imag 
    izcol = input.imag - input.zmag
    jhcol = input.jmag - input.hmag
    hkcol = input.hmag - input.kmag
    gkcol = input.gmag - input.kmag
    vtkcol = input.vtmag - input.kmag
    #rkcol = input.rmag - input.kmag
    gacol = input.gamag - input.kmag
    brpcol = input.bpmag - input.rpmag
    bpkcol = input.bpmag - input.kmag
    rjcol = input.rmag - input.jmag

    bvcole = np.sqrt(input.bmage**2 + input.vmage**2)
    bvtcole = np.sqrt(input.btmage**2 + input.vtmage**2)
    grcole = np.sqrt(input.gmage**2 + input.rmage**2)
    ricole = np.sqrt(input.rmage**2 + input.image**2)
    izcole = np.sqrt(input.image**2 + input.zmage**2)
    jhcole = np.sqrt(input.jmage**2 + input.hmage**2)
    hkcole = np.sqrt(input.hmage**2 + input.kmage**2)
    gkcole = np.sqrt(input.gmage**2 + input.kmage**2)
    vtkcole = np.sqrt(input.vtmage**2 + input.kmage**2)
    #rkcole = np.sqrt(input.rmage**2 + input.kmage**2)
    gakcole = np.sqrt(input.gamage**2 + input.kmage**2)
    brpcole = np.sqrt(input.bpmage**2 + input.rpmage**2)
    bpkcole = np.sqrt(input.bpmage**2 + input.kmage**2)
    rjcole = np.sqrt(input.rmage**2 + input.jmage**2)

    # Compute extra color error term based on underestimation of stellar teff errors with nominal 2% error floor:
    if ((input.gmag > -99.0) & (input.kmag > -99.0)):
        gkexcole = compute_extra_gk_color_error(gkcol)
	# Determine which gK error term is greater and use that one:
        print "g - K error from photometry: ",gkcole
        print "g - K error from best-fit polynomial: ",gkexcole
        gkcole = max(gkcole,gkexcole)
        print "Using g - K error: ",gkcole
    #rjexcole = compute_extra_rj_color_error(rjcol) 

    # Determine which rj error term is greater and use that one:
    #print "r - J error from photometry: ",rjcole
    #print "r - J error from best-fit polynomial: ",rjexcole
    #rjcole = max(rjcole,rjexcole)
    #print "Using r - J error: ",rjcole

    # determine apparent mag to use for distance estimation. K>J>G>Vt>V>g
    redmap = -99.0
    if (input.gmag > -99.0):
        redmap = input.gmag
        mape = input.gmage
        model_mabs = model['gmag']   
        band = 'g'

    if (input.vmag > -99.0):
        redmap = input.vmag
        mape = input.vmage
        band = 'v'
        model_mabs = model['vmag']

    if (input.vtmag > -99.0):
        redmap = input.vtmag
        mape = input.vtmage
        model_mabs = model['vtmag']
        band = 'vt'

    if (input.gamag > -99.0):
	redmap = input.gamag
	mape = input.gamage
	model_mabs = model['gamag']
	band = 'ga'
	
    if (input.jmag > -99.0):
        redmap = input.jmag
        mape = input.jmage
        model_mabs = model['jmag']
        band = 'j'

    if (input.kmag > -99.0):
        redmap = input.kmag
        mape = input.kmage
        model_mabs = model['kmag']
        band = 'k'
        
    # absolute magnitude
    if (input.plx > -99.0):
        mabs = -5.0 * np.log10(1.0 / input.plx) + redmap + 5.0
        mabse = np.sqrt(
            (-5.0 / (input.plx * np.log(10)))**2 * input.plxe**2 
            + mape**2 + bcerr**2
        )
	# Also compute extra error term for M-dwarfs with K band mags only:
	if (mabs > 4.0) and (input.gmag - input.kmag > 4.0):
	    mabseex = compute_extra_MK_error(mabs)

	    # Determine which gK error term is greater and use that one:
            print "M_K from photometry: ",mabse
            print "M_K error from best-fit polynomial: ",mabseex
            mabse = np.sqrt(mabse**2 + mabseex**2)
            print "After adding in quadrature, using M_K error: ",mabse
    else:
        mabs = -99.0
        mabse = -99.0

    # pre-select model grid; first only using reddening-independent quantities
    sig = 4.0
    um = np.arange(0,len(model['teff']),1)
        
    if (input.teff > -99.0):
        ut=np.where((model['teff'] > input.teff-sig*input.teffe) & \
        (model['teff'] < input.teff+sig*input.teffe))[0]
        um=np.intersect1d(um,ut)
        print 'teff',len(um)

    if (input.dnu > 0.0):
        model_dnu = dnusun*model['fdnu']*np.sqrt(10**model['rho'])
        ut = np.where(
            (model_dnu > input.dnu - sig*input.dnue)  
            & (model_dnu < input.dnu + sig*input.dnue)
        )
        ut = ut[0]
        um = np.intersect1d(um, ut)
        print 'dnu', len(um)

    if (input.numax > 0.0):
        model_numax = (numaxsun 
                       * (10**model['logg']/gsun)
                       * (model['teff']/teffsun)**(-0.5))
        ut = np.where(
            (model_numax > input.numax - sig*input.numaxe) 
            & (model_numax < input.numax + sig*input.numaxe)
        )
        ut = ut[0]
        um = np.intersect1d(um, ut)
        print 'numax', len(um)
        
    if (input.logg > -99.0):
        ut = np.where(
            (model['logg'] > input.logg - sig*input.logge) 
            & (model['logg'] < input.logg + sig*input.logge)
        )
        ut = ut[0]
        um = np.intersect1d(um, ut)
        print 'logg',len(um)
        
    if (input.feh > -99.0):
        ut = np.where(
            (model['feh_act'] > input.feh - sig*input.fehe)
            & (model['feh_act'] < input.feh + sig*input.fehe)
        )
        ut = ut[0]
        um = np.intersect1d(um, ut)
        print 'feh_act', len(um)
               
    print 'number of models used within non-phot obsconstraints:', len(um)

    # bail if there are not enough good models
    if (len(um) < 10):
        return result

    # add reddening
    if (redmap > -99.0):

        # if no reddening map is provided, add Av as a new variable
        # and fit for it
        if (isinstance(dustmodel,pd.DataFrame) == False):
            avs = np.arange(-0.3,1.0,0.01)
	    #pdb.set_trace()
            
            # user-specified reddening
            #if (useav > -99.0):
            #    avs = np.zeros(1) + useav
                
            mod = reddening(model, um, avs, extfactors)
	    #pdb.set_trace()

        # otherwise, just redden each model according to the provided map
        else:
            mod = reddening_map(
                model, model_mabs, redmap, dustmodel, um, input, extfactors, band
            )

        # photometry to use for distance
        if (input.vmag > -99.0):
            mod_mabs = mod['vmag']

        if (input.vtmag > -99.0):
            mod_mabs = mod['vtmag']

        if (input.gmag > -99.0):
            mod_mabs = mod['gmag']

        if (input.jmag > -99.0):
            mod_mabs = mod['jmag']

        if (input.kmag > -99.0):
            mod_mabs = mod['kmag']

        um = np.arange(0,len(mod['teff']),1)

        mod['dis'] = 10**((redmap - mod_mabs + 5.0)/5.0)
        print 'number of models incl reddening:',len(um)
    else:
        mod = model

    # next, another model down-select based on reddening-dependent quantities
    # only do this if no spec constraints are available
    if (mabs > -99.0):
        ut = np.where(
            (mod_mabs > mabs - sig*mabse)  
            & (mod_mabs < mabs + sig*mabse)
        )
        ut = ut[0]
        um = np.intersect1d(um, ut)
	#pdb.set_trace()
    #pdb.set_trace()
    if (input.teff == -99.0):
        if ((input.bmag > -99.0) & (input.vmag > -99.0)):
            ut=np.where(
                (mod['bmag'] - mod['vmag'] > bvcol - sig*bvcole) 
                & (mod['bmag'] - mod['vmag'] < bvcol + sig*bvcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)

        if ((input.btmag > -99.0) & (input.vtmag > -99.0)):
            ut=np.where(
                (mod['btmag'] - mod['vtmag'] > bvtcol - sig*bvtcole) 
                & (mod['btmag'] - mod['vtmag'] < bvtcol + sig*bvtcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)
	    #pdb.set_trace()

        if ((input.gmag > -99.0) & (input.rmag > -99.0)):
            ut = np.where(
                (mod['gmag'] - mod['rmag'] > grcol-sig*grcole) 
                & (mod['gmag'] - mod['rmag'] < grcol+sig*grcole))
            ut = ut[0]
            um = np.intersect1d(um, ut)
	    #pdb.set_trace()

        if ((input.rmag > -99.0) & (input.imag > -99.0)):
            ut = np.where(
                (mod['rmag'] - mod['imag'] > ricol - sig*ricole) 
                & (mod['rmag'] - mod['imag'] < ricol + sig*ricole)
            )
            ut = ut[0]
            um = np.intersect1d(um,ut)
	    #pdb.set_trace()

        if ((input.imag > -99.0) & (input.zmag > -99.0)):
            ut = np.where(
                (mod['imag'] - mod['zmag'] > izcol - sig*izcole) 
                & (mod['imag'] - mod['zmag'] < izcol + sig*izcole)
            )
            ut = ut[0]
            um = np.intersect1d(um, ut)

        if ((input.jmag > -99.0) & (input.hmag > -99.0)):
            ut = np.where(
                (mod['jmag'] - mod['hmag'] > jhcol - sig*jhcole) 
                & (mod['jmag'] - mod['hmag'] < jhcol + sig*jhcole)
            )
            ut = ut[0]
            um = np.intersect1d(um, ut)
	    #pdb.set_trace()

        if ((input.hmag > -99.0) & (input.kmag > -99.0)):
            ut = np.where(
                (mod['hmag'] - mod['kmag'] > hkcol - sig*hkcole) 
                & (mod['hmag'] - mod['kmag'] < hkcol + sig*hkcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)
	
	if ((input.gmag > -99.0) & (input.kmag > -99.0)):
            ut = np.where(
                (mod['gmag'] - mod['kmag'] > gkcol - sig*gkcole) 
                & (mod['gmag'] - mod['kmag'] < gkcol + sig*gkcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)

	if ((input.vtmag > -99.0) & (input.kmag > -99.0)):
            ut = np.where(
                (mod['vtmag'] - mod['kmag'] > vtkcol - sig*vtkcole) 
                & (mod['vtmag'] - mod['kmag'] < vtkcol + sig*vtkcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)

        #if ((input.rmag > -99.0) & (input.kmag > -99.0) & ():
        #    ut = np.where(
        #        (mod['rmag'] - mod['kmag'] > rkcol - sig*rkcole) 
        #        & (mod['rmag'] - mod['kmag'] < rkcol + sig*rkcole))
        #    ut = ut[0]
        #    um = np.intersect1d(um,ut)

	if ((input.rmag > -99.0) & (input.jmag > -99.0)):
            ut = np.where(
                (mod['rmag'] - mod['jmag'] > rjcol - sig*rjcole) 
                & (mod['rmag'] - mod['jmag'] < rjcol + sig*rjcole))
            ut = ut[0]
            um = np.intersect1d(um,ut)

    #pdb.set_trace()
    print 'number of models after phot constraints:',len(um)
    print '----'



    # bail if there are not enough good models
    if (len(um) < 10):
        return result

    def gaussian(x, mu, sig):
        return np.exp(-(x-mu)**2./(2.*sig**2.))


    # likelihoods and avg distances
    if ((input.gmag > -99.0) & (input.rmag > -99.0)):
        lh_gr = gaussian(grcol, mod['gmag'][um]-mod['rmag'][um], grcole)
	diff_gr = np.max(lh_gr)
    else:
        lh_gr = np.ones(len(um))
	diff_gr = 1.0

    if ((input.rmag > -99.0) & (input.imag > -99.0)):
        lh_ri = gaussian(ricol, mod['rmag'][um]-mod['imag'][um], ricole)
	diff_ri = np.max(lh_ri)
    else:
        lh_ri = np.ones(len(um))
	diff_ri = 1.0 
        
    if ((input.imag > -99.0) & (input.zmag > -99.0)):
        lh_iz = gaussian(izcol, mod['imag'][um]-mod['zmag'][um], izcole)
	diff_iz = np.max(lh_iz)
    else:
        lh_iz = np.ones(len(um))
	diff_iz = 1.0 
   
    if ((input.jmag > -99.0) & (input.hmag > -99.0)):
        lh_jh = gaussian(jhcol, mod['jmag'][um]-mod['hmag'][um], jhcole)
	diff_jh = np.max(lh_jh)
    else:
        lh_jh = np.ones(len(um))
	diff_jh = 1.0

    if ((input.hmag > -99.0) & (input.kmag > -99.0)):
        lh_hk = gaussian(hkcol, mod['hmag'][um]-mod['kmag'][um], hkcole)
	diff_hk = np.max(lh_hk)
    else:
        lh_hk = np.ones(len(um))
	diff_hk = 1.0

    if ((input.bmag > -99.0) & (input.vmag > -99.0)):
        lh_bv = gaussian(bvcol, mod['bmag'][um]-mod['vmag'][um], bvcole)
	diff_bv = np.max(lh_bv)
    else:
        lh_bv = np.ones(len(um))
	diff_bv = 1.0

    if ((input.btmag > -99.0) & (input.vtmag > -99.0)):
        lh_bvt = gaussian(bvtcol, mod['btmag'][um]-mod['vtmag'][um], bvtcole)
	diff_bvt = np.max(lh_bvt)
    else:
        lh_bvt = np.ones(len(um))
	diff_bvt = 1.0

    if ((input.gmag > -99.0) & (input.kmag > -99.0)):
        lh_gk = gaussian(gkcol, mod['gmag'][um]-mod['kmag'][um], gkcole)
	diff_gk = np.max(lh_gk)
    else:
        lh_gk = np.ones(len(um))
	diff_gk = 1.0

    if ((input.vtmag > -99.0) & (input.kmag > -99.0)):
        lh_vtk = gaussian(vtkcol, mod['vtmag'][um]-mod['kmag'][um], vtkcole)
	diff_vtk = np.max(lh_vtk)
    else:
        lh_vtk = np.ones(len(um))
	diff_vtk = 1.0

    if ((input.rmag > -99.0) & (input.kmag > -99.0)):
        lh_rk = gaussian(rkcol, mod['rmag'][um]-mod['kmag'][um], rkcole)
    	diff_rk = np.max(lh_rk)
    else:
        lh_rk = np.ones(len(um))
        diff_rk = 1.0

    if ((input.rmag > -99.0) & (input.jmag > -99.0)):
        lh_rj = gaussian(rjcol, mod['rmag'][um]-mod['jmag'][um], rjcole)
	diff_rj = np.max(lh_rj)
    else:
        lh_rj = np.ones(len(um))
	diff_rj = 1.0

    if (input.teff > -99):
        lh_teff = gaussian(input.teff, mod['teff'][um], input.teffe)
	diff_teff = np.max(lh_teff)
    else:
        lh_teff = np.ones(len(um))
	diff_teff = 1.0

    if (input.logg > -99.0):
        lh_logg = gaussian(input.logg, mod['logg'][um], input.logge)
	diff_logg = np.max(lh_logg)
    else:
        lh_logg = np.ones(len(um))
	diff_logg = 1.0

    if (input.feh > -99.0):
        lh_feh = gaussian(input.feh, mod['feh_act'][um], input.fehe)
	diff_feh = np.max(lh_feh)
    else:
        lh_feh = np.ones(len(um))
	diff_feh = 1.0

    if (input.plx > -99.0):
        #lh_mabs = np.exp( (-1./(2.*input.plxe**2))*(input.plx-1./mod['dis'][um])**2)
	#lh_mag = gaussian(mabs,mod_mabs[um],mabse) # Added because this error was not previously included
	#disarr = np.linspace(min(mod['dis'][um]),max(mod['dis'][um]),10000)
	#fdis = interp1d(mod['dis'][um],lh_dis)
	#fmag = interp1d(mod['dis'][um],lh_mag)
	#conv_lh = np.convolve(fdis(disarr),fmag(disarr),'same')
	#flh = interp1d(disarr,conv_lh)
	#lh_mabs = flh(mod['dis'][um])/max(flh(mod['dis'][um]))
	#pdb.set_trace()
	#lh_mabs = lh_dis*lh_mag
	mabsIndex = np.argmax(np.exp( (-1./(2.*input.plxe**2))*(input.plx-1./mod['dis'][um])**2))
	downSelMagArr = mod_mabs[um]
	lh_mabs = gaussian(downSelMagArr[mabsIndex],mod_mabs[um],mabse) 
	diff_mabs = np.max(lh_mabs)

        #if (input.plxe/input.plx < 0.1):
        #    lh_mabs = np.exp( -(mabs-mod_mabs[um])**2. / (2.*mabse**2.))
        #else:
        #    dv=mod_mabs[um]-mabs
        #    lh_mabs = 10**(0.2*dv) * np.exp(-((10**(0.2*dv)-1.)**2)/(2.*(input.plxe/input.plx)**2))
    else:
        lh_mabs = np.ones(len(um))
	diff_mabs = 1.0

    if (input.dnu > 0.):
        mod_dnu = dnusun*mod['fdnu']*np.sqrt(10**mod['rho'])
        lh_dnu = np.exp( -(input.dnu-mod_dnu[um])**2.0 / (2.0*input.dnue**2.0))
    else:
        lh_dnu = np.ones(len(um))

    if (input.numax > 0.):
        mod_numax = (numaxsun
                     * (10**mod['logg']/gsun)
                     * (mod['teff']/teffsun)**(-0.5))

        lh_numax = gaussian(input.numax,mod_numax[um],input.numaxe)
    else:
        lh_numax = np.ones(len(um))

    tlh = (lh_gr*lh_ri*lh_iz*lh_jh*lh_hk*lh_bv*lh_bvt*lh_gk*lh_vtk*lh_rk*lh_rj*lh_teff*lh_logg*lh_feh
           *lh_mabs*lh_dnu*lh_numax)

    gof_metric = diff_gr*diff_ri*diff_iz*diff_jh*diff_hk*diff_bv*diff_bvt*diff_gk*diff_vtk*diff_rk*diff_rj*diff_mabs*diff_teff*diff_logg*diff_feh

        
    # metallicity prior (only if no FeH input is given)
    if (input.feh > -99.0):
        fprior = np.ones(len(um))
    else:
        fprior = fehprior(mod['feh_act'][um])

    # age prior (for eric target only):
    #if (input.id_starname == 'K2100'):
    #ageprior = gaussian(.79,mod['age'][um],0.03)
    #radprior = gaussian(1.215,mod['rad'][um],0.032)
    #massprior = gaussian(1.200,mod['mass'][um],0.035)
    #pdb.set_trace()
    #else:
	#ageprior = np.ones(len(um))
	#radprior = np.ones(len(um))
	#massprior = np.ones(len(um))
    
    # distance prior
    if (input.plx > -99.0):
        lscale = 1350.
        dprior = ((mod['dis'][um]**2/(2.0*lscale**3.))
                  *np.exp(-mod['dis'][um]/lscale))
    else:
        dprior = np.ones(len(um))

    # isochrone prior (weights)
    tprior = mod['dage'][um]*mod['dmass'][um]*mod['dfeh'][um]

    # evolutionary state prior (stars not in the red clump based on Hon et al. 2018 and Vrard et al. 2016)
    evprior = np.ones(len(um))
    evprior = np.where((mod['eep'][um] >= 631) & (mod['eep'][um] <= 707) & (input.evstate == 'RGB'),0.0,evprior)

    # posterior
    prob = fprior*dprior*tprior*tlh*evprior#*ageprior*radprior*massprior
    prob = prob/np.sum(prob)

    #pdb.set_trace()
    if (isinstance(dustmodel,pd.DataFrame) == False):
        names = ['teff', 'logg', 'feh_act', 'rad', 'mass', 'rho', 'lum', 'age']
        steps = [0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        fixes = [0, 1, 1, 0, 0, 1, 1, 0]
        
        if (redmap > -99.0):
            names = [
                'teff', 'logg', 'feh_act', 'rad', 'mass', 'rho', 'lum', 'age',
                'avs'
            ]
            steps = [0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            fixes=[0, 1, 1, 0, 0, 1, 1, 0, 1]

        if ((input.plx == -99.0) & (redmap > -99)):
            names=[
                'teff', 'logg', 'feh_act', 'rad', 'mass', 'rho', 'lum', 'age',
                'avs', 'dis'
            ]
            steps=[0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            fixes=[0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
            
        #if ((input.plx == -99.0) & (redmap > -99) & (useav > -99.0)):
        #    names=['teff','logg','feh','rad','mass','rho','lum','age','dis']
        #    steps=[0.001,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        #    fixes=[0,1,1,0,0,1,1,0,0]
            
    else:
        #names=['teff','logg','feh','rad','mass','rho','lum','age']
        #steps=[0.001,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        #fixes=[0,1,1,0,0,1,1,0,1]
        #if (input.plx == -99.0):

        avstep=((np.max(mod['avs'][um])-np.min(mod['avs'][um]))/10.)
	if avstep == 0:
	    avstep = 0.001
        #pdb.set_trace()

        names = [
            'teff', 'logg', 'feh_act', 'rad', 'mass', 'rho', 'lum', 'age', 'avs',
           'dis'
        ]
        steps=[0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, avstep, 0.01]
        fixes=[0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

    # Provision figure
    if plot:
        plotinit()

    ix = 1
    iy = 2
    npar = len(names)

    setattr(result, 'gof', gof_metric)
    print 'gof',gof_metric
   
    for j in range(0,npar):
        if fnmatch.fnmatch(names[j],'*lum*'):
            lum=np.log10((mod['rad'][um]**2. * (mod['teff'][um]/teffsun)**4.))
            x, y, res, err1, err2 = getpdf(
                lum, prob, name=names[j], step=steps[j], fixed=fixes[j],
                dustmodel=dustmodel)
        else:
            if (len(np.unique(mod[names[j]][um])) > 1):
                x, y, res, err1, err2 = getpdf(
                    mod[names[j]][um], prob, name=names[j], step=steps[j],
                    fixed=fixes[j],dustmodel=dustmodel
                )
	    elif ((len(np.unique(mod[names[j]][um])) == 1) and (names[j] == 'avs')):
		res = mod[names[j]][um[0]]
		err1 = 0.0
		err2 = 0.0
		x = res
        	y = 1.0
            else:
                res = 0.0
                err1 = 0.0
                err2 = 0.0

        print names[j], res, err1, err2
        setattr(result, names[j], res)
        setattr(result, names[j]+'ep', err1)
        setattr(result, names[j]+'em', err2)
        setattr(result, names[j]+'px', x)
        setattr(result, names[j]+'py', y)

        # Plot individual posteriors
        if plot:
            plotposterior(x, y, res, err1, err2, names, j, ix, iy)
            ix += 2
            iy += 2
	
	# Output individual posteriors
	#np.savetxt(os.path.join(self.outdir,names[j]+'_posterior.txt'),(x,y),delimiter=',')
    
    # Plot HR diagrams
    if plot:
	plothrd(model,mod,um,input,mabs,mabse,ix,iy)

    # Output posteriors:
    #output_replicated_posteriors(mod[um],prob,input.outdir +'/replicated_posteriors.csv')

    return result


# add extinction as a model parameter
def reddening(model,um,avs,extfactors):

    model2=dict((k, model[k][um]) for k in model.keys())
    nmodels=len(model2['teff'])*len(avs)
    #pdb.set_trace()

    keys = [
            'dage', 'dmass', 'dfeh', 'teff', 'logg', 'feh_act', 'rad', 'mass',
            'rho', 'age', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'jmag', 'hmag', 
            'bmag', 'vmag', 'btmag','vtmag', 'gamag', 'bpmag', 'rpmag',
	    'dis', 'kmag', 'avs', 'fdnu','eep'
    ]

    dtype = [(key, float) for key in keys]
    model3 = np.zeros(nmodels,dtype=dtype)

    start=0
    end=len(um)
    #pdb.set_trace()

    #print start,end
    for i in range(0,len(avs)):
        ix = np.arange(start,end,1)

        # NB: in reality, the model mags should also be Av-dependent;
        # hopefully a small effect!
        for c in 'b v u g r i z j h k bt vt ga bp rp'.split():
            cmag = c + 'mag'
            ac = 'a' + c
            av = extfactors['av']
            model3[cmag][ix] = model2[cmag] + avs[i]*extfactors[ac]/av

        keys = 'teff logg feh_act rad mass rho age dfeh dmass dage fdnu eep'.split()
        for key in keys:
            model3[key][ix]=model2[key]

        model3['avs'][ix] = avs[i]
        start = start + len(um)
        end = end + len(um)

    return model3

# redden model given a reddening map
def reddening_map(model, model_mabs, redmap, dustmodel, um, input, extfactors, 
                  band):

    if (len(band) == 4):
        bd = band[0:1]
    else:
        bd = band[0:2]
        
    equ = ephem.Equatorial(
        input.ra*np.pi/180.0, input.dec*np.pi/180.0, epoch=ephem.J2000
    )
    gal = ephem.Galactic(equ)
    lon_deg = gal.lon*180./np.pi
    lat_deg = gal.lat*180./np.pi

    # zero-reddening distance
    dis = 10**((redmap-model_mabs[um]+5)/5.)
    
    # This value contains fractional random errors in the reddening map for the Kepler field by comparisons between bayestar15 and bayestar17 maps:
    #redMapFracErr = 0.22092203269207672

    #pdb.set_trace()
    # iterate distance and map a few times
    for i in range(0,1):
        xp = np.concatenate(([0.0],np.array(dustmodel.columns[2:].str[3:],dtype='float')))
        fp = np.concatenate(([0.0],np.array(dustmodel.iloc[0][2:])))
        ebvs = np.interp(x=dis, xp=xp, fp = fp)
        #ebvs = ebvs + np.random.randn(len(ebvs))*np.median(ebvs)*redMapFracErr
        ext_band = extfactors['a'+bd]*ebvs	
        dis=10**((redmap-ext_band-model_mabs[um]+5)/5.)

    # if no models have been pre-selected (i.e. input is
    # photometry+parallax only), redden all models
    #pdb.set_trace()
    if (len(um) == len(model['teff'])):
        model3 = copy.deepcopy(model)

        for c in 'b v u g r i z j h k bt vt ga bp rp'.split():
            cmag = c + 'mag'
            ac = 'a' + c
            av = extfactors['av']
            model3[cmag] = model[cmag] + extfactors[ac] * ebvs

        model3['dis'] = dis
        model3['avs'] = extfactors['av']*ebvs
	#pdb.set_trace()
	 
    # if models have been pre-selected, extract and only redden those
    else:
        model2 = dict((k, model[k][um]) for k in model.keys())
        nmodels = len(model2['teff'])
        keys = [
            'dage', 'dmass', 'dfeh', 'teff', 'logg', 'feh_act', 'rad', 'mass',
            'rho', 'age', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'jmag', 'hmag', 
            'bmag', 'vmag', 'btmag','vtmag', 'gamag', 'bpmag', 'rpmag',
	    'dis', 'kmag', 'avs', 'fdnu','eep'
        ]
        
        dtype = [(key, float) for key in keys]
        model3 = np.zeros(nmodels,dtype=dtype)
        for c in 'b v u g r i z j h k bt vt ga bp rp'.split():
            cmag = c + 'mag'
            ac = 'a' + c
            av = extfactors['av']
            model3[cmag] = model2[cmag] + extfactors[ac] * ebvs

        model3['dis']=dis
        model3['avs']=extfactors['av']*ebvs
        keys = 'teff logg feh_act rad mass rho age dfeh dmass dage fdnu eep'.split()
        for key in keys:
            model3[key] = model2[key]

    return model3

def output_replicated_posteriors(grid,p,fileName):
    # grid = structure containing the model parameters, i.e. grid.teff, grid.rad, (in isoclassify: model['teff'], etc)
    # NB: grid should only include models in the x-sigma box, not the whole grid
    # p = posterior probabilities
    # outfile = file where output posteriors go
    x=grid
    y=p

    # normalize
    y=y/max(y)
    y=(y/max(y))*100.
    y=y.astype(np.int64)

    # number of samples desired
    numSamp=5e4
    
    # initialize parameter arrays that will contain the samples
    teffSamp=[]
    loggSamp=[]
    fehSamp=[]
    radSamp=[]
    massSamp=[]
    rhoSamp=[]
    lumSamp=[]
    ageSamp=[]
    disSamp=[]
    avsSamp=[]

    lum = np.log10((x['rad']**2. * (x['teff']/5772.0)**4.))

    n=len(y)
    seed = 12345
    random.seed(seed)
    # loop until we have enough samples
    while (len(teffSamp) < numSamp+1.):
        # draw random models	
        ran=random.randint(0,n-1)
	#pdb.set_trace()
        # skip if its posterior prob is too low
        if (y[ran] >= 1.):
	    # if not, replicate this model by the normalized y value and append it to the parameter arrays
	    teffSamp.extend([x['teff'][ran]]*y[ran])
	    loggSamp.extend([x['logg'][ran]]*y[ran])
	    fehSamp.extend([x['feh_act'][ran]]*y[ran])
	    radSamp.extend([x['rad'][ran]]*y[ran])
	    massSamp.extend([x['mass'][ran]]*y[ran])
	    rhoSamp.extend([x['rho'][ran]]*y[ran])
	    lumSamp.extend([lum[ran]]*y[ran])
	    ageSamp.extend([x['age'][ran]]*y[ran])
	    disSamp.extend([x['dis'][ran]]*y[ran])
	    avsSamp.extend([x['avs'][ran]]*y[ran])
    
    # Combine all samples into a dataframe:
    df = pd.DataFrame(list(zip(teffSamp, loggSamp, fehSamp, radSamp, massSamp, rhoSamp, lumSamp, ageSamp, disSamp, avsSamp)), 
               columns =['teff', 'logg', 'feh', 'rad', 'mass', 'rho', 'lum', 'age', 'dis', 'avs'])

    # Now output this df to a csv file:
    df.to_csv(fileName,index=False)

####### For extra magnitude/color error contribution:
def compute_extra_MK_error(abskmag):
    massPoly = np.array([-1.218087354981032275e-04,3.202749540513295540e-03,
-2.649332720970200630e-02,5.491458806424324990e-02,6.102330369026183476e-02,
6.122397810371335014e-01])

    massPolyDeriv = np.array([-6.090436774905161376e-04,1.281099816205318216e-02,
-7.947998162910602238e-02,1.098291761284864998e-01,6.102330369026183476e-02])

    kmagExtraErr = abs(0.021*np.polyval(massPoly,abskmag)/np.polyval(massPolyDeriv,abskmag))

    return kmagExtraErr

def compute_extra_gk_color_error(gk):
    teffPoly = np.array([5.838899127633915245e-06,-4.579640759410575821e-04,
1.591988911769273360e-02,-3.229622768514631148e-01,4.234782988549875782e+00,
-3.752421323678526477e+01,2.279521336429464498e+02,-9.419602441779162518e+02,
2.570487048729761227e+03,-4.396474893847861495e+03,4.553858427460818348e+03,
-4.123317864249115701e+03,9.028586421378711748e+03])

    teffPolyDeriv = np.array([7.006678953160697955e-05,-5.037604835351633566e-03,
1.591988911769273429e-01,-2.906660491663167978e+00,3.387826390839900625e+01,
-2.626694926574968463e+02,1.367712801857678642e+03,-4.709801220889581600e+03,
1.028194819491904491e+04,-1.318942468154358357e+04,9.107716854921636696e+03,
-4.123317864249115701e+03])

    gkExtraColorErr = abs(0.02*np.polyval(teffPoly,gk)/np.polyval(teffPolyDeriv,gk))

    return gkExtraColorErr

def compute_extra_rj_color_error(rj):
    teffPoly = np.array([-9.033797695411665738e-05,-5.002516634534125151e-03,3.458439034564866899e-01,-7.850074022447024014e+00,9.695827458575601554e+01,-7.445989701176866902e+02,3.738880592898368377e+03,-1.242934535446293012e+04,2.698349141689277167e+04,-3.692273700110502978e+04,3.070683312986312012e+04,-1.699674869726793258e+04,1.140246789013836315e+04])

    teffPolyDeriv = np.array([-1.084055723449399780e-03,-5.502768297987537666e-02,3.458439034564866787e+00,-7.065066620202321701e+01,7.756661966860481243e+02,-5.212192790823806718e+03,2.243328355739021208e+04,-6.214672677231465059e+04,1.079339656675710867e+05,-1.107682110033150821e+05,6.141366625972624024e+04,-1.699674869726793258e+04])

    rjExtraColorErr = abs(0.02*np.polyval(teffPoly,rj)/np.polyval(teffPolyDeriv,rj))

    return rjExtraColorErr

######################################### misc stuff

# calculate parallax for each model
def redden(redmap, mabs, gl, gb, dust):
    logd = (redmap-mabs+5.)/5.
    newd = logd

    for i in range(0,1):
        cur = 10**newd
        ebv = dust(gl,gb,cur/1000.)
        av = ebv*3.1
        aj = av*1.2348743
        newd = (redmap-mabs-aj+5.)/5.

    s_newd = np.sqrt( (0.2*0.01)**2 + (0.2*0.03)**2 + (0.2*0.02)**2 )
    plx=1./(10**newd)
    s_plx=10**(-newd)*np.log(10)*s_newd
    pdb.set_trace()

    return 1./(10**newd)

def readinput(input):
    input = ascii.read('input.txt')
    ra = input['col1'][0]
    dec = input['col2'][0]
    bmag = input['col1'][1]
    bmage = input['col2'][1]
    vmag = input['col1'][2]
    vmage = input['col2'][2]
    gmag = input['col1'][3]
    gmage = input['col2'][3]
    rmag = input['col1'][4]
    rmage = input['col2'][4]
    imag = input['col1'][5]
    image = input['col2'][5]
    zmag = input['col1'][6]
    zmage = input['col2'][6]
    jmag = input['col1'][7]
    jmage = input['col2'][7]
    hmag = input['col1'][8]
    hmage = input['col2'][8]
    kmag = input['col1'][9]
    kmage = input['col2'][9]
    plx = input['col1'][10]
    plxe = input['col2'][10]
    teff = input['col1'][11]
    teffe = input['col2'][11]
    logg = input['col1'][12]
    logge = input['col2'][12]
    feh = input['col1'][13]
    fehe = input['col2'][13]
    out = (
        ra, dec, bmag, bmage, vmag, vmage, gmag, gmage, rmag, rmage, 
        imag, image, zmag, zmage, jmag, jmage, hmag, hmage, kmag, kmage, 
        plx, plxe, teff, teffe, logg, logge, feh, fehe
    )

    return out

