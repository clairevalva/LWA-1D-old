import AtmosphericBlocking
import numpy as np
import h5py
import sys,os
import glob
import matplotlib.pyplot as plt
import logging
import Saving as saving
import random
import AtmosphericBlocking
import numpy as np
import h5py
import sys,os, fnmatch
import glob
import matplotlib.pyplot as plt
import logging
import Saving as saving
import random
import shutil

log_directory='logs/run_model'
logging.basicConfig(filename=log_directory+'.log',level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def gaussforce(x,t,peak=2,inject=True,tw=2.5,xw=2800.0e3,xc=16800.0e3,tc=277.8):
  # Gaussian centered at 277.8 days and 16,800 km
    tc = tc
    tw = tw
    t = t/86400.0
    xc = xc
    xw = xw
    sx = 1.852e-5 + np.zeros(len(x))
    if inject:
        sx *= (1+peak*np.exp(-((x-xc)/xw)**2 - ((t-tc)/tw)**2))
    return sx


def force_all(x,t,peak=2,inject=True,tw=2.5,xw=2800.0e3,xc=16800.0e3,tc=277.8):
  # Gaussian centered at 277.8 days
    tc = tc
    tw = tw
    t = t/86400.0
    xc = xc
    xw = xw
    sx = 1.852e-5 + np.zeros(len(x))
    if inject:
        sx *= (1+peak*np.exp(- ((t-tc)/tw)**2))
    return sx

def find_place(want, arr):
    return np.where(arr==want)[0][0]


def noboru_cx(x,Lx,alpha):
  # The background conditions used in Noboru's paper
    A0 = 10*(1-np.cos(4*np.pi*x/Lx))
    cx = 60 - 2*alpha*A0
    return cx,A0


def get_good_spectra(sized = 30, prescribed = True):
    save_sds, save_sums = np.load("sum_sds_avgs_01.npy", allow_pickle = True)
    
    wavenumbers = 1 / np.fft.fftfreq(240, d=1.5)
    wavenumbers[0] = 0
    frequencies = np.fft.fftfreq(355, d=0.25)
    
    newsize = sized + int(sized//3)
    
    # decide on acceptable wavenumbers
    ok_wavenum = np.unique(np.sort(np.abs(wavenumbers)))
    

    # choose the correct random ones and get indices
    random_wavenum = np.concatenate((np.random.choice(ok_wavenum, size = sized),
                                    np.random.choice(ok_wavenum, size = int(sized / 3))))
    random_sign = np.random.choice([0,0], size = newsize)
    touse_wn = np.multiply(random_wavenum, (-1) ** random_sign)
    
    places_found_wn = [find_place(want, np.abs(np.fft.fftshift(wavenumbers))) for want in touse_wn]
    
    # decide on acceptable frequencies
    ok_freq2 = np.unique(np.sort(frequencies))[178:178 + 11]
    ok_freq1 = np.unique(np.sort(frequencies))[178:178 + 60]
    ok_freq = np.concatenate((ok_freq2, ok_freq1))

    # choose the correct random ones and get indices
    random_f = np.concatenate((np.random.choice(ok_freq, size = sized),
                                    np.random.choice(ok_freq2, size = int(sized / 3))))
    random_sign = np.random.choice([0,0], size = newsize)
    touse_f = np.multiply(random_f, ((-1) ** random_sign))
    places_found_f = [find_place(want, frequencies) for want in touse_f]
    
    # now use these to grab the amplitudes
    amp_mean_use = save_sums[2][places_found_wn,places_found_f]
    amp_std_use = np.random.normal(scale = np.abs(save_sds[2][places_found_wn,places_found_f]))
    amp_use = np.divide(np.add(amp_mean_use, amp_std_use), .155)
    
    ffreqs = np.abs(touse_wn) # this should actually be wavenumbers # rand ints with max size 20, shape of nwforce
    fspeeds = touse_f*(1/(3600*24)) # this should actually be frequencies 
    fphases = np.random.rand(newsize)*2*np.pi # this should stay how it is
    fampls = np.divide(amp_use, 6) # this should be corresponding amplitudes # scaled to be a mean of about 1.7? hmm
    
    return ffreqs, fspeeds, fphases, np.abs(fampls)


def noiseforce(x,t,peak=2,inject=True,freqs=np.arange(10),speeds=np.arange(10),
               phases=np.zeros(10),ampls=np.ones(10),Lx=28000.0e3,tw=2.5,
               xw=2800.0e3,xc=16800.0e3,tc=277.8):
    if t/86400<270:
        return np.zeros(x.shape)+1.852e-5
    sx = np.zeros(x.shape)
    wampls = ampls*peak
    for i in range(0,len(freqs)):
        sx += 1.0/len(freqs)*wampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
        
    if t/86400 < 300 and t/86400 > 299:
        np.save("hm.npy", sx)
        
    sx = 1.852e-5*np.maximum(1,(1 + sx**3))
    return sx



def noisybackground(x,Lx,t=None,freqs=None,speeds=None,phases=None,ampls=None):
    dcx = np.zeros(len(x))
    for i in range(0,len(freqs)):
        dcx += 1.0/len(freqs)*ampls[i]*np.sin(2*np.pi*freqs[i]*x/Lx+speeds[i]*t+phases[i])
    return (1+dcx)




class conditions:
    def __init__(self,peak=2,inject=True,Y=10,beta=60,n=2,
                 alpha=0.55,tau=10.0,sfunc=None,xc=16800.0e3,
                 xw=2800.0e3,tw=2.5,tc=277.8,noisy=False):
        self.peak = peak
        self.inject=inject
        self.Y = Y
        self.sfunc=sfunc
        self.tw = tw
        self.tc = tc
        self.xc = xc
        self.xw = xw
        self.noisy=noisy
        if not sfunc:
            self.sfunc=gaussforce
        self.tau = tau*86400.0
        self.beta = beta
        self.n=n
        self.alpha = alpha
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,
                        tw=self.tw,xc=self.xc,
                        xw=self.xw,tc=self.tc)
        return sx
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        cx = self.beta - 2*self.alpha*A0
        return cx,A0
    

class noisyconditions:
    def __init__(self,peak=2,Y=10,beta=60,n=2,background=True,
                 forcing=True,nwforce=26,nwcx=21,maxforcex=20,
                 maxA0x=10,forcedecay=20,A0decay=40,alpha=0.55,
                 tc=277.8,tw=2.5,xc=16800.0e3,xw=2800.0e3,
                 sfunc=None,cfunc=None,inject=True,
                 cxpeak=0.5,tau=10.0,save_to_disk=False,overwrite=True, path='output/'):
        self.peak = peak
        self.cxpeak = cxpeak
        self.inject=inject
        self.Y  = Y
        self.sfunc=sfunc
        self.tw = tw
        self.tc = tc
        self.xc = xc
        self.xw = xw
        self.background=background
        self.forcingbool=forcing
        self.cfunc=cfunc
        self.tau = tau*86400.0
        self.nwforce = nwforce
        self.nwcx    = nwcx
        self.maxforcex = maxforcex
        self.maxA0x    = maxA0x
        self.forcedecay = forcedecay
        self.A0decay   = A0decay
        self.path      =path
        self.save_to_disk = save_to_disk
        self.overwrite = overwrite
                        
        if not sfunc and not forcing:
            print(forcing,sfunc)
            self.sfunc=gaussforce
        elif not sfunc and forcing:
            self.sfunc = noiseforce
        self.beta = beta
        self.n=n
        self.alpha = alpha
        if forcing:
            self.ffreqs, self.fspeeds, self.fphases, self.fampls = get_good_spectra(sized = 30)
             
            # self.ffreqs = np.random.randint(1,self.maxforcex,size=self.nwforce)
            # self.fspeeds = 2.0*np.pi/(self.forcedecay*86400.0) - 4*np.pi/(forcedecay*86400.0)*  np.random.rand(self.nwforce)  
            # self.fphases = np.random.rand(self.nwforce)*2*np.pi
            # self.fampls = 3.7*np.random.rand(self.nwforce) #6.8
        if background:
            self.cfreqs = np.random.randint(1,self.maxA0x,size=self.nwcx)
            self.cspeeds = 2.0*np.pi/(self.A0decay*86400.0) -  4*np.pi/(self.A0decay*86400.0)*np.random.rand(self.nwcx)  
            self.cphases = np.random.rand(self.nwcx)*2*np.pi
            self.campls = np.random.rand(self.nwcx)
       
    
      
        
        
    def forcing(self,x,t,peak=None,inject=None):
        if peak:
            self.peak = peak
        if inject:
            self.inject = inject
        if not self.forcingbool:
            sx = self.sfunc(x,t,peak=self.peak,inject=self.inject,
                            tw=self.tw,xc=self.xc,
                            xw=self.xw,tc=self.tc)
        else:
            sx = self.sfunc(x,t,peak=self.peak,freqs=self.ffreqs,
                            speeds=self.fspeeds,phases=self.fphases,
                            ampls=self.fampls)
        return sx
    
    def getcx(self,x,Lx,alpha=None,time=None):
        if alpha:
            self.alpha = alpha
        A0 = self.Y*(1-np.cos(2*self.n*np.pi*x/Lx))
        if self.background:
            A0 *= self.cfunc(x,Lx,t=time,freqs=self.cfreqs,
                             speeds=self.cspeeds,
                             phases=self.cphases,
                             ampls=self.cxpeak*self.campls)
        cx = self.beta - 2*self.alpha*A0
        
        return cx,A0
    


# #### Intitalizing the model #######




if __name__=="__main__":
    
    a2Lambda        = float(sys.argv[1]) # give 2*a*Y1, empirical value is 11
    gamma           = float(sys.argv[2]) # input gamma, strength of transient eddies
    Uj              = float(sys.argv[3]) # background jet speed
    no_of_years     = 3 #number of years (really no_of_years*360)
    RRR             = 200 # number of repeats
    
    Y1              = 10
    alpha1          = a2Lambda/(2*Y1)
    
    for xx in range(RRR):
        
        namelab = "a=" + str(a2Lambda).zfill(3) + "_g=" + str(gamma).zfill(3) + "_U=" + str(Uj).zfill(3) + "_num=" + str(xx).zfill(3)

        noisy_initc = noisyconditions(cfunc=noisybackground, cxpeak=0.5,Y=Y1, nwcx=21, n=2, peak=gamma, 
                                  nwforce=26, background=True,forcing=True,beta=Uj, alpha=alpha1, 
                                  path = '/data/clairev/noise_params/', save_to_disk=True)

        cond = noisy_initc

        model = AtmosphericBlocking.Model(nx=1024,Lx = 28000e3,dt=.005*86400/2,alpha=cond.alpha,
                                                tmax=3.5*86400,D=3.26e5,tau=cond.tau,
                                                sfunc=cond.forcing,cfunc=cond.getcx,
                                                forcingpeak=cond.peak,injection=cond.inject, beta=cond.beta,
                                                save_to_disk=True,
                                                overwrite=True,
                                                tsave_snapshots=50,
                                                verbose=False,
                                                path = '/data/clairev/' + namelab)

        
        
        model.verbose=True
        model.save_to_disk = True
        model.tsave_snapshots = 50
        model.tmin = 0*1*86400
        model.tmax = (360)*no_of_years*86400 
        model.beta = cond.beta
        model.run()
        
        pathpath = '/data/clairev/' + namelab
        named = pathpath + str("/snapshots/")
        namedsave = '/data/clairev/tail004/' + namelab                                                                                                 
        found = np.array(find("*.h5", named))
        all_openA = []
        all_openC = []
        all_openF = []
        all_openS = []
        all_openbeta = []
        i = 0
        
        keys = ['A', 'C', 'F', 'S', 'beta']
        
        for entry in found:
            # one can save the other keys by uncommenting the following lines
            
            f = h5py.File(entry, "r")
            # I think f is the forcing?
            openedA = np.array(f["A"])
            #openedC = np.array(f["C"])
            #openedF = np.array(f["F"])
            #openedS = np.array(f["S"])
            #openedbeta = np.array(f["beta"])

            
            
            all_openA.append(openedA)
            #all_openC.append(openedC)
            #all_openF.append(openedF)
            #all_openS.append(openedS)
            #all_openbeta.append(openedbeta)
            
            f.close()
                             
        np.save(namedsave + "_" + str(keys[0]) + ".npy", all_openA)    
    
        shutil.rmtree(pathpath, ignore_errors=False)


