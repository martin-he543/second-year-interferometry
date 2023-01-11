#!/usr/bin/python3

import numpy as np
import numpy.random as npr
import scipy as sp
import pylab as pl
import scipy.fftpack as spf

def add_line(x,y,wl,amp,width,nstep):
    """
    This little function adds the effect
    of a a new line on to the interferogram.
    It does this by assuming that each line is made up of lots of descrete delta functions. Also assumes a gausian line shape
    and calculates to +/- 3 sigma
    x is the separation between the mirrors
    y is the amplitude of the light
    wl is the wavelength
    amp is the amplitude (arbitrary scale)
    width is the line width (actually 1 sigma as we assume gaussian)
    nsteps is the 
    """
    #nwidth=30.
    nsigma=5
    amplitude=amp*calc_amp(nsigma,nstep)
    wl_step=nsigma*2.0*width/nstep
    for i in range(len(amplitude)):
        wavelength=wl-nsigma*width+i*wl_step
        y=y+amplitude[i]*np.sin(np.pi*2.*x/wavelength)        
    return y

def calc_amp(nsigma,nsamp):
    """
    Just calculates the amplitude at the various steps
    """
    yy=np.empty(shape=[nsamp])
    step=nsigma*2.0/nsamp
    for i in range(nsamp):
        x=-nsigma+i*step
        size=np.exp(-x*x/4)
        yy[i]=size
    return yy
  




# Now set up the experiment that you want to do

# Na lines
l3=568.8e-9 # not sure about this one
l1=589e-9 # wavelength of spectral line in m
l2=589.6e-9 # wavelength of a second spectral line in m
w3=w1=w2=0.01e-9 # setting the lines to have the same width in m
#l1=800.e-9
#w1=200.e-9

'''
# When you perform the actual experiment you will move
# one mirror to change the path difference. This move will be 
# by a small, finite, amount. You will then take a reading with your detector. 
# Then you will move the mirror again and take another 
# reading and so on. Here you should set up the what these different 
# separations should be
'''

#changed to make it more like the actual experiment -DJC 09062018
nsamp=34000#number of readings that you will take (set in the software)
dist_per_step=20.e-9 #distance moved (new apparatus prob. 1nm step*2 for path length)
steps_per_sample=1
dsamp=dist_per_step*steps_per_sample

# set the starting point from null point
dstart= -30e-3 # start -3mm from null point

epoint=dstart+dsamp*nsamp



x= np.linspace(dstart,epoint,nsamp) #setting the x locations of the samples
# introduce a little jitter
# make it random for now
#sig_x=10e-9 # 10nm jitter
#for i in x:
#    i=npr.normal(scale=sig_x)+i


y=np.zeros(shape=[len(x)]) #setting the array that will contain your results



# Na spectrum (roughly)

#y=add_line(x,y,l2,1.0,w1,50)
#y=add_line(x,y,l1,0.5,w2,50)
#y=add_line(x,y,l3,0.05,w3,50)
#y=add_line(x,y,l1+0.1e-9,.25,w2,50)
#y=add_line(x,y,l1+0.2e-9,.25,w2,50)
#y=add_line(x,y,l1,1.0,w1,4000)


#white light 
y=add_line(x,y,570e-9,1.0,0.2e-9,400)

#print(2.*np.pi/l2) 

# plot the output

pl.figure(1)
pl.plot(x,y,'bo-')
pl.xlabel("Distance from null point (m)")
pl.ylabel("Amplitude")




# take a fourier transform
yf=spf.fft(y)
xf=spf.fftfreq(nsamp) # setting the correct x-axis for the fourier transform. Osciallations/step

#now some shifts to make plotting easier (google if ineterested)
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)


pl.figure(2)
pl.plot(xf,np.abs(yf))
pl.xlabel("Oscillations per sample")
pl.ylabel("Amplitude")


# Now try to reconstruct the original wavelength spectrum
# only take the positive part of the FT
# need to go from oscillations per step to steps per oscillation
# time the step size


xx=xf[int(len(xf)/2+1):len(xf)]
repx=dsamp/xx

pl.figure(3)
pl.plot(repx,abs(yf[int(len(xf)/2+1):len(xf)]))
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")


pl.show()
