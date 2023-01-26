#!/usr/bin/python

import numpy as np
import numpy.random as npr
import scipy as sp
import pylab as pl
import scipy.fftpack as spf
import matplotlib.font_manager as fnt

titleFont =     {'fontname': 'C059', 'size': 13}
axesFont =      {'fontname': 'C059', 'size': 9}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)

def add_line(x,y,wl,amp,width,nstep):
    """ add_line | Adds a Gaussian line to the distribution.
    This little function adds the effect of a a new line on to the interferogram.
    It does this by assuming that each line is made up of lots of discrete delta
    functions. Also assumes a Gaussian line shape and calculates to ± 5 sigma,
    where x is the separation between the mirrors, y is the amplitude of the 
    light, wl is the wavelength, amp is the amplitude (arbitrary scale), width 
    is the line width (actually 1 sigma as we assume gaussian), nsteps is the 
    number of discrete lines used.
    """ #nwidth=30.
    nsigma=5
    amplitude=(amp*calc_amp(nsigma,nstep))
    area=amp*width*np.sqrt(np.pi*2.)
    amplitude=amp*amplitude/sum(amplitude)
    wl_step=nsigma*2.0*width/nstep
    for i in range(len(amplitude)):
        wavelength=wl-nsigma*width+i*wl_step
        y=y+(amplitude[i]*np.sin(np.pi*2.*x/wavelength)+amplitude[i])        
    return y

def calc_amp(nsigma,nsamp):
    """ calc_amp | Just calculates the amplitude at the various steps.
    """
    yy=np.empty(shape=[nsamp])
    step=nsigma*2.0/nsamp
    for i in range(nsamp):
        x=-nsigma+i*step
        size=np.exp(-x*x/2)
        yy[i]=size
    return yy
  
def add_square(x,y,start,amp,width,nstep):
    """ add_square | Allows for simulation of white light with Gaussian.
    Start is the starting (lowest) wavelength of your tophat function, width is
    how wide your tophat is, and just as in add_line, nstep is the number of
    lines that will be used in the approximation.
    """
    step=width/(nstep-1)
    amplitude=amp/nstep
    for i in range(nstep):
        wavelength=start+i*step
        y=y+(amplitude*np.sin(np.pi*2.*x/wavelength)+amplitude)
    return y
    
# Now set up the experiment that you want to do

# Na lines
l1 = 589e-9             # wavelength of spectral line in m
l2 = 589.6e-9           # wavelength of a second spectral line in m
w2 = 0.1e-9             # setting the lines to have the same width in m
w1 = w2

''' When you perform the actual experiment you will move one mirror to change 
    the path difference.
    Change these to set up the experiment:
'''
nsamp=340000            # number of samples that you will take (set in software)
dsamp=40.e-9            # path difference between samples

# set the starting point from null point
dstart= -3e-3           # start -3mm from null point
epoint=dstart+dsamp*nsamp
# setting the x locations of the samples
x= np.linspace(dstart,epoint,nsamp)
# setting the array that will contain your results
y=np.zeros(shape=[len(x)])


# Na spectrum (roughly)
#y=add_line(x,y,l2,1.0,w1,50)
#y=add_line(x,y,l1,3.0,w2,50)
y=add_square(x,y,590e-9,1.0,10e-9,500)

# plot the output
pl.figure(1)
pl.plot(x,y,'bo-')
pl.xlabel("Distance from null point (m)")
pl.ylabel("Amplitude")

# quick check
#yp=np.sin(x*28248.5875706214*np.pi)/(x*28248.5875706214)+1
#pl.plot(x,yp,'ro-')

#draw a line 
#xl=[-0.00016,0.00016]
#yl=[0.,0.]
#pl.plot(xl,yl,'r-')

# take a fourier transform
yf=spf.fft(y)           # oscillations/step.
xf=spf.fftfreq(nsamp)   # setting the correct x-axis for the fourier transform. 

#now some shifts to make plotting easier (google if interested)
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)

pl.figure(2)
pl.plot(xf,np.abs(yf))
pl.xlabel("Oscillations per sample")
pl.ylabel("Amplitude")

# Now try to reconstruct the original wavelength spectrum. Only take the 
# positive part of the FT need to go from oscillations per step to steps per 
# oscillation. Time the step size.

xx=xf[int(len(xf)/2+1):len(xf)]
repx=dsamp/xx

pl.figure(3)
pl.plot(repx,abs(yf[int(len(xf)/2+1):len(xf)]))
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")
pl.show()
