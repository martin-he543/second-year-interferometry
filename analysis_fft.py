#!/usr/bin/python

#imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.stats as spst
from scipy.signal import argrelextrema



#functions
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

def add_square(x,y,start,amp,width,nstep):
    step=width/(nstep-1)
    amplitude=amp/nstep
    for i in range(nstep):
        wavelength=start+i*step
        y=y+(amplitude*np.sin(np.pi*2.*x/wavelength)+amplitude)
    return y


def fit_func(x,a,mu,sig):
    gaus = a*np.exp(-(x-mu)**2/(2*sig**2))
    return gaus

#Step 1 get the data and the x position
data = np.loadtxt('Output_data.txt', delimiter=' ')
x = data[6000:12000,5]
x2 =  x*(532e-9/(2*7281.144158000063))-0.000205
y1 = data[6000:12000,1]
y2 = data[6000:12000,0]

#for now remove the mean, will need to remove the offset with a filter later
y1 = y1 - y1.mean()

#step 2.1 butterworth filter to correct for misaligment (offset)
filter_order = 2
freq = 1 #cutoff frequency
sampling = 50 # sampling frequency
sos = signal.butter(filter_order, freq, 'hp', fs=sampling, output='sos')
filtered = signal.sosfilt(sos, y1)
y1 = filtered
filtered = signal.sosfilt(sos, y2)
y2 = filtered


#step 2 get the x at which we cross
crossing_pos = []
for i in range(len(y1)-1):
    if (y1[i] <= 0 and y1[i+1] >= 0) or (y1[i] >= 0 and y1[i+1] <= 0) :
    #create the exact crossing point of 0
        xa = x2[i]
        ya = y1[i]
        xb = x2[i+1]
        yb = y1[i+1]
        b = (yb - ya/xa * xb)/(1-xb/xa)
        a = (ya - b)/xa
        extra = -b/a - xa
        crossing_pos.append(x2[i]+extra)

# now find the difference between the crossings
diff=[]
for i in range(len(crossing_pos)-1):
    diff.append(np.abs(crossing_pos[i+1]-crossing_pos[i]))


diff=np.array(diff)

# plt.figure("Crossing points")

# plt.plot(x, y1, 'x-')
# plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko')
# plt.xlabel("Position [$\mu$steps]")
# plt.ylabel("Signal")
# plt.show()


#null point
nsamp = len(x)
dstart = -1.5e-4 # start -3mm from null point
envelope = np.abs(y1)
initial_guess1 = [8e4, 0, 0.001]
fit, cov = sp.optimize.curve_fit(fit_func, x2, envelope, initial_guess1, maxfev=100000)
y_fit = 1.63*fit_func(x2, fit[0], fit[1], fit[2])

print("Fit Parameters: A - %.2e, sigma - %.2e"%(1.63*fit[0], np.abs(fit[2])))
print("Full Width Half Maximum:", str(2.355*np.abs(fit[2])), "+/-", str(2.355*np.sqrt(cov[2][2])))

plt.plot(x2, y1,color='orange', label="Experimental Data")
plt.plot(x2, y_fit, color='black', label='Gaussian Fit')
plt.xlabel("Distance from null point (m)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

plt.figure("Distribution of distance crossing points")
plt.subplot(2,1,1)
plt.plot(crossing_pos[:-1],diff, color='orange')
plt.xlabel("Position")
plt.ylabel("Distance between crossings [$\mu$steps]")

print("The mean difference between crossing points is",diff.mean(),"+/-",diff.std()/np.sqrt(len(diff)))
print("and the standard deviation between crossing points is ",diff.std())
print("Therefore wavelength is",2*diff.mean(),"+/-",2*diff.std()/np.sqrt(len(diff)))

#print(spst.sem(diff))

plt.subplot(2,1,2)
plt.hist(diff,bins=100, color='orange')
plt.xlabel("Distance between crossings")
plt.ylabel("Number of entries")
plt.show()


#FFT

mperstep =  532e-9/(2*7184)
x = x * mperstep
y = y1

mpersample = np.mean(np.diff(x))

yf = spf.fft(y)
xf = spf.fftfreq(nsamp)

xf = xf[1:]

xxf = xf[:int(nsamp/2)]
yyf = yf[:int(nsamp/2)]
xxf = np.abs(mpersample / xxf)
yyf = np.abs(yyf)

initial_guess2 = [1.2e6,4.5e-7,0.1e-7,]
fit2, cov2 = sp.optimize.curve_fit(fit_func, xxf, yyf, initial_guess2)

print("Wavelength:", str(fit2[1]),"+/-", str(np.sqrt(cov2[1][1])*100))

plt.plot(xxf,yyf, color='orange', label='Experimental Data')
plt.plot(xxf,fit_func(xxf,fit2[0],fit2[1],fit2[2]),color='black',label='Gaussian Fit')
plt.xlabel("Wavelength (m)")
plt.ylabel("Amplitude")
plt.xlim(300e-9,800e-9)
plt.legend()
plt.show()