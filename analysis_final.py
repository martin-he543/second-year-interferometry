#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:07:31 2023

@author: oliversharpe
"""

import sys
import read_data_results as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
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

"""

make sure the detectors are the right way round
make sure to put in the right wavelength
make sure line 89 is the right way round 

"""

#Step 1 get the data and the x position
file="Task12Spec.txt" #this is the data
results = rd.read_data3(file)

#Step 1.1 - set the reference wavelength. Whatever units you use here will be theunits of your final spectrum
lam_r = 546/2 # units of nm - factor 2 because there is a crossing every half wavelength
#print(results[0])
#carefull!!! change for the correct detector by swapping onew and zero here
y2 = np.array(results[0])
y1 = np.array(results[1])
#for now remove the mean, will need to remove the offset with a filter later
#y1 = y1 - y1.mean()
#y2 = y2 - y2.mean()


sampling_frequency=50 #frequency, in Hz
speed_test= 2*0.35
x = speed_test * np.arange(0, len(y1), 1)/sampling_frequency#position in mm, no need to be accurate here since we will be shifting the dataset anyways
dist=(speed_test)/(sampling_frequency) # distance between points to be used for uniform sampling


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
        xa = x[i]
        ya = y1[i]
        xb = x[i+1]
        yb = y1[i+1]
        b = (yb - ya/xa * xb)/(1-xb/xa)
        a = (ya - b)/xa
        extra = -b/a - xa
        crossing_pos.append(x[i]+extra)

# plt.figure("Find the crossing points")
# plt.title("Crossing Point Locations",**titleFont)
# plt.plot(x, y1, 'x-', **lineStyle)
# plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko')
# plt.xlabel("Position ($\mu$steps)",**axesFont)
# plt.ylabel("Arbitrary Units",**axesFont)
# plt.xticks(**ticksFont)
# plt.yticks(**ticksFont)
# plt.ticklabel_format(useMathText=True)  
#plt.show()

#step 3 shift the points
k = 0
x_corr_array = [0]
last_pt = 0
last_pt_corr = 0

for period in range(len(crossing_pos)//2-1):
    measured_lam = crossing_pos[2*period+2] - crossing_pos[2*period]
    shifting_ratio = lam_r/measured_lam
    while x[k]<crossing_pos[2*period+2]:
        x_corr = shifting_ratio*(x[k]-last_pt)+last_pt_corr
        x_corr_array.append(x_corr)
        k = k+1
    last_pt = x[k-1]
    last_pt_corr = x_corr_array[-1]
x_corr_array = x_corr_array[1:]

#step 4 create a uniform data set 
## if we want to keep with only the first branch use
y1 = y2
#Cubic Spline part
xr = x_corr_array
N = 10000000 # these are the number of points that you will resample - try changing this and look how well the resampling follows the data.
xs = np.linspace(0, x_corr_array[-1], N)
y = y2[:len(x_corr_array)]
cs = spi.CubicSpline(xr, y)

plt.figure("Correct the points and resample  the points")
plt.title('Correct and Resample Points\nZero-Crossing-Fitted Wavelength after Cubic-Spline \n%s'%file, **titleFont)
plt.xlabel("Position ($\mu$steps)",**axesFont)
plt.ylabel("Arbitrary Units",**axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.plot(xr, y, 'go', label = 'Inital points')
plt.plot(xs,cs(xs), label="Cubic-Spline, N=%i"%N)
plt.legend(prop=font)
plt.ticklabel_format(useMathText=True)  
plt.show()
distance = xs[1:]-xs[:-1]

#step 5 FFT to extract spectra
yf1=spf.fft(cs(xs))
xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]
repx1=2*distance.mean()/xx1  

plt.figure("Fully-Corrected spectrum FFT")
plt.title('Fully-Corrected Spectrum FFT: Zero-Crossing Analysis\n%s'%file, **titleFont)
plt.xlabel("Position ($\mu$steps)",**axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
#plt.plot(abs(repx0),abs(yf0[int(len(xf0)/2+1):len(xf0)]),label='Original')
#plt.plot(abs(repx),abs(yf[int(len(xf)/2+1):len(xf)]),label='After shifting and uniformising full mercury')
plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),label='After Shifting and Cubic-Spline\nN=%i, Full Hg'%(N))
plt.xlim(0,900)
plt.ylabel('Intensity (a.u.)', **axesFont)
plt.legend(prop=font)
plt.ticklabel_format(useMathText=True)  
plt.xlim(400,600)
plt.show()