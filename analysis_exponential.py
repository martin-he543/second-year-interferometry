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
from scipy.optimize import curve_fit

np.set_printoptions(threshold=sys.maxsize)

titleFont =     {'fontname': 'C059', 'size': 11}
subtitleFont =  {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'C059', 'size': 9}
annotationFont ={'fontname': 'C059', 'size': 6, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
pointStyleRed = {'mew': 1, 'ms': 3, 'color': 'red'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

#Step 1 get the data and the x position
file="data/Task_12_green_singlet_2.txt" #this is the data
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
#print(len(x))  64422
def gaussian(x, A, mu, sd, D):  return A * np.exp((-(x-mu)**2)/(2*(sd**2))) + D
#print(x[np.argmax(x)] - x[np.argmin(x)])

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

peak1_start, peak1_end = np.where(x > 0)[0][0], np.where(x > 1.5)[0][0]
x_peak1 = x[peak1_start:peak1_end]
y_peak1 = y1[peak1_start:peak1_end]
peak2_start, peak2_end = np.where(x > 0)[0][0], np.where(x > 0.6)[0][0]
x_peak2 = x[peak2_start:peak2_end]
y_peak2 = y1[peak2_start:peak2_end]
peak3_start, peak3_end = np.where(x > 0.6)[0][0], np.where(x > 1.5)[0][0]
x_peak3 = x[peak3_start:peak3_end]
y_peak3 = y1[peak3_start:peak3_end]

def sinusoidal_wave(A, f, t, phase):
    return A * np.sin(2 * np.pi * f * t + phase)
def yukawa(r, A, lambda_, r_0):
    return A * np.exp(-lambda_ * (r - r_0)) / (r - r_0)
s_fit, s_cov = curve_fit(sinusoidal_wave, x_peak3, y_peak3, p0=[120, 40, -0.6], maxfev=10000000)
popt, pcov = curve_fit(yukawa, x_peak2, y_peak2, maxfev=10000000)
print(popt)

plt.figure("Find the Crossing Points")
plt.suptitle("Task 12: Percentage Error Fits",**titleFont)
plt.title("Reid Potential and Sinusoidal Fits",**subtitleFont)
#plt.plot(x_peak2, yukawa(x_peak2, *popt), **lineStyle, color='red', label="Reid Potential Fit")
plt.plot(x_peak3, sinusoidal_wave(x_peak3, *s_fit), **lineStyle, color='red', label="Reid Potential Fit")
plt.plot(x_peak2, y_peak2, 'x-', **pointStyleRed)
plt.plot(x_peak3, sinusoidal_wave(x_peak3, *s_fit), **lineStyle, color='blue', label="Sinusoidal Fit")
plt.plot(x_peak3, y_peak3, 'x', **pointStyle)
#plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko')
plt.xlabel("Position ($\mu$steps)",**axesFont)
plt.ylabel("Arbitrary Units",**axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(prop=font, loc="upper right")
plt.ticklabel_format(useMathText=True)  
plt.show()


# np.savetxt("task-12_crossings_exp.txt", np.column_stack((x_peak2, y_peak2)),delimiter="\t", fmt='%s')
# np.savetxt("task-12_crossings_sin.txt", np.column_stack((x_peak3, y_peak3)),delimiter="\t", fmt='%s')








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

# plt.figure("Correct and Resample Points")
# plt.title('Correct and Resample Points\nZero-Crossing-Fitted Wavelength After Cubic-Spline \n%s'%file, **titleFont)
# plt.xlabel("Position ($\mu$steps)",**axesFont)
# plt.ylabel("Arbitrary Units",**axesFont)
# plt.xticks(**ticksFont)
# plt.yticks(**ticksFont)
# plt.plot(xr, y, 'go', label = 'Inital points')
# plt.plot(xs,cs(xs), label="Cubic-Spline, N = 1 × 10⁷")
# plt.legend(prop=font)
# plt.ticklabel_format(useMathText=True)  
# plt.show()










# def gaussian(x, A, mu, sd, D):  return A * np.exp((-(x-mu)**2)/(2*(sd**2))) + D

# x,y = np.loadtxt("playground.csv", delimiter = ",", unpack=True)
# x = x[::-1]; y = y[::-1]

# peak1_start, peak1_end = np.where(x > 576)[0][0], np.where(x > 578)[0][0]
# x_peak1 = x[peak1_start:peak1_end]
# y_peak1 = y1[peak1_start:peak1_end]
# ig_peak1 = [1.687e9 - 5e7,576.79,0.1,5e7]
# popt, pcov = curve_fit(gaussian, x_peak1, y_peak1, p0=ig_peak1, maxfev=100000000)

# peak2_start, peak2_end = np.where(x > 578)[0][0], np.where(x > 580)[0][0]
# x_peak2 = x[peak2_start:peak2_end]
# y_peak2 = y1[peak2_start:peak2_end]
# ig_peak2 = [1.397025511e9 - 5e7,578.851,0.2,5e7]
# popt2, pcov2 = curve_fit(gaussian, x_peak2, y_peak2, p0=ig_peak2, maxfev=100000000)

# vals1 = np.linspace(576, 578, 1000)
# vals2 = np.linspace(578, 581, 1000)
# percentage = 1.397025511e9 / 1.687e9
# #bbox=dict(boxstyle="square, pad=0.2", fc="white", ec="r", lw=0.5)

# plt.plot(x,y, label='After Shifting and Cubic-Spline\nN = 1×10⁷, Entire Hg Spectrum')
# plt.suptitle('FFT with Double Gaussian Fitting: Zero-Crossing Analysis', y = 0.945, **titleFont)
# plt.title(file, **subtitleFont, pad=-3)
# plt.xlabel("Position / nm",**axesFont)
# plt.ylabel('Intensity / a.u.', **axesFont)
# plt.xticks(**ticksFont)
# plt.yticks(**ticksFont)
# plt.legend(prop=font)
# plt.ticklabel_format(useMathText=True)

# plt.plot(vals1, gaussian(vals1, *ig_peak1), 'r-', label='First Gaussian ≅ 577nm')
# plt.axvline(x=ig_peak1[1], color='red', linestyle='--', **lineStyle)
# plt.annotate('local maximum at ≅ 577nm', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.6e9), xytext=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.6e9) , rotation=90, **annotationFont)
# plt.axvline(x=ig_peak1[1] - 0.2, color='blue', linestyle='--', **lineStyle)
# plt.annotate('− 2σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.3e9), xytext=(ig_peak1[1] - 0.27, ig_peak1[0] - 1.67e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak1[1] + 0.2, color='blue', linestyle='--', **lineStyle)
# plt.annotate('+ 2σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.3e9), xytext=(ig_peak1[1] + 0.25, ig_peak1[0] - 1.67e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak1[1] - 0.3, color='purple', linestyle='--', **lineStyle)
# plt.annotate('− 3σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.3e9), xytext=(ig_peak1[1] - 0.38, ig_peak1[0] - 1.67e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak1[1] + 0.3, color='purple', linestyle='--', **lineStyle)
# plt.annotate('+ 3σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1.3e9), xytext=(ig_peak1[1] + 0.35, ig_peak1[0] - 1.67e9), rotation=90, **annotationFont)
# plt.axhline(ig_peak1[0] + 5e7, color='green', linestyle='--', **lineStyle)
# plt.axhline(ig_peak1[0] + 5e7, color='green', linestyle='--', **lineStyle)
# plt.annotate("\t\t\t\t\t%.2f%% of the Original Peak"%(100), xy=(ig_peak1[1] , ig_peak1[0]), xytext=(ig_peak1[1], ig_peak1[0]), **annotFontWeak)

# plt.plot(vals2, gaussian(vals2, *ig_peak2), 'r-', label='Second Gaussian ≅ 579nm')
# plt.annotate('local maximum at ≅ 579nm', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), xytext=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak2[1], color='red', linestyle='--', **lineStyle)
# plt.axvline(x=ig_peak2[1] - 0.4, color='blue', linestyle='--', **lineStyle)
# plt.annotate('− 2σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), xytext=(ig_peak2[1] - 0.5, ig_peak2[0] - 1.37e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak2[1] + 0.4, color='blue', linestyle='--', **lineStyle)
# plt.annotate('+ 2σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), xytext=(ig_peak2[1] + 0.45, ig_peak2[0] - 1.37e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak2[1] - 0.6, color='purple', linestyle='--', **lineStyle)
# plt.annotate('− 3σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), xytext=(ig_peak2[1] - 0.7, ig_peak2[0] - 1.37e9), rotation=90, **annotationFont)
# plt.axvline(x=ig_peak2[1] + 0.6, color='purple', linestyle='--', **lineStyle)
# plt.annotate('+ 3σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1.3e9), xytext=(ig_peak2[1] + 0.65, ig_peak2[0] - 1.37e9), rotation=90, **annotationFont)
# plt.axhline(ig_peak2[0] + 5e7, color='green', linestyle='--', **lineStyle)
# plt.annotate("\t\t\t\t\t%.2f%% of the Original Peak"%(percentage*100), xy=(ig_peak2[1] , ig_peak2[0]), xytext=(ig_peak2[1], ig_peak2[0]), **annotFontWeak)

# #plt.savefig("output_2.png", dpi=1000)
# plt.show()



































































# #step 5 FFT to extract spectra
# yf1=spf.fft(cs(xs))
# xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
# xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
# yf1=spf.fftshift(yf1)
# xx1=xf1[int(len(xf1)/2+1):len(xf1)]
# repx1=2*distance.mean()/xx1  

# nu_1, nu_2, nu_3 = np.where(x > 576)[0][0], np.where(x > 578)[0][0], np.where(x > 580)[0][0]
# #print(x[41142:41285]); print(x[41285:41428])
# new_y_data = abs(yf1[int(len(xf1)/2+1):len(xf1)])
# new_x_data = abs(repx1)
# # print(new_y_data, new_x_data)
# print(len(new_x_data), len(new_y_data))
# s_f = 11100
# newest_y_data = new_y_data[3193163:3204185]
# newest_x_data = new_x_data[3193163:3204185]
# a_new = new_x_data[41142:41284]
# b_new = new_x_data[41285:41428]

# plt.figure("Fully-Corrected spectrum FFT")
# plt.title('Fully-Corrected Spectrum FFT: Zero-Crossing Analysis\n%s'%file, **titleFont)
# plt.xlabel("Position ($\mu$steps)",**axesFont)
# plt.xticks(**ticksFont)
# plt.yticks(**ticksFont)
# #plt.plot(abs(repx0),abs(yf0[int(len(xf0)/2+1):len(xf0)]),label='Original')
# #plt.plot(abs(repx),abs(yf[int(len(xf)/2+1):len(xf)]),label='After shifting and uniformising full mercury')
# plt.plot(new_x_data,new_y_data, label='After Shifting and Cubic-Spline\nN=%i, Full Hg'%(N))
# plt.ylabel('Intensity (a.u.)', **axesFont)
# plt.legend(prop=font)
# plt.ticklabel_format(useMathText=True)  
# plt.xlim(575,585)
# plt.ylim(0,2e9)
# plt.show()

# np.savetxt("output.txt", np.column_stack((new_x_data, new_y_data)), delimiter="\t", fmt='%s')
# x,y = np.loadtxt("output.txt", delimiter = "\t", unpack=True)
# x_r, y_r = y[3193163:3204185], 
# print(x_r)
# print(y_r)
# plt.plot(x_r,y_r, label='After Shifting and Cubic-Spline\nN=%i, Full Hg'%(N))
# plt.show()
