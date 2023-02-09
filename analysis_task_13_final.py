import sys
import read_data_results as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import matplotlib.font_manager as fnt

titleFont =     {'fontname': 'Bodoni 72', 'size': 14,'weight': 'bold'}
subtitleFont =  {'fontname': 'Bodoni 72', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'Bodoni 72', 'size': 10}
annotationFont ={'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1= {'fontname': 'SF Mono', 'size': 6, 'weight': 'bold'}
annotFontMini2= {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='Bodoni 72', weight='normal', style='italic', size=8)

#Step 1 get the data and the x position
file="data/Task_13_final_1.txt" #this is the data
results = rd.read_data3(file)

#Step 1.1 - set the reference wavelength. Whatever units you use here will be theunits of your final spectrum
lam_r = 638.216/2 # units of nm - factor 2 because there is a crossing every half wavelength
#print(results[0])
#carefull!!! change for the correct detector by swapping onew and zero here
y1 = np.array(np.abs(results[0]))[2000:40000]
y2 = np.array(results[1])[2000:40000]
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
distance = xs[1:]-xs[:-1]

#step 5 FFT to extract spectra
yf1=spf.fft(cs(xs))
xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]
repx1=2*distance.mean()/xx1  

spectral_lines_y = [1e10, 2e8]
spectral_lines_sub = [893.0847e-1,915.819e-1,942.630e-1,962.711e-1,969.142e-1,1039.6315e-1,1062.7802e-1,1649.9373e-1,1849.499e-1,1942.273e-1,1973.794e-1,1987.841e-1,2026.860e-1,2052.828e-1,2224.711e-1,2252.786e-1,2260.294e-1,2262.223e-1,2263.634e-1,2536.517e-1,2652.039e-1,2653.679e-1,2847.675e-1,2916.250e-1,2947.074e-1,2967.280e-1,3021.498e-1,3125.668e-1,3131.548e-1,3131.839e-1,3208.169e-1,3532.594e-1,3605.762e-1,3650.153e-1,3654.836e-1,3663.279e-1,3983.931e-1]
# spectral_lines = [4046.563e-1,4339.223e-1,4347.494e-1,4358.328e-1,5128.442e-1,5204.768e-1,5425.253e-1,5460.735e-1,5677.105e-1,5769.598e-1,5790.663e-1,5871.279e-1,5888.939e-1,6146.435e-1,6149.475e-1,7081.90e-1,7346.508e-1,7944.555e-1,9520.198e-1]
spectral_lines = [4046.563e-1,4339.223e-1,4347.494e-1,4358.328e-1,5460.735e-1,5769.598e-1,5790.663e-1,6146.435e-1,6149.475e-1,7081.90e-1,7346.508e-1,7944.555e-1,9520.198e-1]
spectral_lines_sup = [10139.76e-1,13570.21e-1,13673.51e-1,15295.82e-1,17072.79e-1,23253.07e-1]

actual_lines_sub = [893.0847e-1,915.819e-1,942.630e-1,962.711e-1,969.142e-1,1039.6315e-1,1062.7802e-1,1649.9373e-1,1849.499e-1,1942.273e-1,1973.794e-1,1987.841e-1,2026.860e-1,2052.828e-1,2224.711e-1,2252.786e-1,2260.294e-1,2262.223e-1,2263.634e-1,2536.517e-1,2652.039e-1,2653.679e-1,2847.675e-1,2916.250e-1,2947.074e-1,2967.280e-1,3021.498e-1,3125.668e-1,3131.548e-1,3131.839e-1,3208.169e-1,3532.594e-1,3605.762e-1,3650.153e-1,3654.836e-1,3663.279e-1,3983.931e-1]
# actual_lines = [4046.563e-1,4339.223e-1,4347.494e-1,4358.328e-1,5128.442e-1,5204.768e-1,5425.253e-1,5460.220e-1,5677.105e-1,5767.671e-1,5790.185e-1,5871.279e-1,5888.939e-1,6146.435e-1,6149.475e-1,7081.90e-1,7346.508e-1,7944.555e-1,9520.198e-1]
actual_lines = [4046.55267e-1,4339.223e-1,4347.494e-1,4357.1275e-1,5461.3551e-1,5769.765e-1,5791.081e-1,6146.435e-1,6149.475e-1,7081.90e-1,7346.508e-1,7944.555e-1,9520.198e-1]
actual_lines_sup = [10139.76e-1,13570.21e-1,13673.51e-1,15295.82e-1,17072.79e-1,23253.07e-1]

si_lines = [2.0268600000000002e-07,2.0528280000000001e-07,2.2247110000000003e-07,2.2527860000000002e-07,2.2602940000000001e-07,2.262223e-07,2.2636340000000002e-07,2.536517e-07,2.652039e-07,2.6536790000000006e-07,2.847675e-07,2.91625e-07,2.947074e-07,2.9672800000000003e-07,3.0214980000000006e-07,3.1256680000000004e-07,3.131548e-07,3.131839e-07,3.208169e-07,3.532594000000001e-07,3.605762e-07,3.6501530000000007e-07,3.6548360000000005e-07,3.6632790000000003e-07,3.9839310000000003e-07,4.046563e-07,4.3392230000000004e-07,4.347494e-07,4.3583280000000005e-07,5.128442e-07,5.204768e-07,5.425253000000001e-07,5.460220000000001e-07,5.677105000000001e-07,5.767671e-07,5.790185000000001e-07,5.871278999999999e-07,5.888939000000001e-07,6.146435e-07,6.149475000000001e-07,7.081900000000001e-07,7.346508000000001e-07,7.944555000000001e-07,9.520198000000001e-07]

plt.figure("Fully-Corrected Spectrum FFT", figsize=(10,5))
plt.suptitle('Fully-Corrected Spectrum FFT: Zero-Crossing Analysis', **titleFont)
plt.title("Hg Lamp with Red He-Ne Laser Spectrum", **subtitleFont)
plt.plot(abs(repx1)- 4.5845,2.41545*abs(yf1[int(len(xf1)/2+1):len(xf1)])/1e10,color='black',label='Post-Shifting and Cubic-Spline Interpolation\nfor N=%i, Full Hg'%(N))

sub = False; mid = True; sup = False

if sub:
    for i in range(len(spectral_lines_sub)):
        plt.axvline(x=spectral_lines_sub[i], color="red", linestyle='--', **lineStyle)
        plt.annotate('local maximum at λ ≅ %.2f nm'%(spectral_lines_sub[i]), xy=(spectral_lines_sub[i] - 0.1, spectral_lines_y[1]), xytext=(spectral_lines_sub[i] - 0.1, spectral_lines_y[1]) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1))
        delta_percentage = (spectral_lines_sub[i]-actual_lines_sub[i])/spectral_lines_sub[i]
        #plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines_sub[i] - 0.1, spectral_lines_y[1] - 1e8), xytext=(spectral_lines_sub[i] - 0.1, spectral_lines_y[1] - 1e8) , rotation=315, **annotationFont, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
if sup:
    for i in range(len(spectral_lines_sup)):
        plt.axvline(x=spectral_lines_sup[i], color="red", linestyle='--', **lineStyle)
        plt.annotate('local maximum at λ ≅ %.2f nm'%(spectral_lines_sup[i]), xy=(spectral_lines_sup[i] - 0.1, spectral_lines_y[1]), xytext=(spectral_lines_sup[i] - 0.1, spectral_lines_y[1]) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1))
        delta_percentage = (spectral_lines_sup[i]-actual_lines_sup[i])/spectral_lines_sup[i]
        #plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines_sup[i] - 0.1, spectral_lines_y[1]-1e8), xytext=(spectral_lines_sup[i] - 0.1, spectral_lines_y[1]-1e8) , rotation=315, **annotationFont, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))

if mid:
    # 404
    plt.axvline(x=spectral_lines[0], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[0]), xy=(spectral_lines[0], 0.3), xytext=(spectral_lines[0], 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    delta_percentage = abs(spectral_lines[0]-actual_lines[0])/spectral_lines[0]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[0], 0.1), xytext=(spectral_lines[0], 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 434 I
    plt.axvline(x=spectral_lines[1], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[1]), xy=(spectral_lines[1] - 3, 0.3), xytext=(spectral_lines[1] - 3, 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    # delta_percentage = abs(spectral_lines[1]-actual_lines[1])/spectral_lines[1]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[1] - 3, 0.1), xytext=(spectral_lines[1] - 3, 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 434 II
    plt.axvline(x=spectral_lines[2], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[2]), xy=(spectral_lines[2], 0.3), xytext=(spectral_lines[2], 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    # delta_percentage = abs(spectral_lines[2]-actual_lines[2])/spectral_lines[2]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[2], 0.1), xytext=(spectral_lines[2], 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 436
    plt.axvline(x=spectral_lines[3], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[3]), xy=(spectral_lines[3] + 3, 0.3), xytext=(spectral_lines[3] + 3, 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    delta_percentage = abs(spectral_lines[3]-actual_lines[3])/spectral_lines[3]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[3] + 3, 0.1), xytext=(spectral_lines[3] + 3, 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 546
    plt.axvline(x=spectral_lines[4], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[4]), xy=(spectral_lines[4], 0.3), xytext=(spectral_lines[4], 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    delta_percentage = abs(spectral_lines[4]-actual_lines[4])/spectral_lines[4]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[4], 0.1), xytext=(spectral_lines[4], 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 577
    plt.axvline(x=spectral_lines[5], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[5]), xy=(spectral_lines[5] - 2, 0.3), xytext=(spectral_lines[5] - 2, 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    delta_percentage = abs(spectral_lines[5]-actual_lines[5])/spectral_lines[5]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[5] - 0.1, 0.1), xytext=(spectral_lines[5] - 2, 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))
    # 579
    plt.axvline(x=spectral_lines[6], color="red", linestyle='--', **lineStyle)
    plt.annotate('local maximum at λ ≈ %.2f nm'%(spectral_lines[6]), xy=(spectral_lines[6], 0.3), xytext=(spectral_lines[6], 0.3) , rotation=90, bbox=dict(boxstyle="square, pad=0.5", fc="#f4dadf", ec="k", lw=0.5, alpha=0.1), **annotFontMini1)
    delta_percentage = abs(spectral_lines[6]-actual_lines[6])/spectral_lines[6]
    plt.annotate('Δλ ≅ %.4f%%'%(delta_percentage), xy=(spectral_lines[6], 0.1), xytext=(spectral_lines[6], 0.1) , rotation=90, **annotFontMini1, bbox=dict(boxstyle="square, pad=0.5", fc="#f9ffff", ec="k", lw=0.5, alpha=0.1))

#%%
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
distance = xs[1:]-xs[:-1]

#step 5 FFT to extract spectra
yf1=spf.fft(cs(xs))
xf1=spf.fftfreq(len(xs)) # setting the correct x-axis for the fourier transform. Osciallations/step  
xf1=spf.fftshift(xf1) #shifts to make it easier (google if interested)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]
repx1=2*distance.mean()/xx1  
def gaussian(x, A, mu, sd, D):  return A * np.exp((-(x-mu)**2)/(2*(sd**2))) + D

#print(x[41142:41285]); print(x[41285:41428])

#%%
new_y_data = abs(yf1[int(len(xf1)/2+1):len(xf1)])
new_x_data = abs(repx1)
# print(new_y_data, new_x_data)
# s_f = 11100

peak1_start, peak1_end = np.where(x > 576)[0][0], np.where(x > 578)[0][0]
x_peak1 = x[peak1_start:peak1_end]
y_peak1 = y1[peak1_start:peak1_end]
y_peakfinder_1 = new_y_data[peak1_start:peak1_end]
ig_peak1 = [7.104e8 - 1.5e7,581.831 - 4.5845,0.09,1.5e7]

peak2_start, peak2_end = np.where(x > 578)[0][0], np.where(x > 580)[0][0]
x_peak2 = x[peak2_start:peak2_end]
y_peak2 = y1[peak2_start:peak2_end]
y_peakfinder_2 = new_y_data[peak2_start:peak2_end]
ig_peak2 = [7.3891e8 - 1.5e7,583.9626 - 4.5845,0.09,1.5e7]

peak3_start, peak3_end = np.where(x > 540)[0][0], np.where(x > 550)[0][0]
x_peak3 = x[peak3_start:peak3_end]
y_peak3 = y1[peak3_start:peak3_end]
y_peakfinder_3 = new_y_data[peak3_start:peak3_end]
ig_peak3 = [4.1263e9 - 1.5e7,546.13551,0.05,1.5e7]

peak4_start, peak4_end = np.where(x > 430)[0][0], np.where(x > 440)[0][0]
x_peak4 = x[peak4_start:peak4_end]
y_peak4 = y1[peak4_start:peak4_end]
y_peakfinder_4 = new_y_data[peak4_start:peak4_end]
ig_peak4 = [2.91881e9-1.5e7,435.71275,0.05,1.5e7]

peak5_start, peak5_end = np.where(x > 430)[0][0], np.where(x > 440)[0][0]
x_peak5 = x[peak5_start:peak5_end]
y_peak5 = y1[peak5_start:peak5_end]
y_peakfinder_4 = new_y_data[peak5_start:peak5_end]
ig_peak5 = [5.09600e8-1.5e7,404.55267,0.05,1.5e7]

vals1 = np.linspace(560, 578, 5000)
vals2 = np.linspace(578, 600, 5000)
vals3 = np.linspace(460, 560, 10000)
vals4 = np.linspace(420, 460, 10000)
vals5 = np.linspace(400, 420, 10000)
plt.plot(new_x_data[8008:],2.41545*(new_y_data[8008:] - 0.4e8)/1e10,color='black')
plt.plot(vals2, 2.41545*gaussian(vals2, *ig_peak2)/1e10,'-',color='orangered',label='First Gaussian at 579nm')
plt.plot(vals1, 2.41545*gaussian(vals1, *ig_peak1)/1e10,'-',color='gold', label='Second Gaussian at 577nm')
plt.plot(vals3, 2.41545*gaussian(vals3, *ig_peak3)/1e10,'-',color='palegreen', label='Third Gaussian at 546nm')
plt.plot(vals4, 2.41545*gaussian(vals4, *ig_peak4)/1e10,'-',color='dodgerblue', label='Fourth Gaussian at 436nm')
plt.plot(vals5, 2.41545*gaussian(vals5, *ig_peak5)/1e10,'-',color='indigo', label='Fifth Gaussian at 404nm')

percentage_peak1 = 100*ig_peak1[0]/ig_peak3[0]
percentage_peak2 = 100*ig_peak2[0]/ig_peak3[0]
percentage_peak3 = 100*ig_peak3[0]/ig_peak3[0]
percentage_peak4 = 100*ig_peak4[0]/ig_peak3[0]
percentage_peak5 = 100*ig_peak5[0]/ig_peak3[0]

plt.axhline(2.41545*ig_peak1[0]/1e10, color='green', linestyle='--', **lineStyle)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak1), xy=(ig_peak1[1], 2.41545*ig_peak1[0]/1e10 - 0.025), xytext=(ig_peak1[1], 2.41545*ig_peak1[0]/1e10 - 0.025), **annotFontWeak)
plt.axhline(2.41545*ig_peak2[0]/1e10, color='green', linestyle='--', **lineStyle)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak2), xy=(ig_peak1[1], 2.41545*ig_peak2[0]/1e10 + 0.01), xytext=(ig_peak2[1], 2.41545*ig_peak2[0]/1e10 + 0.01), **annotFontWeak)
plt.axhline(2.41545*ig_peak3[0]/1e10, color='green', linestyle='--', **lineStyle)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak3), xy=(ig_peak1[1], 2.41545*ig_peak3[0]/1e10), xytext=(ig_peak3[1], 2.41545*ig_peak3[0]/1e10), **annotFontWeak)
plt.axhline(2.41545*ig_peak4[0]/1e10, color='green', linestyle='--', **lineStyle)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak4), xy=(ig_peak1[1], 2.41545*ig_peak4[0]/1e10), xytext=(ig_peak4[1], 2.41545*ig_peak4[0]/1e10), **annotFontWeak)
plt.axhline(2.41545*ig_peak5[0]/1e10, color='green', linestyle='--', **lineStyle)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak5), xy=(ig_peak1[1], 2.41545*ig_peak5[0]/1e10), xytext=(ig_peak5[1], 2.41545*ig_peak5[0]/1e10), **annotFontWeak)


# np.savetxt("output_task_13_final.txt", np.column_stack((new_x_data, new_y_data)))

plt.xlabel("Wavelength / nm",**axesFont)
plt.ylabel('Relative Intensity / a.u.', **axesFont)
# ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*(1/0.41527)))
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.plot(abs(repx1),np.linspace(-2.9e10,-3e10,4999999),"r--", label="Local Maxima")
plt.legend(prop=font, loc="upper center")
plt.ticklabel_format(useMathText=True)  
plt.xlim(400,600)
plt.ylim(0, 1)
plt.savefig("output_"+file+".png", dpi=1000)
plt.show()