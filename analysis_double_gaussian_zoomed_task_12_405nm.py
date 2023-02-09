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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)

titleFont =     {'fontname': 'C059', 'size': 11}
subtitleFont =  {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'C059', 'size': 9}
annotationFont ={'fontname': 'C059', 'size': 6, 'weight': 'bold'}
annotFontTiny = {'fontname': 'C059', 'size': 5, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
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

new_y_data = abs(yf1[int(len(xf1)/2+1):len(xf1)])
new_x_data = abs(repx1)
# print(new_y_data, new_x_data)
# s_f = 11100

peak1_start, peak1_end = np.where(x > 401)[0][0], np.where(x > 409)[0][0]
x_peak1 = x[peak1_start:peak1_end]
y_peak1 = y1[peak1_start:peak1_end]
y_peakfinder_1 = new_y_data[peak1_start:peak1_end]
ig_peak1 = [5.4977e8 - 2.5e7,404.553 + 0.15,0.2,2.5e7]

# peak2_start, peak2_end = np.where(x > 578)[0][0], np.where(x > 580)[0][0]
# x_peak2 = x[peak2_start:peak2_end]
# y_peak2 = y1[peak2_start:peak2_end]
# y_peakfinder_2 = new_y_data[peak2_start:peak2_end]
# ig_peak2 = [1.3967e9 - 5e7,578.851,0.25,5e7]

# print(new_x_data[np.argmax(new_x_data)])

vals1 = np.linspace(401, 409, 1000)
# vals2 = np.linspace(578, 580, 1000)
plt.figure("Fully-Corrected spectrum FFT")
plt.suptitle('FFT with Double Gaussian Fitting: Zero-Crossing Analysis', y = 0.945, **titleFont)
plt.title(file, **subtitleFont, pad=-3)
plt.xlabel("Wavelength (λ) / nm",**axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.plot(new_x_data + 0.15,new_y_data, label='After Shifting and Cubic-Spline\nN = 1 × 10⁷, Entire Hg Spectrum')
plt.plot(vals1, gaussian(vals1, *ig_peak1), 'r-', label='Gaussian ≅ 405nm')
# plt.plot(vals2, gaussian(vals2, *ig_peak2), 'r-', label='Second Gaussian ≅ 579nm')
plt.ylabel('Intensity / a.u.', **axesFont)
plt.legend(prop=font, loc="upper right")
plt.ticklabel_format(useMathText=True)  
plt.xlim(401,409)
plt.ylim(0,0.7e9)

percentage_peak1 = 100*ig_peak1[0]/ig_peak1[0]
# percentage_peak2 = 100*ig_peak2[0]/ig_peak1[0]
# print(percentage_peak1, percentage_peak2)

plt.axvline(x=ig_peak1[1], color='red', linestyle='--', **lineStyle)
plt.axvline(x=ig_peak1[1] - 0.2, color='blue', linestyle='--', **lineStyle)
plt.axvline(x=ig_peak1[1] + 0.2, color='blue', linestyle='--', **lineStyle)
plt.axvline(x=ig_peak1[1] - 0.3, color='purple', linestyle='--', **lineStyle)
plt.axvline(x=ig_peak1[1] + 0.3, color='purple', linestyle='--', **lineStyle)
plt.axhline(ig_peak1[0] + 2.5e7, color='green', linestyle='--', **lineStyle)

plt.annotate('local maximum at ≅ 405nm', xy=(ig_peak1[1] - 0.1, ig_peak1[0]-0.3e9), xytext=(ig_peak1[1] - 0.1, ig_peak1[0]-0.3e9) , rotation=90, **annotationFont)
plt.annotate('− 2σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0]- 1e9), xytext=(ig_peak1[1] - 0.27, ig_peak1[0]- 1e9), rotation=90, **annotFontTiny)
plt.annotate('+ 2σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0]- 1e9), xytext=(ig_peak1[1] + 0.25, ig_peak1[0]- 1e9), rotation=90, **annotFontTiny)
plt.annotate('− 3σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1e9), xytext=(ig_peak1[1] - 0.35, ig_peak1[0]- 1e9), rotation=90, **annotFontTiny)
plt.annotate('+ 3σ', xy=(ig_peak1[1] - 0.1, ig_peak1[0] - 1e9), xytext=(ig_peak1[1] + 0.35, ig_peak1[0] - 1e9), rotation=90, **annotFontTiny)
plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak1), xy=(ig_peak1[1], ig_peak1[0]), xytext=(ig_peak1[1], ig_peak1[0]), **annotFontWeak)


# plt.axvline(x=ig_peak2[1], color='red', linestyle='--', **lineStyle)
# plt.axvline(x=ig_peak2[1] - 0.4, color='blue', linestyle='--', **lineStyle)
# plt.axvline(x=ig_peak2[1] + 0.4, color='blue', linestyle='--', **lineStyle)
# plt.axvline(x=ig_peak2[1] - 0.6, color='purple', linestyle='--', **lineStyle)
# plt.axvline(x=ig_peak2[1] + 0.6, color='purple', linestyle='--', **lineStyle)
# plt.axhline(ig_peak2[0] + 5e7, color='green', linestyle='--', **lineStyle)

# plt.annotate('local maximum at ≅ 579nm', xy=(ig_peak2[1] - 0.1, ig_peak2[0]-0.3e9), xytext=(ig_peak2[1] - 0.1, ig_peak2[0] -0.3e9) , rotation=90, **annotationFont)
# plt.annotate('− 2σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0]- 1e9), xytext=(ig_peak2[1] - 0.54, ig_peak2[0]- 1e9), rotation=90, **annotFontTiny)
# plt.annotate('+ 2σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0]- 1e9), xytext=(ig_peak2[1] + 0.50, ig_peak2[0]- 1e9), rotation=90, **annotFontTiny)
# plt.annotate('− 3σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1e9), xytext=(ig_peak2[1] - 0.70, ig_peak2[0]- 1e9), rotation=90, **annotFontTiny)
# plt.annotate('+ 3σ', xy=(ig_peak2[1] - 0.1, ig_peak2[0] - 1e9), xytext=(ig_peak2[1] + 0.70, ig_peak2[0] - 1e9), rotation=90, **annotFontTiny)
# plt.annotate("\t\t\t\t\t%.2f%% of Max Peak"%(percentage_peak2), xy=(ig_peak2[1], ig_peak2[0]), xytext=(ig_peak2[1], ig_peak2[0]), **annotFontWeak)


plt.savefig("output_" + file + "_zoomed_405nm.png", dpi=1000)
plt.show()
