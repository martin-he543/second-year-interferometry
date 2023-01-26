import sys
import read_data_results as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.stats as spst
import matplotlib.font_manager as fnt

titleFont =     {'fontname': 'C059', 'size': 13}
axesFont =      {'fontname': 'C059', 'size': 9}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'green', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'green'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)

#Step 1 get the data and the x position
file='%s'%(sys.argv[1]) #this is the data
results = rd.read_data3(file)
#print(results[0])
#carefull!!! change for the correct detector by swapping onew and zero here
y2 = np.array(results[0])
y1 = np.array(results[1])
#for now remove the mean, will need to remove the offset with a filter later
#y1 = y1 - y1.mean()
#y2 = y2 - y2.mean()
x=np.array(results[5])
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
# now find the difference between the crossings
diff=[]
for i in range(len(crossing_pos)-1):
    diff.append(np.abs(crossing_pos[i+1]-crossing_pos[i]))
diff=np.array(diff)

plt.figure("Crossing Points")
plt.plot(x, y1, 'x-', **lineStyle, **pointStyle)
plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko', **lineStyle, **pointStyle)
plt.xlabel("Position ($\mu$steps)", **axesFont)
plt.ylabel("Signal", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ticklabel_format(useMathText=True)
plt.title("Crossing Points", **titleFont)

plt.figure("Distribution of Distance Crossing Points")
plt.subplot(2,1,1)
plt.plot(crossing_pos[:-1],diff, **lineStyleBold, **pointStyle)
plt.xlabel("Position [$\mu$steps]", **axesFont)
plt.ylabel("Distance between Crossings ($\mu$steps)", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ticklabel_format(useMathText=True)
plt.title("Distance between Crossings", **titleFont)
#print(spst.sem(diff))
plt.subplot(2,1,2)
plt.hist(diff,bins=100, **histStyle)
plt.xlabel("Distance between Crossings ($\mu$steps)", **axesFont)
plt.ylabel("Number of Entries", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ticklabel_format(useMathText=True)

print("The mean difference between crossing points is",diff.mean(),"+/-",diff.std()/np.sqrt(len(diff)))
print("and the standard deviation between crossing points is ",diff.std())

plt.show()
