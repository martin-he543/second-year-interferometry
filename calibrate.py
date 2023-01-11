#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import iminuit as im
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import scipy.stats as spst

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
plt.figure("Crossing points")

plt.plot(x, y1, 'x-')
plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko')
plt.xlabel("Position [$\mu$steps]")
plt.ylabel("Signal")

plt.figure("Distribution of distance crossing points")
plt.subplot(2,1,1)
plt.plot(crossing_pos[:-1],diff)
plt.xlabel("Position [$\mu$steps]")
plt.ylabel("Distance between crossings [$\mu$steps]")

print("The mean difference between crossing points is",diff.mean(),"+/-",diff.std()/np.sqrt(len(diff)))
print("and the standard deviation between crossing points is ",diff.std())
#print(spst.sem(diff))

plt.subplot(2,1,2)
plt.hist(diff,bins=100)
plt.xlabel("Distance between crossings [$\mu$steps]")
plt.ylabel("Number of entries")


plt.show()
