###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for your analysis software 
###################################################################################################

import sys
import numpy as np
import matplotlib as plt
import pylab as pl
import read_data_results as rd
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
cutoff = 3.8e-7

#Step 1 get the data and the x position
file = "data/Task_11_final_green.txt"
#file='%s'%(sys.argv[1]) #this is the data
results = rd.read_data3(file)
y1 = np.array(results[0])
y2 = np.array(results[1])
x = np.array(results[5])

for i in range(len(x)):
    if x[i] > cutoff:
        break
print(i)

x_left_mean = np.mean(y1[:16400])
x_right_mean = np.mean(y1[16400:])
y1[170000:] = y1[170000:] - 10000

pl.figure("Detector 1")
pl.plot(x,y1,'o-', **pointStyle)
pl.xlabel("Position ($\mu$steps)", **axesFont)
pl.ylabel("Signal 1", **axesFont)
pl.ticklabel_format(useMathText=True)
pl.xticks(**ticksFont)
pl.yticks(**ticksFont)
#pl.savefig(file + '_Detector_1.png',dpi=500)
#print("Detector 1 Saved: ",file)

# pl.figure("Detector 2")
# pl.plot(x,y2,'o-', **pointStyle)
# pl.xlabel("Position ($\mu$steps)", **axesFont)
# pl.ylabel("Signal 2", **axesFont)
# pl.ticklabel_format(useMathText=True)
# pl.xticks(**ticksFont)
# pl.yticks(**ticksFont)
#pl.savefig(file + '_Detector_2.png',dpi=500)
#print("Detector 2 Saved: ",file)
pl.show()
