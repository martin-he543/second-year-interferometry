#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import pylab as pl
import read_data_results as rd
import matplotlib.font_manager as fnt

titleFont =     {'fontname': 'C059', 'size': 13}
axesFont =      {'fontname': 'C059', 'size': 9}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blu`e', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)

#Step 1 get the data and the x position
file='%s'%(sys.argv[1]) #this is the data
results = rd.read_data3(file)
y1 = np.array(results[0])
y2 = np.array(results[1])
x=np.array(results[5])

pl.figure("Detector 1")
pl.plot(x,y1,'o-')
pl.xlabel("Position ($\mu$steps)", **axesFont)
pl.ylabel("Signal 1", **axesFont)
pl.ticklabel_format(useMathText=True)
pl.xticks(**ticksFont)
pl.yticks(**ticksFont)

pl.figure("Detector 2")
pl.plot(x,y2,'o-')
pl.xlabel("Position ($\mu$steps)", **axesFont)
pl.ylabel("Signal 2", **axesFont)
pl.ticklabel_format(useMathText=True)
pl.xticks(**ticksFont)
pl.yticks(**ticksFont)

pl.show()
