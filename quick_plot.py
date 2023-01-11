#!/usr/bin/python

###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for you analysis software 
###################################################################################################

import sys
import numpy as np
import pylab as pl
import read_data_results as rd

#Step 1 get the data and the x position
file='%s'%(sys.argv[1]) #this is the data
results = rd.read_data3(file)

y1 = np.array(results[0])
y2 = np.array(results[1])

x=np.array(results[5])

pl.figure("Detector 1")
pl.plot(x,y1,'o-')
pl.xlabel("Position $\mu$steps]")
pl.ylabel("Signal 1")



pl.figure("Detector 2")
pl.plot(x,y2,'o-')
pl.xlabel("Position $\mu$steps]")
pl.ylabel("Signal 2")


pl.show()
