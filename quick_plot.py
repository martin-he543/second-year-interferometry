###################################################################################################
### A little program that reads in the data and plots it
### You can use this as a basis for your analysis software 
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

automation = True
if automation:
    file_list = ['Task_11_actual_3.txt', 'Task_11_final_2.txt', 'Task_10_finito_2.txt', 'Task_10_2.txt', 'Task_9_orange_7500.txt', 'Task_6c_60000.txt', 'Task_10_prelim.txt', 'Task_6c_15000.txt', 'Task_10_finito_prelim.txt', 'Task_8_6.txt', 'Task_11_final_3.txt', 'Task_6a_prelim_A.txt', 'Task_9_white.txt', 'Task_6c_30000.txt', 'Task_10_actual_1.txt', 'Task_10_final_3.txt', 'Task_10_actual_2.txt', 'Task_9.txt', 'Task_8_final_2.txt', 'Task_10_final_2.txt', 'Task_11_1.txt', 'Task_10_finito_1.txt', 'Task_9_orange.txt', 'Task_6a_prelim_D.txt', 'Task_11_finito_1.txt', 'Task_11_fast_1.txt', 'Task_6a_3.txt', 'Task_10_trimmed_1.txt', 'Task_11_actual_4.txt', 'Task_6a_1.txt', 'Task_9_blue.txt', 'Task_8_3.txt', 'Task_11_finito_2.txt', 'Task_6a_prelim_B.txt', 'Task_8_4.txt', 'Task_11_2.txt', 'Task_10_final_1.txt', 'Task_11_actual_0.txt', 'Task_8_final_1.txt', 'Task_9_yellow.txt', 'Task_6a_2.txt', 'Task_10_1.txt', 'Task_11_actual_1.txt', 'Task_10_final_4.txt', 'Task_6c_7500.txt', 'Task_10_actual_4.txt', 'Task_8_5.txt', 'Task_8_1.txt', 'Task_8_2.txt', 'Task_9_orange_7500_add.txt', 'Task_10_actual_3.txt', 'Task_6a_prelim_C.txt', 'Task_10_fast_1.txt']

    for file in range(len(file_list)):
        file = "data/" + file_list[file]
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
        pl.savefig(file + '_Detector_1.png',dpi=500)
        print("Detector 1 Saved: ",file)

        pl.figure("Detector 2")
        pl.plot(x,y2,'o-')
        pl.xlabel("Position ($\mu$steps)", **axesFont)
        pl.ylabel("Signal 2", **axesFont)
        pl.ticklabel_format(useMathText=True)
        pl.xticks(**ticksFont)
        pl.yticks(**ticksFont)
        pl.savefig(file + '_Detector_2.png',dpi=500)
        print("Detector 2 Saved: ",file)

else:
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
    #pl.savefig(file + '_Detector_1.png',dpi=500)
    #print("Detector 1 Saved: ",file)

    pl.figure("Detector 2")
    pl.plot(x,y2,'o-')
    pl.xlabel("Position ($\mu$steps)", **axesFont)
    pl.ylabel("Signal 2", **axesFont)
    pl.ticklabel_format(useMathText=True)
    pl.xticks(**ticksFont)
    pl.yticks(**ticksFont)
    #pl.savefig(file + '_Detector_2.png',dpi=500)
    #print("Detector 2 Saved: ",file)
    pl.show()
