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
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'green', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'green'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)


file_list = ['Task_6a_1.txt', 'Task_6a_2.txt', 'Task_6a_3.txt', 'Task_6a_prelim_A.txt',
 'Task_6a_prelim_B.txt', 'Task_6a_prelim_C.txt', 'Task_6a_prelim_D.txt',
 'Task_6c_15000.txt', 'Task_6c_30000.txt', 'Task_6c_60000.txt',
 'Task_6c_7500.txt', 'Task_8_1.txt', 'Task_8_2.txt', 'Task_8_3.txt',
 'Task_8_4.txt', 'Task_8_5.txt', 'Task_8_6.txt', 'Task_8_final_1.txt',
 'Task_8_final_2.txt', 'Task_9.txt', 'Task_9_blue.txt', 'Task_9_orange.txt',
 'Task_9_orange_7500.txt', 'Task_9_orange_7500_add.txt', 'Task_9_white.txt',
 'Task_9_yellow.txt', 'Task_10_1.txt', 'Task_10_2.txt', 'Task_10_actual_1.txt',
 'Task_10_actual_2.txt', 'Task_10_actual_3.txt', 'Task_10_actual_4.txt',
 'Task_10_fast_1.txt', 'Task_10_final_1.txt', 'Task_10_final_2.txt',
 'Task_10_final_3.txt', 'Task_10_final_4.txt', 'Task_10_finito_1.txt',
 'Task_10_finito_2.txt', 'Task_10_finito_prelim.txt', 'Task_10_prelim.txt',
 'Task_10_trimmed_1.txt', 'Task_11_1.txt', 'Task_11_2.txt',
 'Task_11_actual_0.txt', 'Task_11_actual_1.txt', 'Task_11_actual_3.txt',
 'Task_11_actual_4.txt', 'Task_11_fast_1.txt', 'Task_11_final_2.txt',
 'Task_11_final_3.txt', 'Task_11_finito_1.txt', 'Task_11_finito_2.txt']

t, u = 50, 1
file = "data/" + file_list[10]
results = rd.read_data3(file)
y1 = np.array(results[0])[t:-u]
x=np.array(results[5])[t:-u]


pl.figure("Detector 1")
pl.plot(x,y1,'o-', **pointStyle, **lineStyle)
pl.xlabel("Position ($\mu$steps)", **axesFont)
pl.ylabel("Signal 1", **axesFont)
pl.ticklabel_format(useMathText=True)
pl.xticks(**ticksFont)
pl.yticks(**ticksFont)
pl.savefig(file + '_Detector_1_trimmed.png',dpi=500)
print("Detector 1 Saved: ",file)

pl.show()

a = np.linspace(0,t,t+1)
c = sum(1 for line in open(file))
b = np.linspace(c - 1 - u, c, u)

lines = []
# read file
with open(file, 'r') as fp:
    # read an store all lines into list
    lines = fp.readlines()

# Write file
with open(file, 'w') as fp:
    # iterate each line
    for number, line in enumerate(lines):
        # delete line 5 and 8. or pass any Nth line you want to remove
        # note list index starts from 0
        if number not in a:
            fp.write(line)

