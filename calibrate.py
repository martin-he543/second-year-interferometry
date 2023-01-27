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

automation = False
if automation:
    file_list = ['Task_11_actual_3.txt', 'Task_11_final_2.txt', 'Task_10_finito_2.txt', 'Task_10_2.txt', 'Task_9_orange_7500.txt', 'Task_6c_60000.txt', 'Task_10_prelim.txt', 'Task_6c_15000.txt', 'Task_10_finito_prelim.txt', 'Task_8_6.txt', 'Task_11_final_3.txt', 'Task_6a_prelim_A.txt', 'Task_9_white.txt', 'Task_6c_30000.txt', 'Task_10_actual_1.txt', 'Task_10_final_3.txt', 'Task_10_actual_2.txt', 'Task_9.txt', 'Task_8_final_2.txt', 'Task_10_final_2.txt', 'Task_11_1.txt', 'Task_10_finito_1.txt', 'Task_9_orange.txt', 'Task_6a_prelim_D.txt', 'Task_11_finito_1.txt', 'Task_11_fast_1.txt', 'Task_6a_3.txt', 'Task_10_trimmed_1.txt', 'Task_11_actual_4.txt', 'Task_6a_1.txt', 'Task_9_blue.txt', 'Task_8_3.txt', 'Task_11_finito_2.txt', 'Task_6a_prelim_B.txt', 'Task_8_4.txt', 'Task_11_2.txt', 'Task_10_final_1.txt', 'Task_11_actual_0.txt', 'Task_8_final_1.txt', 'Task_9_yellow.txt', 'Task_6a_2.txt', 'Task_10_1.txt', 'Task_11_actual_1.txt', 'Task_10_final_4.txt', 'Task_6c_7500.txt', 'Task_10_actual_4.txt', 'Task_8_5.txt', 'Task_8_1.txt', 'Task_8_2.txt', 'Task_9_orange_7500_add.txt', 'Task_10_actual_3.txt', 'Task_6a_prelim_C.txt', 'Task_10_fast_1.txt']
    
    for file in range(len(file_list)):
        try:
            #Step 1 get the data and the x position
            file = '%s'%(sys.argv[1])
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

            print("The mean difference between crossing points is",diff.mean(),"+/-",diff.std()/np.sqrt(len(diff)))
            print("and the standard deviation between crossing points is ",diff.std())
            value_mean = diff.mean()
            std_mean = diff.std()/np.sqrt(len(diff))
            std_crossing = diff.std()
            
            plt.figure("Crossing Points")
            plt.plot(x, y1, 'x-', **lineStyle, **pointStyle)
            plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko', **lineStyle, **pointStyle)
            plt.xlabel("Position ($\mu$steps)", **axesFont)
            plt.ylabel("Signal", **axesFont)
            plt.xticks(**ticksFont)
            plt.yticks(**ticksFont)
            plt.ticklabel_format(useMathText=True)
            plt.title("Crossing Points", **titleFont)
            plt.savefig(file + "_crossing_points_[mean_"+ str(value_mean) + "±" + str(std_mean) + "_std_c_" + str(std_crossing) + "].png", dpi=500)
            plt.cla()
            plt.clf()

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
            plt.savefig(file + "_distribution_[mean_"+ str(value_mean) + "±" + str(std_mean) + "_std_c_" + str(std_crossing) + "].png", dpi=500)
            plt.cla()
            plt.clf()
        except Exception:
            pass
else:
    try:
        file = '%s'%(sys.argv[1])
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

        print("The mean difference between crossing points is",diff.mean(),"+/-",diff.std()/np.sqrt(len(diff)))
        print("and the standard deviation between crossing points is ",diff.std())
        value_mean = diff.mean()
        std_mean = diff.std()/np.sqrt(len(diff))
        std_crossing = diff.std()
        
        plt.figure("Crossing Points")
        plt.plot(x, y1, 'x-', **lineStyle, **pointStyle)
        plt.plot(crossing_pos, 0*np.array(crossing_pos), 'ko', **lineStyle, **pointStyle)
        plt.xlabel("Position ($\mu$steps)", **axesFont)
        plt.ylabel("Signal", **axesFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)
        plt.ticklabel_format(useMathText=True)
        plt.title("Crossing Points", **titleFont)
        plt.savefig(file + "_crossing_points_[mean_"+ str(value_mean) + "±" + str(std_mean) + "_std_c_" + str(std_crossing) + "].png", dpi=500)
        plt.cla()
        plt.clf()

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
        plt.savefig(file + "_distribution_[mean_"+ str(value_mean) + "±" + str(std_mean) + "_std_c_" + str(std_crossing) + "].png", dpi=500)
        plt.cla()
        plt.clf()
    except Exception:
        pass
