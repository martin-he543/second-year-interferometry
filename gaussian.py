#%%
import sys
import numpy as np
import pylab as pl
import read_data_results as rd
import matplotlib.font_manager as fnt
import scipy.optimize as opt

titleFont =     {'fontname': 'C059', 'size': 13}
axesFont =      {'fontname': 'C059', 'size': 9}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'green', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'green'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)
def gaussian(x, A, mu, sd, D):  return A * np.exp((-(x-mu)**2)/(2*(sd**2))) + D

np.set_printoptions(threshold=sys.maxsize)

file = '%s'%(sys.argv[1])
results = rd.read_data3(file)
x = np.array(results[5])
y1 = np.array(results[0])
y2 = np.array(results[1])

xr = x[2208:4930]; y1r = y1[2208:4930]

peak_A = np.amax(y1)
mean = np.mean(y1)
y1_new, x1_new = [], []
y1_nouveau, x1_nouveau = [], []
print(peak_A, mean)

for i in range(len(y1r)):
    if y1[i] > mean:
        y1_new.append(y1[i])
        x1_new.append(xr[i])
    a = ((mean - 110e6 < x1_new)[0])
    b = (x1_new[i-1] < mean + 110e6)
    myList = [a, b]
    if all(myList):
        y1_nouveau.append(y1_new[i])
        x1_nouveau.append(x1_new[i])

initial_guess2 = [40000, -1865000, 700000, -2600]
fit2, cov2 = opt.curve_fit(gaussian, x1_nouveau, y1_nouveau, initial_guess2, maxfev=1000000)

print("Mean Wavelength 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))
print("Standard Deviation 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))

pl.figure("Detector 1")
pl.title("Gaussian Fitting on Task 9", **titleFont)
pl.plot(x,y1,'o-', **lineStyle, **pointStyle)
pl.plot(x,gaussian(x, 0, 0, 0, fit2[3]), color='red')
pl.plot(x,gaussian(x, *fit2), color='blue')
pl.xlabel("Position ($\mu$steps)", **axesFont)
pl.ylabel("Signal 1", **axesFont)
pl.ticklabel_format(useMathText=True)
pl.xticks(**ticksFont)
pl.yticks(**ticksFont)
#pl.savefig(file + '_Detector_1.png',dpi=500)
#print("Detector 1 Saved: ",file)

print(fit2)

# pl.figure("Detector 2")
# pl.plot(x,y2,'o-', **lineStyle, **pointStyle)
# pl.xlabel("Position ($\mu$steps)", **axesFont)
# pl.ylabel("Signal 2", **axesFont)
# pl.ticklabel_format(useMathText=True)
# pl.xticks(**ticksFont)
# pl.yticks(**ticksFont)
#pl.savefig(file + '_Detector_2.png',dpi=500)
#print("Detector 2 Saved: ",file)
pl.show()