import sys
import read_data_results as rd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate
np.set_printoptions(threshold=sys.maxsize)

titleFont =     {'fontname': 'C059', 'size': 13}
subtitleFont =  {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'C059', 'size': 9}
annotationFont ={'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1= {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2= {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

x, y = np.loadtxt("hg_lines.csv",delimiter=",",unpack=True)
x = x/10

error = np.abs((y-x)/x)*100
# print(error)
     
def linear_func(x, a, b):   return a * x + b
params, covar = curve_fit(linear_func, x, y)
slope,intercept = params[0], params[1]
x_fit = np.linspace(min(x), max(x), 100)
y_fit = linear_func(x_fit, slope, intercept)

# plt.figure("$λ_{theoretical}$ against $λ_{observed}$")
# plt.suptitle('Task 12: Difference between Measured and Theoretical Values', **titleFont)
# plt.title('A Plot Displaying $λ_{theoretical}$ against $λ_{observed}$', **subtitleFont)
# plt.plot(x, y,'x',**pointStyle, label="$λ_{theoretical}$ against $λ_{observed}$")
# plt.plot(x_fit, y_fit,'-',**lineStyle, label="Line of Best Fit (Linear)")
# plt.xlabel("Theoretical Values / nm",**axesFont)
# plt.ylabel("Observed Values / nm",**axesFont)
# plt.xticks(**ticksFont)
# plt.yticks(**ticksFont)
# plt.legend(prop=font, loc="lower right")
# #plt.savefig("hg_lines.png",dpi=1000)
# plt.show()

def gaussian(x, A, m, s, k):
    return k + (A / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - m) / s)**2 / 2)
#ig_peak1, cov = curve_fit(gaussian, x, y, p0=[])
x_fit = np.linspace(0,1000,1000)
x2_fit = np.linspace(1000,2400,1000)
ig_peak1 = [890, 708.19, 45.5, 0]
ig_peak2 = [100, 1500, 100, 0]

plt.figure("$λ_{theoretical}$ against $λ_{observed}$")
plt.suptitle('Task 12: Percentage Error', **titleFont)
plt.title('Percentage Error between $λ_{theoretical}$ and $λ_{observed}$', **subtitleFont)
plt.plot(x, error,"x",**pointStyle, label="% Error between $λ_{theoretical}$ and $λ_{observed}$")
plt.plot(x_fit, gaussian(x_fit, *ig_peak1), 'r-', label='First Gaussian ≅ 708.19nm', **lineStyle)
plt.plot(x2_fit, gaussian(x2_fit, *ig_peak2), 'r-', label='Second Gaussian ≅ 1500.0nm', **lineStyle)
#plt.plot(x, error,**lineStyle, label="Line of Best Fit (Gaussian)")
plt.xlabel("Theoretical Values / nm",**axesFont)
plt.ylabel("Observed Values / nm",**axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(prop=font, loc="upper right")
plt.savefig("hg_lines_gaussian.png",dpi=1000)
plt.show()

np.savetxt("errors.txt",np.column_stack((x, error)),delimiter="\t", fmt='%s')
np.savetxt("err_vals.txt",np.column_stack((x, y)),delimiter="\t", fmt='%s')