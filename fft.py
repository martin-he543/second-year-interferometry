import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf
import matplotlib.font_manager as fnt

titleFont =     {'fontname': 'C059', 'size': 13}
axesFont =      {'fontname': 'C059', 'size': 9}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
font = fnt.FontProperties(family='C059', weight='bold', style='normal', size=8)

def Gaussian_Fit(x, A, mu, sd, D):
    top = -(x-mu)**2
    bottom = 2*(sd**2)
    return A*np.exp(top/bottom) + D

results = np.loadtxt('data/Output_data_task_11_1.txt', unpack=True)
y1 = np.array(results[1])
x = np.array(results[5])
nsamp = len(x)
'''
# take a fourier transform
yf = spf.fft(y1)
# setting the correct x-axis for the fourier transform. Osciallations/step
xf = spf.fftfreq(nsamp)

# now some shifts to make plotting easier (google if ineterested)
xf = spf.fftshift(xf)
yf = spf.fftshift(yf)
'''
mperstep = 2*1.56*10**-11
x = x * mperstep
y = y1
mpersample = np.mean(np.diff(x))

yf = spf.fft(y)
xf = spf.fftfreq(nsamp)
xf = xf[1:]
yf = yf[1:]

xxf = xf[:int(nsamp/2)]
yyf = yf[:int(nsamp/2)]
xxf = np.abs(mpersample / xxf)
yyf = np.abs(yyf)  # to ensure no negative wavelength

#xxf = xxf + 50e-9
initial_guess2 = [2e8, 3.6e-7, 3e-9, 0]
fit2, cov2 = sp.optimize.curve_fit(Gaussian_Fit, xxf, yyf, initial_guess2, bounds=(
    (0, 0, 0, 0,), (np.inf, np.inf, np.inf, np.inf)), maxfev=1000000)

print("Mean Wavelength 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))
print("Standard Deviation 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))

plt.figure("Wavelength", dpi=500)
ax = plt.axes()
#plt.grid(lw=0.5)
ax.set_ylim(0, 2.3*10**8)
ax.set_xlim(0.1*10**-6, 800e-9)
ax.set_facecolor('black')
plt.plot(xxf, yyf, color='Cyan', marker='', label='Experimental Data')
plt.plot(xxf, Gaussian_Fit(xxf, fit2[0], fit2[1], fit2[2], fit2[3]), color='Red', label='Gaussian Fit', ls='--')
plt.xlabel("Wavelength (m)",**axesFont)
plt.ylabel("Amplitude",**axesFont)
plt.legend(prop=font)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.show()