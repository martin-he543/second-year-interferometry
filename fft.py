import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as spf


def Gaussian_Fit(x, A, mu, sd, D):
    top = -(x-mu)**2
    bottom = 2*(sd**2)
    return A*np.exp(top/bottom) + D


results = np.loadtxt('C:/Users/44738/Downloads/blueled1.txt', unpack=True)
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
'''
initial_guess1 = [1.2e8, 5.5e-7, 0.5e-7, 0]
fit1, cov1 = sp.optimize.curve_fit(Gaussian_Fit, xxf[:67], yyf[:67], initial_guess1, bounds=(
    (0, 0, 0, 0,), (np.inf, np.inf, np.inf, np.inf)), maxfev=1000000)
'''
initial_guess2 = [2e8, 3.6e-7, 3e-9, 0]
fit2, cov2 = sp.optimize.curve_fit(Gaussian_Fit, xxf, yyf, initial_guess2, bounds=(
    (0, 0, 0, 0,), (np.inf, np.inf, np.inf, np.inf)), maxfev=1000000)

print("Mean Wavelength 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))
print("Standard Deviation 1:", str(fit2[1]), "±", str(np.sqrt(cov2[1][1])))
'''
print("Mean Wavelength 2:", str(fit1[1]), "±", str(np.sqrt(cov1[2][2])))
print("Standard Deviation 2:", str(fit1[1]), "±", str(np.sqrt(cov1[2][2])))
'''
plt.figure("Wavelength", dpi=500)
ax = plt.axes()
plt.grid(lw=0.5)
ax.set_ylim(0, 2.3*10**8)
ax.set_xlim(0.1*10**-6, 800e-9)
ax.set_facecolor('black')
plt.plot(xxf, yyf, color='Cyan', marker='', label='Experimental Data', lw=1.5)
'''
plt.plot(xxf[:67], Gaussian_Fit(
    xxf[:67], fit1[0], fit1[1], fit1[2], fit1[3]), color='red', label='Gaussian Fit 2',ls='--')
'''
plt.plot(xxf, Gaussian_Fit(
    xxf, fit2[0], fit2[1], fit2[2], fit2[3]), color='Red', label='Gaussian Fit', ls='--')
plt.xlabel("Wavelength (m)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()