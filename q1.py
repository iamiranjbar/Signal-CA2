import numpy as np
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

def butter_bandstop_filter(lowcut, highcut, fs, order=5):
	nyq = 0.5* fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = signal.butter(N=order, Wn=[low, high], btype='bandstop')
	return b, a

def nextpow2(x):
	return (x-1).bit_length()

fs, x = wavfile.read('soundCA2.wav')

L = len(x)
print('L =', L, sep=' ')
NFFT = 2** nextpow2(L)
X = fft.fft(x, n=NFFT)
X_abs = 2* np.absolute(X) / L
half = int(NFFT/2)
freq = fft.fftfreq(NFFT, d=1/fs)
b, a = butter_bandstop_filter(1550, 1750, fs)
w, h = signal.freqz(b, a)
y = signal.lfilter(b, a, x)

Ly = len(y)
print('L =', Ly, sep=' ')
NFFTy = 2** nextpow2(Ly)
Y = fft.fft(y, n=NFFTy)
Y_abs = 2* np.absolute(Y) / Ly
halfy = int(NFFTy/2)
freqy = fft.fftfreq(NFFTy, d=1/fs)


fig, ax = plt.subplots(1, 1)
ax.plot(freq[:half], X_abs[:half])
ax.set_title('Single-Sided Amplitude Spectrum of x(t)')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|X(f)|')
ax.grid()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(w, abs(h))
ax.set_title('Filter abs')
ax.set_xlabel('w')
ax.set_ylabel('|H(w)|')
ax.grid()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(freqy[:halfy], Y_abs[:halfy])
ax.set_title('Single-Sided Amplitude Spectrum of y(t)')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|Y(f)|')
ax.grid()
plt.show()

wavfile.write('noiseless.wav', fs, y)