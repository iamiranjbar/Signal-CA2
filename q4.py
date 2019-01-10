import numpy as np
import csv
import sys
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

def nextpow2(x):
	return (x-1).bit_length()

def show_fft(arr, name, L):
    NFFT = 2** nextpow2(L)
    X = 2*np.absolute(fft.fft(arr, n=NFFT))/L
    half = int(NFFT/2)
    freq = fft.fftfreq(NFFT, d=1/fs)
    plt.plot(freq[:half], X[:half])
    plt.title(name)
    plt.grid()
    plt.show()

fs , x = wavfile.read('sound.wav')
print(len(x))
print(fs)
arr = np.zeros(len(x))
for i in range(0,len(x)):
    a = (x[i][0] + x[i][1])/2
    arr[i] = a
before = np.array(arr, dtype=np.int16)
show_fft(before, 'Before Downsample', len(x))
arr = []
for i in range(0,len(before),3):
  arr.append(before[i])
after = np.array(arr)
show_fft(after, 'After Downsample', int(len(x)/3))
wavfile.write('result.wav', int(fs/3), after)