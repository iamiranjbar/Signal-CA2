import numpy as np
import csv
import sys
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

def show_fft(arr, name, L):
	X = fft.fft(x)/L
	plt.plot(X)
	plt.title(name)
	plt.grid()
	plt.show()

fs , x = wavfile.read('sound.wav',False)
print(len(x))
print(fs)
arr = np.array([0 for n in range(0, len(x))], float)
for i in range(0,len(x)):
  a = (x[i][0] + x[i][1])/2
  arr[i] = a
carray = np.array(arr, dtype=np.int16)
show_fft(carray, 'Before Downsample', fs)
arr = []
for i in range(0,len(carray),3):
  arr.append(carray[i])
nparr = np.array(arr)
show_fft(carray, 'After Downsample', fs/3)
wavfile.write('result.wav', int(fs/3), nparr)
	
