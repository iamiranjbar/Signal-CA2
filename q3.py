import numpy as np
import csv
import sys
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

def p1():
	n = np.array([n for n in range(0, 41)])
	t=np.linspace(0, 1, num=1000, endpoint=False)
	yt = np.sin((np.pi*t)/10)
	yn = np.sin((np.pi*n)/10)
	plt.plot(t, yt)
	plt.stem(n, yn)
	plt.legend(['x(t)', 'x(n)'], loc='best')
	plt.show()

p1()