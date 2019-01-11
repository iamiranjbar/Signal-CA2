import numpy as np
import csv
import sys
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

csv.field_size_limit(sys.maxsize)

def part1():
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	for x in range(0,10):
		wavfile.write('q2_part1_' + str(x) + '.wav', 8192 , d[x])

def part2():
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	for x in range(0,10):
		L = len(d[x])
		X = fft.fft(d[x], n=2048)
		X_abs = 2* np.absolute(X) / L
		half = 1024
		freq = fft.fftfreq(2048, d=1/8192)
		fig, ax = plt.subplots(1, 1)
		ax.plot(freq[:half], X_abs[:half])
		ax.set_title('d' + str(x))
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('|d' +str(x)+'|')
		ax.grid()
		plt.show()

def part3():
	n = np.array([n for n in range(0, 1000)])
	space = np.zeros(100)
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	pattern = [0,-1,1,-1,9,-1,5,-1,4,-1,0,-1,2,-1]
	phone = []
	for x in pattern:
		if x == -1:
			phone.extend(space)
		else:
			phone.extend(d[x])
	wavfile.write('sid.wav', 8192 , np.array(phone))

def part4():
	phoneData = [[], []]
	phoneData[0] = list(csv.reader(open('phone1.csv', newline=''), delimiter=','))
	phoneData[1] = list(csv.reader(open('phone2.csv', newline=''), delimiter=','))
	for r in range(0,2):
		dt = [[0 for k in range(0,1000)] for x in range(0,7)]
		base = 0
		for x in range(0,7):
			for i in range(0,1000):
				dt[x][i] = phoneData[r][0][base+i]
			base += 1100
		for x in range(0,7):
			L = len(dt[x])
			X = fft.fft(dt[x], n=2048)
			X_abs = 2* np.absolute(X) / L
			half = int(2048/2)
			freq = fft.fftfreq(2048, d=1/8192)
			fig, ax = plt.subplots(1, 1)
			ax.plot(freq[:half], X_abs[:half])
			ax.set_title('digit' + str(x+1) + 'signal' +str(r+1))
			ax.set_xlabel('Frequency [Hz]')
			ax.set_ylabel('|dt' +str(x)+'|')
			ax.grid()
			plt.show()

def ttdecode(inlist):
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	dt = np.zeros((7,1000))
	base = 0
	for x in range(0,7):
		for i in range(0,1000):
			dt[x][i] = inlist[base+i]
		base += 1100
	result=[]
	for x in range(0,7):
		for y in range(0,10):
			if np.allclose(dt[x],d[y],atol=1):
				result.append(y)
				break
	return result


def part5():
	phoneReader = [[], []]
	phoneReader[0] = np.array(list(csv.reader(open('phone1.csv', newline=''), delimiter=',')))
	phoneReader[1] = np.array(list(csv.reader(open('phone2.csv', newline=''), delimiter=',')))
	for r in range(0,2):
		testout = ttdecode(phoneReader[r][0])
		print(testout)

part5()