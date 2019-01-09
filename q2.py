import numpy as np
import csv
import sys
from scipy.io import wavfile
from numpy import fft
from matplotlib import pyplot as plt
from scipy import signal

csv.field_size_limit(sys.maxsize)

def p1():
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	#print(d)
	for x in range(0,10):
		wavfile.write('q2p1' + str(x) + '.wav', 8192 , d[x])

def p2():
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	for x in range(0,10):
		L = len(d[x])
		#print('L =', L, sep=' ')
		#* nextpow2(L)
		X = fft.fft(d[x], n=2048)
		X_abs = 2* np.absolute(X) / L
		half = int(2048/2)
		freq = fft.fftfreq(2048, d=1/8192)
		fig, ax = plt.subplots(1, 1)
		ax.plot(freq[:half], X_abs[:half])
		ax.set_title('d' + str(x))
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('|d' +str(x)+'|')
		ax.grid()
		plt.show()

def p3():
	n = np.array([n for n in range(0, 1000)])
	z = [0 for n in range(0, 100)]
	print(z)
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	pat = ['0','z','1','z','9','z','5','z','4','z','2','z','7','z']
	phone = []
	for x in pat:
		if x == 'z':
			for n in range(0, 100):
				phone += [0]
		else:
			for n in range(0, 1000):
				phone += [d[ord(x) - ord('0')][n]]
	#phone = d[0] + z + d[1] + z + d[9] + z + d[5] + z + d[4] + z + d[2] + z + d[7] + z
	wavfile.write('sid.wav', 8192 , np.array(phone))

def p4():
	phoneReader = [[], []]
	phoneReader[0] = list(csv.reader(open('phone1.csv', newline=''), delimiter=','))
	phoneReader[1] = list(csv.reader(open('phone2.csv', newline=''), delimiter=','))
	print(phoneReader[0][0])
	print(phoneReader[1][0])
	for r in range(0,2):
		dt = [[0 for k in range(0,1000)] for x in range(0,7)]
		# print(dt)
		base = 0
		for x in range(0,7):
			for i in range(0,1000):
				dt[x][i] = phoneReader[r][0][base+i]
			base += 1100
		for x in range(0,7):
			L = len(dt[x])
			#print('L =', L, sep=' ')
			#* nextpow2(L)
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

def comp(l1, l2):
	print(l1[1])
	print(l2[1])
	for x in range(0,1000):
		if l1[x] != l2[x]:
			return False
	return True


def ttdecode(inlist):
	n = np.array([n for n in range(0, 1000)])
	c = [[0.7217, 1.0247], [0.5346, 0.9273], [0.5346, 1.0247], [0.5346, 1.1328], [0.5906, 0.9273], [0.5906, 1.0247], [0.5906, 1.1328], [0.6535, 0.9273], [0.6535, 1.0247], [0.6535, 1.1328]]
	d = [[] for x in range(0,10)]
	for x in range(0,10):
		d[x] = np.sin(c[x][0]*n) + np.sin(c[x][1]*n)
	dt = [[0 for k in range(0,1000)] for x in range(0,7)]
	# print(dt)
	base = 0
	for x in range(0,7):
		for i in range(0,1000):
			dt[x][i] = inlist[base+i]
		base += 1100
	for x in range(0,7):
		for y in range(0,10):
			# c = (np.array(dt[x]) == np.array(d[y]))
			if comp(dt[x],d[y]):
				if not x == 6:
					print(y, end = " ")
				else:
					print(y)


def p5():
	phoneReader = [[], []]
	phoneReader[0] = list(csv.reader(open('phone1.csv', newline=''), delimiter=','))
	phoneReader[1] = list(csv.reader(open('phone2.csv', newline=''), delimiter=','))
	# print(phoneReader[0][0])
	# print(phoneReader[1][0])
	for r in range(0,2):
		ttdecode(phoneReader[r][0])

p5()