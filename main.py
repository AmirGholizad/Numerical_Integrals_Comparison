# Add the libraries that we need
import time
import math
import scipy as sp
from scipy import integrate as intg
import numpy as np
import pandas as pd
import decimal
import random
from matplotlib import pyplot as plt
import tabulate

# Define the function we are going to integrate and fit data to
def f(x):
	#define the function which you're going to fit the data to
	f = math.sin(x)
	return f

# This function takes a continuous range of numbers and returns the same range as a discrete data
# then calculates the f(x) in these points and fits them to function
def Discretizator(N, a, b):

	F = np.zeros(N+1)
	X = np.zeros(N+1)
	DeltaX = (b-a)/N
	for i in range(0,len(X)):
		X[i] = a+(i*DeltaX)

	for j in range(0,len(F)):
		F[j] = f(X[j])

	return [DeltaX,X,F]

# This function uses a polynomial to fit the data
# We will not use it during this project
def Discretizator2(N, a, b):

	X = func1(N,a,b)[0]
	F = func1(N,a,b)[1]
	S = np.zeros((N+1,N+1))
	for l in range(0,len(X)):
		for k in range(0,len(S)):
			S[l][k] = X[l]**k
	SInverse = np.linalg.inv(S)
	C = SInverse.dot(F)
	Fit = S.dot(C)
	#for m in range(0,len(C)):
	#	print('+',C[m],'*','x','**',m)
	return [X,S,SInverse,C,Fit]

# This function integrates the defined f(x) based on MonteCarlo Method
def MontecarloIntegral(N, a, b, turn):
	start_time = time.time()
	result = np.zeros(turn)
	for q in range(0,turn):
		summation = 0
		for i in range(0,N):
			x = random.uniform(a,b)
			summation = summation + f(x)
		result[q] = ((b-a)/N)*summation
	resultavg = np.mean(result)
	stop_time = time.time()
	run_time = round(stop_time - start_time,3)
	return [resultavg,run_time]

# This function integrates the defined f(x) based on Trapezoidal Method
def TrapezoidalIntagral(N, a, b):
	start_time = time.time()
	DeltaX = Discretizator(N, a, b)[0]
	X = Discretizator(N, a, b)[1]
	F = Discretizator(N, a, b)[2]
	Result = 0
	for k in range(1,N+1):
		Result = Result + DeltaX*(F[k-1]+F[k])/2
	stop_time = time.time()
	run_time = round(stop_time - start_time,3)
	return [Result,run_time]

# This function integrates the defined f(x) using direct integral of Scipy
def DirectIntegral(a, b):
	f = lambda x:math.sin(x)
	I = intg.quad(f,a,b)
	return I[0]

# This function integrates the defined f(x) based on Simpson Method
def SimpsonIntegral(N, a, b, ):
	start_time = time.time()
	Deltax = Discretizator(N, a, b)[0]
	X = Discretizator(N, a, b)[1]
	F = Discretizator(N, a, b)[2]
	I = 0
	for i in range(0,N+1):
		if i == 0 or i == N:
			I = I + F[i]
		elif (i % 2) == 0:
			I = I + 2*F[i]
		elif (i % 2) != 0:
			I = I + 4*F[i]
	I = I*(Deltax/3)
	stop_time = time.time()
	run_time = round(stop_time - start_time,3)
	return [I,run_time]


# gather all the data into a DataFrame
Intg0 = DirectIntegral(0, math.pi)
Intg1 = TrapezoidalIntagral(100000, 0, math.pi)
Intg2 = MontecarloIntegral(100000, 0, math.pi, 10)
Intg3 = SimpsonIntegral(100000, 0, math.pi)

df = pd.DataFrame([['sin(x)','0,pi','Direct Integral',Intg0,'---','---','1']
					 ,['','','Trapezoidal Intagral',Intg1[0],round(abs(100*(Intg1[0]-Intg0)/Intg0),3),Intg1[1],'1']
					 ,['','','Montecarlo Integral',Intg2[0],round(abs(100*(Intg2[0]-Intg0)/Intg0),3),Intg2[1],'10']
					 ,['','','Simpson Integral',Intg3[0],round(abs(100*(Intg3[0]-Intg0)/Intg0),3),Intg3[1],'1']]
				  ,columns=['Function','Range','Method','Result','Error%','Runtime(seconds)','Run(s)'])

# print the DataFrame
print(tabulate.tabulate(df, showindex=False, headers=df.columns,numalign="left"))