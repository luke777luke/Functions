#Optimization of various Pearson Correlation Coeffienct algorithms for time efficiency

import time
import numpy as np

#classic formula
def corr(a,b):
    amean = a - np.mean(a)
    bmean = b - np.mean(b)
    top = 0
    bota = 0
    botb = 0
    for i,j in zip(amean,bmean):
        top += (i)*(j)
        bota += (i)**2
        botb += (j)**2
    bota = np.sqrt(bota)
    botb = np.sqrt(botb)
    p = (top)/(bota*botb)
    return p

#checks how long it takes to run the first function to find the correlation between two 1D random arrays 1000x over
tic = time.time()
for i in range(1000):
    a = np.random.randn(500)
    b = np.random.randn(500)
    corr(a,b)
tic2 = time.time()-tic
print(f'Using the classic formula, the time it takes to calculate the correlation between two 1D random arrays 1000 times over is {tic2} seconds.')
    
#classic formula translated to linear algebra
def corr2(a,b):
    amean = a - np.mean(a)
    bmean = b - np.mean(b)
    top = np.dot(amean,bmean)
    bota = np.linalg.norm(amean)
    botb = np.linalg.norm(bmean)
    p = (top)/(bota*botb)
    return p

#checks how long it takes to run the second function to find the correlation between two 1D random arrays 1000x over
tic = time.time()
for i in range(1000):
    a = np.random.randn(500)
    b = np.random.randn(500)
    corr2(a,b)
tic2 = time.time()-tic
print(f'Using the algorithm based on linear algebraic functions, the time it takes to calculate the correlation between two 1D random arrays 1000 times over is {tic2} seconds.')

#built-in NumPy function
def corr3(a,b):
    p = np.corrcoef(a,b)[0,1]
    return p

#checks how long it takes to run the third function to find the correlation between two 1D random arrays 1000x
tic = time.time()
for i in range(1000):
    a = np.random.randn(500)
    b = np.random.randn(500)
    corr3(a,b)
tic2 = time.time()-tic
print(f'Using the built in function, the time it takes to calculate the correlation between two 1D random arrays 1000 times over is {tic2} seconds.')

#additionally, check to confirm the accuracy of calculating the Pearson Correlation Coeffienct by comparing the first two functions with the results of the built in NumPy function
veca = np.random.randn(500)
vecb = np.random.randn(500)
print(f'If the resulting value of the first function ({corr(veca,vecb).round(5)}) and second function ({corr2(veca,vecb).round(5)})  match the value of the built-in function ({corr3(veca,vecb).round(5)}), then accuracy is preserved.')

print(f'The most time efficiient algoritm  is  the one using linear algebraic functions.')