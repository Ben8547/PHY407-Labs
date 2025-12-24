#Author: Ben Campbell
#Purpose: Compare the numerical implementation of two equivalent formulas for the standard deviation of a data set.

import numpy as np

array = np.loadtxt("./cdata.txt") #load the array

#find the standard deviation with numpy:

real_std = np.std(array,ddof=1)

def quick_std(data): #define the equivalent formula for standard deviation
    n = len(data)
    sum_x2 = 0
    sum_x = 0
    for i in range(n):
        sum_x += data[i]
        sum_x2 += data[i]**2
    sum = sum_x2 - sum_x**2/n
    return np.sqrt(sum/(n-1))

def long_std(data):
    n = len(data)
    #My CS friends said my loops were slow so I implemeted a vectorized calculation instead:
    #Initialize the matrix 
    '''up_triangle_matrix = np.zeros((n,n))
    for i in range(n):
        up_triangle_matrix[i,:] = [data[i] if k >= i else 0 for k in range(n)]''' # not needed
    matrix = np.zeros((n,n))
    for i in range(n): #not counting this loop because it just sets up the matrix - there is proably a quicker way to do this anyway
        matrix[i,:] = [data[i]] * n
    #compute the inner sum:
    vec = np.matvec(matrix,data)/n # loop number 1
    return np.sqrt((np.sum(data**2-vec)) / (n-1) ) #loop number 2


std_q = quick_std(array)
std_l = long_std(array)

#find relative error

error_q = np.abs(real_std - std_q)/real_std
error_l = np.abs(real_std - std_l)/real_std

print("The realtive error in equation (2): %.3e"%(error_q))
print("The realtive error in equation (1): %.3e"%(error_l))