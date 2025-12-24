#Author: Ben Campbell
#Purpose: Compare the numerical implementation of two equivalent formulas for the standard deviation of a data set.

'''
Pseudocode:

Sample (2000 trials) two normal distrobutions with means 0, 1.e7 standard deviations 1, 1.

Use the method from the previous questions to compare both user std function to the numpy one.

'''

import numpy as np

mean_1, sigma_1, n_1 = (0., 1., 2000)
mean_2, sigma_2, n_2 = (1.e7, 1., 2000)

sample_1 = np.random.normal(mean_1,sigma_1,n_1)
sample_2 = np.random.normal(mean_2,sigma_2,n_2)

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
    matrix = np.zeros((n,n))
    for i in range(n): #not counting this loop because it just sets up the matrix - there is proably a quicker way to do this anyway
        matrix[i,:] = [data[i]] * n
    #compute the inner sum:
    vec = np.matvec(matrix,data)/n # loop number 1
    return np.sqrt((np.sum(data**2-vec)) / (n-1) ) #loop number 2

std_real_1 = np.std(sample_1,ddof=1)
std_real_2 = np.std(sample_2,ddof=1)

std_q_1 = quick_std(sample_1)
std_q_2 = quick_std(sample_2)

std_l_1 = long_std(sample_1)
std_l_2 = long_std(sample_2)

rel_error_q_1 = np.abs(std_real_1 - std_q_1)/ std_real_1
rel_error_q_2 = np.abs(std_real_2 - std_q_2)/ std_real_2
rel_error_l_1 = np.abs(std_real_1 - std_l_1)/ std_real_1
rel_error_l_2 = np.abs(std_real_2 - std_l_2)/ std_real_2

print("Relative error in sample 1 for equation (1): {0:.3e} \nRelative error in sample 1 for equation (2): {1:.3e} \nRelative error in sample 2 for equation (1): {2:.3e} \nRelative error in sample 2 for equation (2): {3:.3e}".format(rel_error_l_1,rel_error_q_1,rel_error_l_2,rel_error_q_2))