# %% [markdown]
# Author: Ben Campbell
# 
# Purpose: To determine the relative efficiency of various methods of computing the solution to a system of linear equations and to apply this to a system of resistors, capacitors, and inductors.

# %%
!pip install warp-lang # for TAs that may not have the package installed

# %%
'''Part a: make code involve partial pivoting; I also want to try to incorperate the NVIDIA warp package to speed up the algorithm'''

import numpy as np
from numpy import empty
# The following will be useful for partial pivoting
from numpy import empty
from copy import copy
import warp as wp
wp.init()



# %%
#define standar Gaussian elim without pivoting - ripped from lecture notes
def GaussElim_sans_pivot(A_in, v_in):
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    for m in range(N):
        #partial pivot
        j = np.argmax(np.abs(A[m:,m]))#look at mth colum and find index with element furthest from zero
        q = len(A[m:,m])-1
        # notably j is not an idex of the original matrix, but of the subvector that we searched so we need to account for this
        j += m #add m because there are m rows above where we started looking
        A[m,:], A[j,:] = copy(A[j,:]), copy(A[m,:])
        current_v = copy(v[m])
        v[m] = copy(v[j])
        v[j] = current_v

        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x

# %%
#Gauss elim with warp - won't use for much of the lab - I just want to try this; feel free to skip this when grading
# realistically we can't do this completely simultaneuosly because we must work row by row. However, we can perform all of the shear operations in usison which will spead up the process somewhat.
# the tradeoff is that to define the kernal we need to specify the dimension of out matrix in advance which does cut down on the generality of the algorithm
@wp.kernel
def shear_3x3(A_in: wp.mat((3,3),float), v_in: wp.vec(3,float), current_row: int): # must take A_in and v_in as warp arrays, current row takes the index of the row to subtract from the others below it
  tid = wp.tid() # get the current thread id so that warp can work on several steps at once; should wp.launch so that
                 # tid ranges from current_row + 1 to mat_shape[0] (i.e. num threads = mat_shape[0] - current_row + 1)
                #mat_shape = A_in._shape_ # get the dimension of the matrix
  #given the current_row is in the upper triangular form and that our matrix is square we know that each
  #row current_row and below have zero in columns of index 0 through current_row-1 thus we want to multiply by the element in the column of the same index as current_row
  A_in[current_row + tid + 1,:] -= A_in[current_row + tid + 1,current_row] * A_in[current_row,:]
  v_in[current_row + tid + 1] -= A_in[current_row + tid + 1,current_row] * v_in[current_row]

@wp.kernel
def shear_4x4(A_in: wp.mat((4,4),float), v_in: wp.vec(4,float), current_row: int): # must take A_in and v_in as warp arrays, current row takes the index of the row to subtract from the others below it
  tid = wp.tid() # get the current thread id so that warp can work on several steps at once; should wp.launch so that
                 # tid ranges from current_row + 1 to mat_shape[0] (i.e. num threads = mat_shape[0] - current_row + 1)
                #mat_shape = A_in._shape_ # get the dimension of the matrix
  #given the current_row is in the upper triangular form and that our matrix is square we know that each
  #row current_row and below have zero in columns of index 0 through current_row-1 thus we want to multiply by the element in the column of the same index as current_row
  A_in[current_row + tid + 1,:] -= A_in[current_row + tid + 1,current_row] * A_in[current_row,:]
  v_in[current_row + tid + 1] -= A_in[current_row + tid + 1,current_row] * v_in[current_row]


def GaussElim_warp_3x3(A_in, v_in):
  if type(A_in) == np.ndarray: # make sure the inputs are in warp data types otherwise we cannot run the shear function to multithread
    dim = np.shape(A_in)
    A_in = wp.mat(shape=dim,dtype=float)(A_in)
  if type(v_in) == np.ndarray:
    dim = len(v_in)
    v_in = wp.vec(dim,dtype=float)(v_in)
  A = copy(A_in)
  v = copy(v_in)
  N = 3

  for m in range(N):
    #partial pivot; unfortunatly since wp.mat and wp.vec are not easily mutable except elementwise we will convert back to numpy and then to warp again
        A = np.array([[A[i, j] for j in range(3)] for i in range(3)], dtype=float)
        v = np.array([v[i] for i in range(3)], dtype=float)
        j = np.argmax(np.abs(A[m:,m]))#look at mth colum and find index with element furthest from zero
        q = len(A[m:,m])-1
        # notably j is not an idex of the original matrix, but of the subvector that we searched so we need to account for this
        j += m #add m because there are m rows above where we started looking
        current_row = copy(A[m,:])
        A[m,:] = copy(A[j,:])
        A[j,:] = current_row
        current_v = copy(v[m])
        v[m] = copy(v[j])
        v[j] = current_v
        A = wp.mat(shape=(3,3),dtype=float)(A)
        v = wp.vec(3,dtype=float)(v)
    # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div
    # Now perform the shear operations
        wp.launch(shear_3x3,dim=3-m+1,inputs=[A,v,m]) #theretically one step if there is a GPU; auto updates the matrix A and vector v

  # Backsubstitution; could probably find a way to use warp for this too but it's not obvious to me
  # create an array of the same type as the input array
  x = empty(N, dtype=float)
  for m in range(N-1, -1, -1):
      x[m] = v[m]
      for i in range(m+1, N):
          x[m] -= A[m, i]*x[i]
  return x


def GaussElim_warp_4x4(A_in, v_in):
  if type(A_in) == np.ndarray: # make sure the inputs are in warp data types otherwise we cannot run the shear function to multithread
    dim = np.shape(A_in)
    A_in = wp.mat(shape=dim,dtype=float)(A_in)
  if type(v_in) == np.ndarray:
    dim = len(v_in)
    v_in = wp.vec(dim,dtype=float)(v_in)
  A = copy(A_in)
  v = copy(v_in)
  N = 4

  for m in range(N):
    #partial pivot; unfortunatly since wp.mat and wp.vec are not easily mutable except elementwise we will convert back to numpy and then to warp again
        A = np.array([[A[i, j] for j in range(4)] for i in range(4)], dtype=float)
        v = np.array([v[i] for i in range(4)], dtype=float)
        j = np.argmax(np.abs(A[m:,m]))#look at mth colum and find index with element furthest from zero
        q = len(A[m:,m])-1
        # notably j is not an idex of the original matrix, but of the subvector that we searched so we need to account for this
        j += m #add m because there are m rows above where we started looking
        current_row = copy(A[m,:])
        A[m,:] = copy(A[j,:])
        A[j,:] = current_row
        current_v = copy(v[m])
        v[m] = copy(v[j])
        v[j] = current_v
        A = wp.mat(shape=(4,4),dtype=float)(A)
        v = wp.vec(4,dtype=float)(v)
    # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div
    # Now perform the shear operations
        wp.launch(shear_4x4,dim=4-m+1,inputs=[A,v,m]) #theretically one step if there is a GPU; auto updates the matrix A and vector v

  # Backsubstitution; could probably find a way to use warp for this too but it's not obvious to me
  # create an array of the same type as the input array
  x = empty(N, dtype=float)
  for m in range(N-1, -1, -1):
      x[m] = v[m]
      for i in range(m+1, N):
          x[m] -= A[m, i]*x[i]
  return x

# %%
#test
matrix = np.array([[100,2,3],[4,5,6],[7,8,9]])
vector = np.array([1,2,3])
print(GaussElim_warp_3x3(matrix,vector))

matrix = np.array([[100,20,3],[4,50,600],[70,800,9]])
vector = np.array([1,2,3])
print(GaussElim_warp_3x3(matrix,vector))

# %% [markdown]
# So it basically get the right answer, but the error is kind of large.

# %%
def GaussElim(A_in, v_in): # numpy partial pivoting
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        #partial pivot
        j = np.argmax(np.abs(A[m:,m]))#look at mth colum and find index with element furthest from zero
        q = len(A[m:,m])-1
        # notably j is not an idex of the original matrix, but of the subvector that we searched so we need to account for this
        j += m #add m because there are m rows above where we started looking
        A[m,:], A[j,:] = copy(A[j,:]), copy(A[m,:])
        current_v = copy(v[m])
        v[m] = copy(v[j])
        v[j] = current_v

        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


# %%
#test
matrix = np.array([[100.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
vector = np.array([1.,2.,3.])
print(GaussElim(matrix,vector))
matrix = np.array([[100.,20.,3.],[4.,50.,600.],[70.,800.,9.]])
vector = np.array([1.,2.,3.])
print(GaussElim(matrix,vector))

# %% [markdown]
# Error for the numpy method is better and this method works for a general N×N matrix.

# %%
#implement equation 6.2 from Newman:
M = np.array([
    [2,1,4,1],
    [3,4,-1,-1],
    [1,-4,1,5],
    [2,-2,1,3]
],float)
v = np.array([-4,3,9,7],float)

print("Solution to equation 6.2: ",sol:=GaussElim(M,v))
#print("Solution to equation 6.2 with warp: ", sol_warp:=GaussElim_warp_4x4(M,v)); something seems to break in the 4x4 case but its not important to fix for the purpose of this lab

# %% [markdown]
# This matches what Newman reports in equation 6.16.

# %%
# Part B
#For each of a range of values of N, creates a random matrix A and a random array v
def rand_mat_vec(N):
  return np.random.rand(N,N), np.random.rand(N)

print(rand_mat_vec(5))

# %%
from time import time
def part_b(N,should_print: bool):
  mat, vec = rand_mat_vec(N) #generate matrix
  try:
    s_clock = time() #start the timer
    sol_sans_pivot = GaussElim_sans_pivot(mat,vec) # standard gaussian elim - may not work due to divide by zero
    elapsed_sans_pivot = time() - s_clock
  except:
    sol_sans_pivot = False
    elapsed_sans_pivot = False

  s_clock = time() #start the timer
  sol_pivot = GaussElim(mat,vec)
  elapsed_pivot = time() - s_clock

  s_clock = time() #start the timer
  sol_LU = np.linalg.solve(mat,vec) #uses LU decomposition
  elapsed_LU = time() - s_clock

  #calculate errors
  err_sans_pivot = np.mean(np.abs(vec - np.dot(mat,sol_sans_pivot)))
  err_pivot = np.mean(np.abs(vec - np.dot(mat,sol_pivot)))
  err_LU = np.mean(np.abs(vec - np.dot(mat,sol_LU)))

  '''if N ==3:#out of curiosity, I want to test the warp method
      s_clock = time() #start the timer
      sol_warp = GaussElim_warp_3x3(mat,vec)
      elapsed_warp = time() - s_clock
      err_warp = np.mean(np.abs(vec - np.dot(mat,sol_warp)))
      print("Solution with warp: {0}\nError with warp: {1}\nTime with warp: {2}".format(sol_warp, err_warp, elapsed_warp))'''

  if should_print:
    print(
        "Solution without pivoting: {0}\nSolution with pivoting: {1}\nSolution with LU decomposition {2}\n".format(
            sol_sans_pivot, sol_pivot, sol_LU)
    )
    print(
        "Error without pivoting: {0}\nError with pivoting: {1}\nError with LU decomposition {2}\n".format(
            err_sans_pivot, err_pivot, err_LU)
    )
    print(
        "Time without pivoting: {0}\nTime with pivoting: {1}\nTime with LU decomposition {2}\n".format(
            elapsed_sans_pivot, elapsed_pivot, elapsed_LU)
    )

  return [sol_sans_pivot, sol_pivot, sol_LU, err_sans_pivot, err_pivot, err_LU, elapsed_sans_pivot, elapsed_pivot, elapsed_LU]

# %%
#Now we run this for many differently sized matricies

store_informaion = {
    "error_sans_pivot": [],
    "error_pivot": [],
    "error_LU": [],
    "time_sans_pivot": [],
    "time_pivot": [],
    "time_LU": []
}

for n in range(5,200): # size of matrix
  for i in range(1): # do some number of matricies of each size
    L = part_b(n,should_print=False)
    store_informaion["error_sans_pivot"].append(L[3])
    store_informaion["error_pivot"].append(L[4])
    store_informaion["error_LU"].append(L[5])
    store_informaion["time_sans_pivot"].append(L[6])
    store_informaion["time_pivot"].append(L[7])
    store_informaion["time_LU"].append(L[8])

#graph errors
import matplotlib.pyplot as plt
mat_size = np.arange(5,200,1)
plt.plot(mat_size,store_informaion["error_sans_pivot"], label="error without pivoting", color = "r")
plt.plot(mat_size,store_informaion["error_pivot"], label="error with pivoting",ls = "-.", color = 'cyan')
plt.plot(mat_size,store_informaion["error_LU"], label="error with LU decomposition", ls = ":", color = 'k')
plt.ylabel('error')
plt.xlabel('matrix size')
plt.title("Errors of three methods of solving linear\nsystems for matricies of dimension 5 to 200")
ax = plt.gca()
ax.set_yscale('log')
plt.legend()
plt.show()


#graph times
mat_size = np.arange(5,200,1)
plt.plot(mat_size,store_informaion["time_sans_pivot"], label="time ellapsed without pivoting")
plt.plot(mat_size,store_informaion["time_pivot"], label="time ellapsed with pivoting",ls = "-.")
plt.plot(mat_size,store_informaion["time_LU"], label="time ellapsed with LU decomposition", ls = ":")
plt.ylabel('Time Ellapsed (s)')
plt.xlabel('matrix size')
plt.title("Computation time of three methods of solving\nlinear systems for matricies of dimension 5 to 200")
ax = plt.gca()
ax.set_yscale('log')
plt.legend()
plt.show()

# %%
# Stack the error plots for clairity
fig, (ax1,ax2,ax3) = plt.subplots(3)
ax1.set_title("Errors of three methods of solving linear\nsystems for matricies of dimension 5 to 200")
ax1.plot(mat_size,store_informaion["error_sans_pivot"], label="error without pivoting", color = "r")
ax2.plot(mat_size,store_informaion["error_pivot"], label="error with pivoting", color = 'cyan')
ax3.plot(mat_size,store_informaion["error_LU"], label="error with LU decomposition", color = 'k')
plt.ylabel('error')
plt.xlabel('matrix size')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()

# %%
#Part c - voltages

R_1, R_3, R_5 = [1e3 for i in range(3)] # Ohms
R_2, R_4, R_6 = [2e3 for i in range(3)] # Ohms

C_1 = 1e-6 # F
C_2 = 5e-7 # F
xp = 3 #V
omega = 1000 #Hz

Matrix = np.array(
    [
        [(1/R_1 + 1/R_4 + 1j*omega*C_1), -1j*omega*C_1 ,0], #equation 1
        [-1j*omega*C_1, (1/R_2 + 1/R_5 + 1j*omega*(C_1+C_2))  , -1j*omega*C_2], #equation 2
        [0, -1j*omega*C_2, (1/R_3 + 1/R_6 + 1j*omega*C_2) ] #equation 3
    ]
)

vector = np.array(
    [xp/R_1, xp/R_2, xp/R_3],dtype=np.complex128
)

# %%
# Solve the system
print("solution to voltages: ", sol := GaussElim(Matrix,vector))
print("Phases: ", phases:= np.angle(sol,deg=False))
print("Amplitude: ", amps:= np.abs(sol))

# %%
def real_voltage(t,omega,voltage,phase): #voltage should be comple
  return voltage.real*np.cos(omega*t+phase) - voltage.imag*np.sin(omega*t+phase)

# plot the real voltages for 2 periods
times = np.linspace(0,4*np.pi/omega,500)
for i in range(3):
  plt.plot(times, real_voltage(times,omega,sol[i],phases[i]), label="voltage {0}".format(i+1) )
plt.legend()
plt.xlabel("time (s)")
plt.title("The real parts of the voltages in the\ncircuit show over two periods")
plt.ylabel(r"voltage ($\Omega$)")
plt.show()

# %%
#Now repeat the above for an inductor

R_6 = 2j*omega

Matrix = np.array(
    [
        [(1/R_1 + 1/R_4 + 1j*omega*C_1), -1j*omega*C_1 ,0], #equation 1
        [-1j*omega*C_1, (1/R_2 + 1/R_5 + 1j*omega*(C_1+C_2))  , -1j*omega*C_2], #equation 2
        [0, -1j*omega*C_2, (1/R_3 + 1/R_6 + 1j*omega*C_2) ] #equation 3
    ]
)

vector = np.array(
    [xp/R_1, xp/R_2, xp/R_3],dtype=np.complex128
)

# Solve the system
print("solution to voltages: ", sol := GaussElim(Matrix,vector))
print("Phases: ", phases:= np.angle(sol,deg=False))
print("Amplitude: ", amps:= np.abs(sol))

#plot
times = np.linspace(0,4*np.pi/omega,500)
for i in range(3):
  plt.plot(times, real_voltage(times,omega,sol[i],phases[i]), label="voltage {0}".format(i+1) )
plt.legend()
plt.xlabel("time (s)")
plt.ylabel(r"voltage ($\Omega)$")
plt.title("The real parts of the voltages in the\ncircuit show over two periods")
plt.show()

# %%



