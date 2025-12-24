# %% [markdown]
# Author: Ben Campbell
# 
# Purpose: To integrate Burger's equation and test the efficacy of solutions. Burger's equation is $\partial_t u + ϵu\partial_x u = 0$.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# parameters

eps = 1.
Delta_x = 0.02 # spatial step
Delta_t = 0.05 # time step
beta = eps*Delta_t/Delta_x

L_x = 2*np.pi # spatial domain size
T_f = 2 # period of simulation

N_x = round(L_x/Delta_x) # number of space partitions
N_t = round(T_f/Delta_t) # number of time partitions

# initial conditions

u_t0 = lambda x: np.sin(x) # u(x,t=0)=sin(x)
u_x0 = 0 # u(x=0,t) = 0
u_xf = 0 # u(x=L_x,t) = 0

# %%
# initialize arrays

xt = np.zeros([N_x,N_t]) # so time progresses from left to right and space from top to bottom in the array
xt[:,0] = u_t0(np.linspace(0,L_x,N_x)) # initial condition

# %%
# define the leapfrog method function

def Burger_leapfrog(M : np.ndarray):
  S = np.copy(M) # we return a mutated array - this ensures we do not overwrite the original array in case it is still desired.
  n, m = S.shape #(rows, columns)
  # Initial euler time step:
  dudx = [np.cos(k*Delta_x) for k in range(1,n)] # use the derivative of the initial condition to estimate $\partial_x u(x,t=0)$
  dudt = -eps*S[1:n,0]*dudx # use the derivative of x and Burger's equation to estimate the time derivative
  S[1:n,1] = S[1:n,0] + dudt*Delta_t # Initial Euler estimate
  for j in range(1,m-1): # each j is a time position; already have the initial conditions
    # must do columns first - for loops may not be interchanged
    for i in range(1,n-1): # each i is a spatial position; we already have the boundary conditions at the first and last points
      # perform the array update
      S[i,j+1] = S[i,j-1] - (beta/2 * ( S[i+1,j]**2 - S[i-1,j]**2 ))
  return S

# %%
#Run the function on our array

Solution_xt = Burger_leapfrog(xt)

# %%
# make an animation
from matplotlib.animation import FuncAnimation

def update_plot(frame):
  p.set_ydata(Solution_xt[:,frame])
  fig.suptitle("Burgers' Wave (t = {0:.3f})".format(Delta_t*frame))
  return p,
#original plot
fig, ax = plt.subplots()
p, = ax.plot(np.linspace(0,L_x,N_x),Solution_xt[:,0])
fig.suptitle("Burgers' Wave (t = {0:.3f})".format(0))
ax.set_ylabel("u(x,t)")
ax.set_xlabel("x")

ani = FuncAnimation(fig,update_plot,frames=N_t)

# %%
ani.save("./wave.gif", writer="pillow", fps=5)

# %%
# now show the graphs required for the report

plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,10])
plt.title("Burgers' Wave (t = {0:.3f})".format(10*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()

# %%
plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,30])
plt.title("Burgers' Wave (t = {0:.3f})".format(30*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()

# %%
plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,20])
plt.title("Burgers' Wave (t = {0:.3f})".format(20*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()


