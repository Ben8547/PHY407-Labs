# %%
'''
Author: Ben Campbell

Purpose: To compare the Lax-Wendroff algorithm to the leapfrog algorithm in the case of Burgers' wave
'''
#setup
import numpy as np

# %%
# parameters

eps = 1.
Delta_x = 0.1 # spatial step
Delta_t = 0.1 # time step
beta = eps*Delta_t/Delta_x

L_x = 2*np.pi # spatial domain size
T_f = 2 # period of simulation

N_x = round(L_x/Delta_x) # number of space partitions
N_t = round(T_f/Delta_t) # number of time partitions

# initial conditions

u_t0 = lambda x: np.sin(x) # u(x,t=0)=sin(x)
u_x0 = 0 # u(x=0,t) = 0
u_xf = 0 # u(x=L_x,t) = 0

# initialize arrays

xt = np.zeros([N_x,N_t]) # so time progresses from left to right and space from top to bottom in the array
xt[:,0] = u_t0(np.linspace(0,L_x,N_x)) # initial condition


# %%
# define function for Lax-Wendroff method

def Lax_Wendroff(M : np.ndarray):
  S = np.copy(M) # we will mutate S so we copy so that M is not destroyed
  n, m = S.shape #dimensions of the array; n is number of x sites and m is number of t sites
  for j in range(0,m-1):
    for i in range(0,n-1):
      S[i,j+1] = S[i,j] - beta*( S[i+1,j]**2 - S[i-1,j]**2 )/4 + beta**2 / 4 * ( ( S[i,j] + S[i+1,j] )*( S[i+1,j]**2 - S[i,j]**2 ) + ( S[i,j] + S[i-1,j] )*( S[i-1,j]**2 - S[i,j]**2 ) )
  return S

# %%
#Run the function on our array

Solution_xt = Lax_Wendroff(xt)

# %%
import matplotlib.pyplot as plt

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
ani.save("./wave.gif", writer="pillow", fps=5) # save the animation

# %%
# now show the graphs required for the report

plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,10])
plt.title("Burgers' Wave (t = {0:.3f})".format(10*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()

plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,30])
plt.title("Burgers' Wave (t = {0:.3f})".format(30*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()

plt.plot(np.linspace(0,L_x,N_x),Solution_xt[:,20])
plt.title("Burgers' Wave (t = {0:.3f})".format(20*Delta_t))
plt.ylabel("u(x,t)")
plt.xlabel("x")
plt.show()





