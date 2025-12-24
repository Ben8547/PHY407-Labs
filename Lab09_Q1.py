# %% [markdown]
# Author: Ben Campbell
# 
# Purpose: To investigate fourier analysis as amethod for solving PDEs numerically, specifically the wave equation.

# %%
# setup
import numpy as np
from dcst import dst, idst
import matplotlib.pyplot as plt

# %%
# parameters

N = round(1e6) # number of Fourier coeficients

L = 1 #m
d = 0.1#m
nu = 100 #m/s
C = 1 #m/s
sigma = .3 #m

omega_k = np.array([k*np.pi / L * nu for k in range(1,N+1)])

#print(omega_k)


# %%
# initial conditions

x = np.linspace(0,L,N) # resolution of the spactial domain

#print(len(x)) # sanity check

phi_0 = np.zeros(N) # initial position

psi_0 = C*(L-x)*x/L**2 * np.exp(-(x-d)**2 / (2*sigma**2)) # initial velocity

#plt.plot(psi_0) #Sanity check

# %%
# Sine transform the initial conditions

position_fc = dst(phi_0) # positional Fourier coeficients

#print(position_fc)

velocity_fc = dst(psi_0) # velocity Fourier coeficients

#print(len(position_fc)) # sanity check

'''from numpy.fft import fftfreq
print(fftfreq(N,x[1]-x[0]))
print(np.array([k for k in range(1,N+1)])/L/nu )'''

# %%
# compute the coeficients of the Ansatz:

def Ansatz_fc(t): # function of the time
    # time must be a scalar
    return position_fc*np.cos(omega_k*t) + velocity_fc*np.sin(omega_k*t)/omega_k


Coefs_for_plots = {"%.3f"%t : Ansatz_fc(t) for t in [0.002, 0.004, 0.006, 0.012, 0.1]}

h = 0.001 # animation time-step
q = np.arange(0,0.25,h)
Coefs_for_anims = {"%.3f"%t : Ansatz_fc(t) for t in q}
# do a bunch here so that we can make an animation

#print(Coefs_for_plots) # sanity check

# %%
# reconstruct the wave function

sample_points_for_plots = {"%.3f"%t : idst(Coefs_for_plots["%.3f"%t]) for t in [0.002, 0.004, 0.006, 0.012, 0.1]}

sample_points_for_anims = {"%.3f"%t : idst(Coefs_for_anims["%.3f"%t]) for t in q}

#print(sample_points_for_plots) # sanity check

# %%
# plot for t = 0.002

curr_time = 0.002
plt.plot(x,sample_points_for_plots['%.3f'%curr_time])
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%
# plot for t = 0.004

curr_time = 0.004
plt.plot(x,sample_points_for_plots['%.3f'%curr_time])
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%
# plot for t = 0.006

curr_time = 0.006
plt.plot(x,sample_points_for_plots['%.3f'%curr_time])
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%
# plot for t = 0.012

curr_time = 0.012
plt.plot(x,sample_points_for_plots['%.3f'%curr_time])
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%
# plot for t = 0.1

curr_time = 0.1
plt.plot(x,sample_points_for_plots['%.3f'%curr_time])
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%
# find appropriate bounds for the graph; this takes a very long time to run (~1 min)
"""all_amps = []
for l in sample_points_for_anims.keys():
    all_amps.extend(sample_points_for_anims[l])
LB = np.min(all_amps)
UB = np.max(all_amps)"""

LB = -0.0004
UB = 0.0004

# %%
# animate

from matplotlib.animation import FuncAnimation

# inital plot
fig, ax = plt.subplots()
p, = plt.plot(x,sample_points_for_anims['0.000'])
fig.suptitle("Evolution of the wave at 0.000 seconds")
plt.ylabel("Amplitude (meters)")
plt.ylim(-max(LB,UB), max(UB,LB))
plt.xlabel("Distance along string (meters)")

#update function

def update(frame):
    p.set_ydata(sample_points_for_anims['%.3f'%(h*frame)])
    fig.suptitle(f"Evolution of the wave at {h*frame:.3f} seconds")
    return p,

#produce animation

ani = FuncAnimation(fig,update,frames=len(q))


# %%
ani.save("./wave_on_string.gif",writer="pillow",fps=10)

# %%
#sandbox
curr_time = 45
plt.plot(x,idst(Ansatz_fc(curr_time)))
plt.title("Evolution of the wave at %.3f seconds"%curr_time)
plt.ylabel("Amplitude (meters)")
plt.xlabel("Distance along string (meters)")
plt.show()

# %%



