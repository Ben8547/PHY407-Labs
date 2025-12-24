# -*- coding: utf-8 -*-
"""Lab05_Q2_a.ipynb

Purpose: To run a simulation of an object on a spring moving a relativistic speeds and to analyze the fundemental frequencies of the motion by means of a fourrier transform.

Author: Ben Campbell
"""

#!pip install warp-lang

import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
#import warp as wp
from scipy.constants import c
import sympy as s
s.init_printing()

#get expression for the acceleration of the relativistic spring; you can skip this part-it does not get called in the current setup; I didnt see at first that the acceleration was provided in the lab handout

x, t, x_0, c_l, k, m = s.symbols('x t x_0 c_l k m')
x1 = s.Function('x')(t)

mc2 = m*c_l**2
hkx2 = 0.5*k*(x_0**2 - x1**2)
tmp = hkx2*(2*mc2 + hkx2) / (mc2 + hkx2)**2
v = c_l*s.sqrt(tmp)

a = v.diff(t)
a = a.subs(x1.diff(t), v)
a = a.subs(x1,x)
a = a.simplify()

print(s.latex(a))
print(a.free_symbols)
a

# model the spring system with euler-cromer

mass = 1.0  # [kg] mass
k_s = 12.0  # [N/m] stiffness

xc = c*(mass/k_s)**0.5 # Corrected to use mass and k_s

def g(x0, x):
    """ Reletivistic Spring Velocity function """
    mc2 = mass*c**2
    hkx2 = 0.5*k_s*(x0**2 - x**2)
    tmp = hkx2*(2*mc2 + hkx2) / (mc2 + hkx2)**2
    return c*tmp**0.5

acc = s.lambdify(args=[x, x_0, m, k, c_l], expr=a, modules='numpy')

acc1 = lambda x,x_0: acc(x,x_0,mass,k_s,c)

def acc2(x,v):
  return -k_s/mass * x * (1- ((v**2)/(c**2)))**(1.5)

def spring_ODE(x_0,v_x_0,N,Dt):

  t = np.linspace(0,N*Dt,N) # set the times

  x = 0*t # make arrays for the system parameters
  y = 0*t
  vx = 0*t
  vy = 0*t

  x[0] = x_0  # [m]
  vx[0] = v_x_0 # [m]

  for i in range(N-1):

      # Update velocity first:
      # Pass numerical values for m, k, and c to acc
      vx[i+1] = vx[i] + Dt*acc2(x[i],vx[i]) #acc1(x[i],x_0) # seems to produce the same graphs using acc1 or acc2 despite one taking x_0 and the other not

      # Update positions with the new velocities
      x[i+1] = x[i] + Dt*vx[i+1]

  return x, vx, t

# test in the classical limit
x, v, t = spring_ODE(0.01,0,1000,0.001)

plt.plot(t,x)
plt.title("Test Euler-Cromer in Classical Limit")
plt.show()

#end test

# Generating the curves for several initial conditions

#first x_0 = 1m

x_1, v_1, t = spring_ODE(1,0,20000,0.001)

plt.plot(t,x_1)
plt.grid()
plt.title(r"The trajectory of a relativistc mass on a spring for $x_0=1$ m")
plt.ylabel(r"$x$ [m]")
plt.xlabel(r"$t$ [s]")
plt.show()

# second: x_0 = x_c

x_2, v_2, t = spring_ODE(xc,0,20000,0.001)

plt.plot(t,x_2)
plt.grid()
plt.title(r"The trajectory of a relativistc mass on a spring for $x_0=x_c$ m")
plt.ylabel(r"$x$ [m]")
plt.xlabel(r"$t$ [s]")
plt.show()

# third: x_0 = 10*x_c

x_3, v_3, t = spring_ODE(10*xc,0,400000,0.0001)

plt.plot(t,x_3)
plt.grid()
plt.title(r"The trajectory of a relativistc mass on a spring for $x_0=10 x_c$ m")
plt.ylabel(r"$x$ [m]")
plt.xlabel(r"$t$ [s]")
plt.show()

# now get the fourier transform of the positions
x_hat_1 = rfft(x_1)
x_hat_2 = rfft(x_2)
x_hat_3 = rfft(x_3)
v_hat_1 = rfft(v_1)
v_hat_2 = rfft(v_2)
v_hat_3 = rfft(v_3)

# find period using gaussian quadrature
def gaussxw(N):
    """Newman's function for Gaussian quad from -1 to +1
    N is now the number of samplebegin{} points"""
    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3, 4*N-1, N)/(4*N+2)
    x = np.cos(np.pi*a + 1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x, w


def gaussxwab(N, a, b):
    """Newman's function for Gaussian quad from a to b
    N is now the number of sample points"""
    x, w = gaussxw(N)
    return 0.5*(b-a)*x + 0.5*(b+a), 0.5*(b-a)*w

def gauss_int(a, b, N, f):
    """ This function implements the Guassian quadrature integration from a to
    b in a similar way as the trapz and Simpson's rules above
    IN:
    a, b: [floats] lower and upper bounds of integration
    N: [int] number of sample points (NOT slices!)
    f: [function] a function to integrate
    new [bool]: True if we need to compute x, w, False otherwise
    OUT:
    [float] the integral"""
    x, w = gaussxwab(N, a, b)
    result = 0.
    for i in range(N):
        result += w[i]*f(x[i])
    return result

def T_gauss(x0, x1, w1):
    """ Compute the integral of 1/g from 0 to x0
    k=stiffness, m=mass
    x1 and w1 are the locations and weights for a quadrature over [-1, +1] """
    xp = 0.5*x0*(x1 + 1)  # Newman's eqn (5.61)
    wp = 0.5*x0*w1  # Newman's eqn (5.62)
    gper = g(x0, xp)
    T = 0.  # future period of the oscillator; will accumulate
    for ii in range(len(x1)):  # len(x1)=N
        T += wp[ii]/gper[ii]
    T *= 4
    return T, gper, wp, xp

N=400
x, w = gaussxw(N)
T_1, gk_1, wk_1, xk_1 = T_gauss(x_1[0], x, w)
T_2, gk_2, wk_2, xk_2 = T_gauss(x_2[0], x, w)
T_3, gk_3, wk_3, xk_3 = T_gauss(x_3[0], x, w)

# get the frequency that each coeficient corresponds to
frequencies_1 = np.fft.fftfreq(len(x_hat_1), d=0.001)
frequencies_2 = np.fft.fftfreq(len(x_hat_2), d=0.001)
frequencies_3 = np.fft.fftfreq(len(x_hat_3), d=0.0001)

#print(np.sort(frequencies_1))

#plot
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,6),sharex=True)
plt.style.use('fast')
ax1.plot(frequencies_1[frequencies_1 >= 0]/2.,np.abs(x_hat_1[frequencies_1 >= 0])/np.max(np.abs(x_hat_1[frequencies_1 >= 0])),label=r"DFT for $x_0=1$ m",marker='.', color = 'cyan')
ax2.plot(frequencies_2[frequencies_2 >= 0]/2,np.abs(x_hat_2[frequencies_2 >= 0])/np.max(np.abs(x_hat_2[frequencies_2 >= 0])), label=r"DFT for $x_0=x_c$",marker='.', color='r')
ax3.plot(frequencies_3[frequencies_3 >= 0]/2,np.abs(x_hat_3[frequencies_3 >= 0])/np.max(np.abs(x_hat_3[frequencies_3 >= 0])), label=r"DFT for $x_0=10x_c$",marker='.',color='g')
ax2.vlines(1/T_2,0,1,color='b')
ax3.vlines(1/T_3,0,1,color='b')
ax1.vlines(1/T_1,0,1,label="Receiprical period of the system (Hz)",color='b')
ax3.set_xlabel("Frequency corresponding to the coeficient (Hz)")
ax2.set_ylabel(r"$\frac{|c_k|}{|c_{\text{max}}|}$")
plt.xlim(left=-0.1,right=5)
fig.suptitle("The normalized Fourier coeficients of the positions\nfor each tested initial condition")
fig.legend(loc="upper right")
plt.show()

#plot vecolcities
#plot
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,6),sharex=True)
plt.style.use('fast')
ax1.plot(frequencies_1[frequencies_1 >= 0]/2.,np.abs(v_hat_1[frequencies_1 >= 0])/np.max(np.abs(v_hat_1[frequencies_1 >= 0])),label=r"DFT for $x_0=1$ m",marker='.', color = 'cyan')
ax2.plot(frequencies_2[frequencies_2 >= 0]/2,np.abs(v_hat_2[frequencies_2 >= 0])/np.max(np.abs(v_hat_2[frequencies_2 >= 0])), label=r"DFT for $x_0=x_c$",marker='.', color='r')
ax3.plot(frequencies_3[frequencies_3 >= 0]/2,np.abs(v_hat_3[frequencies_3 >= 0])/np.max(np.abs(v_hat_3[frequencies_3 >= 0])), label=r"DFT for $x_0=10x_c$",marker='.',color='g')
ax2.vlines(1/T_2,0,1,color='b')
ax3.vlines(1/T_3,0,1,color='b')
ax1.vlines(1/T_1,0,1,label="Receiprical period of the system (Hz)",color='b')
ax3.set_xlabel("Frequency corresponding to the coeficient (Hz)")
ax2.set_ylabel(r"$\frac{|c_k|}{|c_{\text{max}}|}$")
plt.xlim(left=-0.1,right=5)
fig.suptitle("The normalized Fourier coeficients of the velocities\nfor each tested initial condition")
fig.legend(loc="upper right")
plt.show()

