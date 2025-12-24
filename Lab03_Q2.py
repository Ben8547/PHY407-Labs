# %%
#imports
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setting up the integration
def gaussxw(N):#from Newman's Website

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3,4*N-1,N)/(4*N+2)
    x = np.cos(np.pi*a+1/(8*N*N*np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = np.ones(N,float)
        p1 = np.copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = np.max(np.abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gauss_quad(function,a,b,N): #perform the integration (function must be able to take an array)
  x,w = gaussxw(N)
  w *= 0.5*(b-a)
  x = 0.5*(b-a)*x+0.5*(b+a)
  return np.sum(w*function(x))

# %%
#test the integration
x,w = gaussxw(10)
print(2.-gauss_quad(np.sin,0,np.pi,10))
print(np.pi-gauss_quad(lambda x: 4/(1+x**2),0,1,25))
machine_error = np.finfo(float).eps
print("machine precision: ",machine_error)
#perfect - on the order of machine error; seem to be exactly 4 times the machine prescision

# %%
#We will be performing many integrations at N=8 and N=16 so we can store the points and weights as to aboid superflous calculation
x8,w8 = gaussxw(8)
x16,w16 = gaussxw(16)
x100,w100 = gaussxw(100)

def int_8(function,a,b):
  w = 0.5*(b-a)*w8
  x = 0.5*(b-a)*x8+0.5*(b+a)
  return np.sum(w*function(x))

def int_16(function,a,b):
  w = 0.5*(b-a)*w16
  x = 0.5*(b-a)*x16+0.5*(b+a)
  return np.sum(w*function(x))

def int_100(function,a,b):# for checking purposes
  w = 0.5*(b-a)*w100
  x = 0.5*(b-a)*x100+0.5*(b+a)
  return np.sum(w*function(x))


# %%
#set parameters
m = 1 #kg
k = 12 #N/m
from scipy.constants import c

# %%
# define the reletivistic g(x) according to the physics background section:
def g_rel(x,x_0):
  return c * np.sqrt(np.abs(((k*(x_0**2-x**2)*(2*m*c**2 + k*(x_0**2-x**2)/2)) / (2*(m*c**2 + k*(x_0**2-x**2)/2)**2))))#absolute value needed because sometimes python seems to think that small numbers are negative

# %%
#define the integrand of the period:
def T_integrand(x,x_0):
  return 4/(g_rel(x,x_0))
#print(T_integrand(0,0.01)) #sanity check

# %%
# the computation for x_0 = 1 cm
x_0_1cm = 0.01 #m
T_1cm = lambda x: T_integrand(x,x_0_1cm)
Period_1cm_8 = int_8(T_1cm,0,x_0_1cm)
Period_1cm_16 = int_16(T_1cm,0,x_0_1cm)
Period_1cm_100 = int_100(T_1cm,0,x_0_1cm)
#the analytical value should be close to 2\pi\sqrt{m/k}
print("Period with 8 intervals: {0:.5f}".format(Period_1cm_8))
print("Period with 16 intervals: %.5f"%Period_1cm_16)
print("Period with 100 intervals: %.5f"%Period_1cm_100)
classical_period = 2*np.pi*np.sqrt(m/k)
print("classical period: {0:.5f}".format(classical_period))
print("difference from classical limit (N=8): {0:.5f}".format(np.abs(Period_1cm_8-classical_period)))
print("difference from classical limit (N=16): {0:.5f}".format(np.abs(Period_1cm_16-classical_period)))
print("difference from classical limit (N=100): {0:.5f}".format(np.abs(Period_1cm_100-classical_period)))

# %%
# Error estimate
# Newman claim that I_2N is approximately correct, and is sufficient for error computation, but we may as well use I_100 instead since it will be more accurate especially given the asymptotic nature of the curve
error_8 = np.abs(Period_1cm_8-Period_1cm_100)/Period_1cm_100
error_16 = np.abs(Period_1cm_16-Period_1cm_100)/Period_1cm_100
print("relative error with 8 intervals: %.5e"%error_8) #gives about double the error if N=16 were used
print("relative error with 16 intervals: %.5e"%error_16)

# %%
#plot the integrand
domain = np.linspace(0,x_0_1cm,int(1e4))
plt.plot(domain, T_integrand(domain,x_0_1cm))
plt.title("The period integrand (densely sampled)")
plt.xlabel("x (m)")
plt.ylabel("4/g(x) (s/m)")
plt.show()

# %%
#Plot the integrand at the sameple points
plt.plot(0.5*(x_0_1cm)*x8+0.5*(x_0_1cm), T_integrand(0.5*(x_0_1cm)*x8+0.5*(x_0_1cm),x_0_1cm),label="N=8",marker='.',ls='--') # N=8
plt.plot(0.5*(x_0_1cm)*x16+0.5*(x_0_1cm), T_integrand(0.5*(x_0_1cm)*x16+0.5*(x_0_1cm),x_0_1cm),label="N=16",marker='.',ls=':') # N=16
#plt.plot(0.5*(x_0_1cm)*x100+0.5*(x_0_1cm), T_integrand(0.5*(x_0_1cm)*x100+0.5*(x_0_1cm),x_0_1cm),label="N=100",marker='.',ls=':') # N=16
plt.title("The period integrand")
plt.xlabel("x (m)")
plt.ylabel("4/g(x) (s/m)")
plt.legend()
plt.show()


# %%
#Plot the integrand time the weights at the sameple points
plt.plot(0.5*(x_0_1cm)*x8+0.5*(x_0_1cm), 0.5*(x_0_1cm)*w8 * T_integrand(0.5*(x_0_1cm)*x8+0.5*(x_0_1cm),x_0_1cm),label="N=8",marker='.',ls='--') # N=8
plt.plot(0.5*(x_0_1cm)*x16+0.5*(x_0_1cm), 0.5*(x_0_1cm)*w16 * T_integrand(0.5*(x_0_1cm)*x16+0.5*(x_0_1cm),x_0_1cm),label="N=16",marker='.',ls=':') # N=16
plt.plot(0.5*(x_0_1cm)*x100+0.5*(x_0_1cm), 0.5*(x_0_1cm)*w100 * T_integrand(0.5*(x_0_1cm)*x100+0.5*(x_0_1cm),x_0_1cm),label="N=100",marker='none',ls='-.') # N=16
plt.title("The weighted period integrand")
plt.xlabel("x (m)")
plt.ylabel(r"$\frac{4}{g(x_i)}\cdot \omega_i$ (s)")
plt.legend()
plt.show()

# %%
#Part c
#Define integration methods
x200,w200 = gaussxw(200)
x400,w400 = gaussxw(400)

def int_200(function,a,b):
  w = 0.5*(b-a)*w200
  x = 0.5*(b-a)*x200+0.5*(b+a)
  return np.sum(w*function(x))

def int_400(function,a,b):
  w = 0.5*(b-a)*w400
  x = 0.5*(b-a)*x400+0.5*(b+a)
  return np.sum(w*function(x))

# %%
# estimate the relative error:
Period_1cm_200 = int_200(T_1cm,0,x_0_1cm)
Period_1cm_400 = int_400(T_1cm,0,x_0_1cm)
print("Period with 200 intervals: %.5f"%Period_1cm_200)
print("Period with 400 intervals: %.5f"%Period_1cm_400)
rel_error_200 = np.abs(Period_1cm_200-Period_1cm_400)/Period_1cm_400
print("relative error with 200 intervals: %.5e"%rel_error_200)

# %%
# graph the periods as a function of x_0
x_c = c*np.sqrt(m/k) # in the classical limit, v = \sqrt{k/m}x_0 when x is at the origin. Then we solve for c = \sqrt{k/m}x_0 to find that x_0 = c\sqrt{m/k}
#range1 = np.linspace(0.001,10*x_c,int(1e4))
range1 = [0.9**n for n in range (200,1,-1)] + [1.1**n for n in range(int(np.ceil(np.log(10*x_c)/np.log(1.1))))] # range of x_0 at which to compute the period, want to sample a lot near 0 and less further away because the curve appraoches a line - should give a visably smooth curve

# %%
#get the periods
list_of_periods = []
for i in range1:
  T = lambda x: T_integrand(x,i)
  list_of_periods.append(int_200(T,0,i))

print(range1[157]) # this is about 0.01 so we can check our answers
print(list_of_periods[107])

# %%
#plot
plt.hlines(classical_period,range1[0],range1[-1],label="classical limit",ls='--',color='orange')
plt.plot(range1,list_of_periods,label = "Relativistic dependance of the period")
plt.ylabel("Period (s)")
#plt.rcParams['text.usetex'] = True # this doesn not seem to work
plt.title(r"Period as a function of $x_0$")
plt.xlabel(r"$x_0$")
#ax = plt.gca()
#x.set_xscale('log')
plt.legend()
plt.show()

# %%



