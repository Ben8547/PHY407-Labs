# %% [markdown]
# Author: Ben Campbell
# 
# Purpose: To implement a monte-carlo simulation of an ideal gas

# %%
# Setup
from numpy import histogram, sum, array
import matplotlib.pyplot as plt
from mcsim import ideal_gas_sim

# %%
# setup paramters
T = 10.0 # k_b * T (in units where h_bar = 1)
L = 10.0 # length of box (in units where h_bar = 1)
N = 1000 # number of molecules
steps = 10000000 # number of simulation steps

# %%
# run simulation
eplot, n = ideal_gas_sim(T,L,N,steps,guess_n_init=False)

# %%
#plot
plt.plot(eplot)
plt.ylabel("Energy")
plt.ticklabel_format(style="plain")
plt.xlabel("Simulation Iteration")
plt.title("Evolution of the system energy in a\n"+r"Monte-Carlo Markov chain simulation ($\frac{1}{\beta}=$"+f"{T:.3f})")
plt.show()

# %%
# Code in this block by NG
# This calculates the energy of each particle, neglecting constant factors
energy_n = n[:, 0]**2 + n[:, 1]**2 + n[:, 2]**2
# This calculates the frequency distribution and creates a plot
#plt.figure(2)
#plt.clf()
hist_output = histogram(energy_n, 50)
# This is the frequency distribution
energy_frequency = hist_output[0]
# This is what the x-axis of the plot should look like
# if we plot the energy distribution as a function of n
# the 2nd axis of hist_output contains the boundaries of the array.
# Instead, we want their central value, below.
energy_vals = 0.5*(hist_output[1][:-1] + hist_output[1][1:]) # average of the two edges - center point of the bin
n_vals = energy_vals**0.5
# Create the desired plot
plt.figure(3)
plt.clf()
plt.bar(n_vals, energy_frequency/N, width=0.1)
plt.title("Frequency of molecular energy as a function of $n$")
plt.xlabel("Energy")
plt.ylabel("Frequency of occurence")
plt.show()

# %%
# Average value of n:
'''from sympy import init_printing
init_printing()'''
def n_bar_approx(energy_frequency_,n_vals_): # arguements from a histrogram
    n_bar_list_top = [energy_frequency_[i]*n_vals_[i] for i in range(50)]
    n_bar = sum(n_bar_list_top)/sum(energy_frequency_)
    return n_bar
print(r"$\bar{n}\approx$"+f"{n_bar_approx(energy_frequency,n_vals)}")

# %%
#average energy computation

def E_bar_approx(E_vals_,N_): return sum(E_vals_)/N_

print(r"$\bar{E}\approx$"+f"{E_bar_approx(array(eplot),steps)}")

# %%
# now repeat for several value of k_b*T

recip_beta = [10,40,100,400,1200,1600]
steps_list = [round(1e7), round(1e7), round(3e7),round(4e7),round(5e7),round(6e7)]

# %%
average_quantum_num = {} # initialize dictionaries
average_total_energy = {}

for i in range(len(recip_beta)):
    T = recip_beta[i] # k_b * T (in units where h_bar = 1)
    L = 10.0 # length of box (in units where h_bar = 1)
    N = 1000 # number of molecules
    steps = steps_list[i] # number of simulation steps; modified to be more for larger temps
    
    eplot, n = ideal_gas_sim(T,L,N,steps)
    plt.plot(eplot)
    plt.ylabel("Energy")
    plt.ticklabel_format(style="plain")
    plt.xlabel("Simulation Iteration")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title("Evolution of the system energy in a\n"+r"Monte-Carlo Markov chain simulation ($\frac{1}{\beta}=$"+f"{T:.1f})")
    plt.show()

    energy_n = n[:, 0]**2 + n[:, 1]**2 + n[:, 2]**2
    hist_output = histogram(energy_n, 50)
    energy_frequency = hist_output[0]
    energy_vals = 0.5*(hist_output[1][:-1] + hist_output[1][1:]) # average of the two edges - center point of the bin
    n_vals = energy_vals**0.5
    # Create the desired plot
    plt.bar(n_vals, energy_frequency/N, width=0.1)
    plt.title("Frequency of molecular energy as a function of $n$"+"\n"+r"$\frac{1}{\beta}=$"+f"{T:.1f}")
    plt.xlabel("Energy")
    plt.ylabel("Frequency of occurence")
    plt.show()
    #print(r"$\bar{n}\approx$"+f"{n_bar_approx(energy_frequency,n_vals)}")
    average_quantum_num[recip_beta[i]] = n_bar_approx(energy_frequency,n_vals)
    average_total_energy[recip_beta[i]] = E_bar_approx(array(eplot)[steps//2::],steps//2)

# %%
# now make plots accorss all simulations

energies = [average_total_energy[recip_beta[i]] for i in range(len(recip_beta))]
n_bars = [average_quantum_num[recip_beta[i]] for i in range(len(recip_beta))]

plt.plot(recip_beta,energies,ls='--',marker='o')
plt.title(r"Average simulation energy as a function of $\frac{1}{\beta}$")
plt.xlabel(r"$\frac{1}{\beta}$")
plt.ylabel(r"Energy")
plt.show()

plt.plot(recip_beta,n_bars,ls='--',marker='o')
plt.title(r"Average simulation quantum number as a function of $\frac{1}{\beta}$")
plt.xlabel(r"$\frac{1}{\beta}$")
plt.ylabel(r"$\bar{n}$")
plt.show()

# %%
#compute heat capacity

def fit(x,a,b): return a*x+b

from scipy.optimize import curve_fit

popt,pcov = curve_fit(fit,recip_beta,energies)

# %%
heat_cap = popt[0]
print("heat cap: ",heat_cap)

# %%



