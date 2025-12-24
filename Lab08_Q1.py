# %% [markdown]
# Author: Ben Campbell
# 
# Purpose: To determine the heat distrobution inside of a conductor with static boundary conditions. We also evaluate the efficacy of the Gauss-Seidel method and the impact of overrelaxation.

# %%
# setup
from numpy import empty, zeros, sum
import matplotlib.pyplot as plt

count_num_frames_needed = 0 # will need for next part

grid_mesh = 0.1 # cm
grid_width = 20 # cm
grid_height = 8 # cm
M = int(round(grid_height/grid_mesh)) # rows
N = int(round(grid_width/grid_mesh)) # columns
target = 1e-6 # target accuracy

# create phi
def set_phi():
  # Create arrays to hold temperature values
  phi = zeros([M+2,N+2],float) # +2 adds an extra row an column on each side that can be used for the boundaries
  phi[M+1,:] = 10 # set boundaries; line GH
  phi[:,0] = [10*(i/(M+1)) for i in range(M+2)] # line HA - increases from 0 at A to 10 at H
  phi[:,N+1] = [10*(i/(M+1)) for i in range(M+2)] # line HA - increases from 0 at F to 10 at G
  phi[0,0:(N//4 + 1)] = [5*(i/(N//4 + 1)) for i in range(N//4 + 1)] # line AB: 0 at A, 5 at B
  phi[0,(3*N//4 + 1):(N+2)] = [5*(1 - i/(N//4 + 1)) for i in range(N//4 + 1)] # line EF: 0 at F, 5 at E
  phi[int(round(3/grid_mesh)),(N//4 + 1):(3*N//4 + 1)] = 7 # line CD
  phi[0:int(round(3/grid_mesh)),(N//4 + 1)] = [ 5+2*i/int(round(3/grid_mesh)) for i in range(int(round(3/grid_mesh)))] # line BC: 5 at B, 7 at C
  phi[0:int(round(3/grid_mesh))+1,(3*N//4 + 1)] = [ 5+2*i/int(round(3/grid_mesh)+1) for i in range(int(round(3/grid_mesh))+1)] # line DE: 5 at E, 7 at D
  return phi


# %%
# the following code is adapted from laplace.py (Newmann) - it perfoms overrelaxation Gauss-Spiegel integration

phi = set_phi()
omega = 0.5 # relaxation perameter

if True: # testing gate
  # Main loop
  delta = 1.0
  previous = 1e3 # some number that allos the first difference to be much larger that delta; we need to change how we check accuracy since we can't save the entire previous array with this method
  while delta>target:
      count_num_frames_needed += 1
      # Calculate new values of the potential
      for i in range(1,M+1):
        for j in range(1,N+1):
          if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
            phi[i,j] = phi[i,j]
          else:
            phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                    + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)

      # Calculate maximum difference from old values
      delta = abs(previous - (previous:=sum(phi)))/(M**2) # thus we can store a single number instead of an entire array


# %%
# Make a plot
plt.imshow(phi,origin='lower',cmap='bwr',extent=[0,20,0,8])
plt.ylabel("cm")
plt.xlabel("cm")
plt.title("Internal temperature of a conuductor\nwith specified boundary conditions")
c = plt.colorbar()
c.set_label(r"Temperature ($^\circ$C)")
plt.show()

print("number of steps required: {0}".format(count_num_frames_needed))

# %%
# Now we animate the output
import matplotlib.pyplot as plt
import matplotlib.animation as animation

phi = set_phi()


#simulation paramters:

omega = 0.5 # relaxation perameter

# initial plot
fig, ax = plt.subplots()
p = ax.imshow(phi, origin='lower', cmap='bwr', extent=[0,20,0,8])
c = fig.colorbar(p, ax=ax)
c.set_label(r"Temperature ($^\circ$C)")

ax.set_xlabel("cm")
ax.set_ylabel("cm")
ax.set_title("Animation of convergence of temperature\n"
             "using the overrelaxed Gauss-Seidel\n"
             r"algorithm ($\omega = 0.5$)")

# update function:

def update_plot(frame):
  # get the previous phi array
  phi = p.get_array() # extract array from the imshow object
  # Calculate new values of the potential
  for i in range(1,M+1):
    for j in range(1,N+1):
      if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
        phi[i,j] = phi[i,j]
      else:
        phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)
  # update the [plot]
  p.set_array(phi)
  return p

#animate

ani = animation.FuncAnimation(fig,update_plot,frames=count_num_frames_needed)

ani.save("./animation.gif", writer="pillow", fps=60) # save the animation; this seems to take > 10 minutes; skip this if you don't need to save the animation


# %%
# evaluate the impact of overrelaxation

phi = set_phi()

count_num_frames_needed = 0

omega = 0

while count_num_frames_needed < 100:
      count_num_frames_needed += 1
      # Calculate new values of the potential
      for i in range(1,M+1):
        for j in range(1,N+1):
          if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
            phi[i,j] = phi[i,j]
          else:
            phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                    + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)

      # Calculate maximum difference from old values
      delta = abs(previous - (previous:=sum(phi)))/(M**2) # thus we can store a single number instead of an entire array

# Make a plot
fig, ax = plt.subplots()
plt.title("computed internal temperature of\na conductor using the overrelaxed\n"+"Gauss-Seidel algorithm ($\omega = 0$)")
p = ax.imshow(phi, origin='lower', cmap='bwr', extent=[0,20,0,8])
c = fig.colorbar(p, ax=ax)
c.set_label(r"Temperature ($^\circ$C)")
plt.ylabel("cm")
plt.xlabel("cm")
plt.show()

# %%
# evaluate the impact of overrelaxation (continued)

phi = set_phi()

count_num_frames_needed = 0

omega = 0.9

while count_num_frames_needed < 100:
      count_num_frames_needed += 1
      # Calculate new values of the potential
      for i in range(1,M+1):
        for j in range(1,N+1):
          if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
            phi[i,j] = phi[i,j]
          else:
            phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                    + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)

      # Calculate maximum difference from old values
      delta = abs(previous - (previous:=sum(phi)))/(M**2) # thus we can store a single number instead of an entire array

# Make a plot
fig, ax = plt.subplots()
p = ax.imshow(phi, origin='lower', cmap='bwr', extent=[0,20,0,8])
c = fig.colorbar(p, ax=ax)
plt.title("computed internal temperature of a\nconductor using the overrelaxed\n"+rf"Gauss-Seidel algorithm ($\omega = 0.9$)")
c.set_label(r"Temperature ($^\circ C$)")
plt.ylabel("cm")
plt.xlabel("cm")
plt.show()

# %%
# animate ome
phi = set_phi()

#simulation paramters:

omega = 0.9 # relaxation perameter

# initial plot
fig, ax = plt.subplots()
p = ax.imshow(phi, origin='lower', cmap='bwr', extent=[0,20,0,8])
c = fig.colorbar(p, ax=ax)
c.set_label(r"Temperature ($^\circ C$)")
plt.ylabel("cm")
plt.xlabel("cm")
plt.title("Animation of convergence of\ntemperature using the overrelaxed\nGauss-Seidel algorithm ($\omega = 0.9$)")

# update function:

def update_plot(frame):
  # get the previous phi array
  phi = p.get_array() # extract array from the imshow object
  # Calculate new values of the potential
  for i in range(1,M+1):
    for j in range(1,N+1):
      if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
        phi[i,j] = phi[i,j]
      else:
        phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)
  # update the [plot]
  p.set_array(phi)
  return p

#animate

ani = animation.FuncAnimation(fig,update_plot,frames=count_num_frames_needed)

ani.save("./animation_09.gif", writer="pillow", fps=60)

# %%
#compute phi at (2.5 cm, 1 cm) 

phi = set_phi()
omega = 0.9

if True: # testing gate
  # Main loop
  delta = 1.0
  previous = 1e3 # some number that allos the first difference to be much larger that delta; we need to change how we check accuracy since we can't save the entire previous array with this method
  while delta>target:
      count_num_frames_needed += 1
      # Calculate new values of the potential
      for i in range(1,M+1):
        for j in range(1,N+1):
          if i <= int(round(3/grid_mesh)) and j <= (3*N//4 + 1) and j >= (N//4 + 1): # do not perturb the boundary conditions
            phi[i,j] = phi[i,j]
          else:
            phi[i,j] = (1+omega)*(phi[i+1,j] + phi[i-1,j] \
                                    + phi[i,j+1] + phi[i,j-1])/4 - omega*phi[i,j] # overrelaxed Laplace's equation (in this case: heat diffusion)

      # Calculate maximum difference from old values
      delta = abs(previous - (previous:=sum(phi)))/(M**2) # thus we can store a single number instead of an entire array


# %%
x = round(2.5/grid_mesh)
y = round(1/grid_mesh)
print("phi at (x,y): ", phi[x,y])


