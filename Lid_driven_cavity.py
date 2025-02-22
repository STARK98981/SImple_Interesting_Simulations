# This code simulates the lid driven cavity case with Chorin's projection method using Navier-Stokes equation

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cmasher as cmr

n_points = 41
domain_size = 1
element_length = domain_size/(n_points-1)
n_iterations = 500
dt = 0.001
nu = 0.1
rho = 1.0
u_velocity_top = 1.0
n_iterations_poisson = 50


x = np.linspace(0.0,domain_size,n_points)
y = np.linspace(0.0,domain_size,n_points)

X,Y = np.meshgrid(x,y)

u_prev = np.zeros_like(X)
v_prev = np.zeros_like(X)
press_prev = np.zeros_like(X)

def central_diff_x(field):
    diff = np.zeros_like(field)
    diff[1:-1,1:-1] = ( field[1:-1,2:] - field[1:-1,0:-2] ) / (2 * element_length)
    return diff

def central_diff_y(field):
    diff = np.zeros_like(field)
    diff[1:-1,1:-1] = ( field[2:,1:-1] - field[0:-2,1:-1] ) / (2 * element_length)
    return diff

def laplace (field):
    diff = np.zeros_like(field)
    diff[1:-1,1:-1] = (field[1:-1,2:] + field[1:-1,0:-2] + field[2:,1:-1] + field[0:-2,1:-1] - 4*field[1:-1,1:-1]) / (element_length**2)
    return diff

u_tent = np.zeros_like(u_prev)
v_tent = np.zeros_like(v_prev)

for _ in tqdm(range(n_iterations)):  # Use tqdm directly
    
    # Step 1: Solve momentum equation without pressure gradient
    u_tent = u_prev + dt*(nu*laplace(u_prev) - u_prev*central_diff_x(u_prev) - v_prev*central_diff_y(u_prev) )

    v_tent = v_prev + dt*(nu*laplace(v_prev) - u_prev*central_diff_x(v_prev) - v_prev*central_diff_y(v_prev))

    # Enforce BC for velocities: No slip at all domains except at top surface

    u_tent[-1,:] = u_velocity_top
    u_tent[0,:] = 0.0
    u_tent[:,0] = 0.0
    u_tent[:,-1] = 0.0
    v_tent[-1,:] = 0.0
    v_tent[0,:] = 0.0
    v_tent[:,0] = 0.0
    v_tent[:,-1] = 0.0

    # Step 2: Compute divergence of velocities

    divergence_velocity = central_diff_x(u_tent) + central_diff_y(v_tent)

    # Step 3: Solve pressure poisson equation
    # Computing rhs of pressure poisson equation...

    RHS = rho/dt * divergence_velocity

    # Setup iterative procedure for pressure poisson

    for _ in range(n_iterations_poisson):
        press_next = np.zeros_like(press_prev)
        press_next[1:-1,1:-1] = 0.25 * (
            press_prev[1:-1,2:] + press_prev[1:-1,0:-2] + 
            press_prev[2:,1:-1] + press_prev[0:-2,1:-1] -
            element_length**2 * RHS[1:-1,1:-1]
        )

        # Enfore pressure BC over the domain

        press_next[:,-1] = press_next[:,-2]
        press_next[:,0] = press_next[:,1]
        press_next[-1,:] = press_next[-2,:]
        press_next[0,:] = press_next[1,:]

        press_prev = press_next

    # Step 4: Velocity corrections:

    u_next = u_tent - dt/rho * central_diff_x(press_next)
    v_next = v_tent - dt/rho * central_diff_y(press_next)

    # Enfore BC for velocities again
    u_next[-1,:] = u_velocity_top
    u_next[0,:] = 0.0
    u_next[:,0] = 0.0
    u_next[:,-1] = 0.0
    v_next[-1,:] = 0.0
    v_next[0,:] = 0.0
    v_next[:,0] = 0.0
    v_next[:,-1] = 0.0

    # Advance in next time step

    u_prev = u_next
    v_prev = v_next
    press_prev = press_next

plt.figure()
plt.contourf(X,Y, press_next,cmap=cmr.amber_r, levels = 100)
plt.colorbar()
plt.quiver(X,Y,u_next,v_next,color = "black")
#plt.streamplot(X,Y,u_next,v_next,color="black")
plt.show()
