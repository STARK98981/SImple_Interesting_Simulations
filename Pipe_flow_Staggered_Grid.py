# This code produces the Hagen_Poiseuille profile (parabolic profile) for velocity based on a "Staggered Grid Approach"
# Navier-Stokes Equation (incompressible flow)  is solved using FDM (Central difference for spatial discretization
# and Forward Difference for time discretization)

# This code is a follow through of the channel ("https://www.youtube.com/@MachineLearningSimulation")

# IMPORT ALL THE DEPENDENCIES

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

# DEFINE THE CONSTANTS
n_points_y = 15
aspect_ratio = 10
nu = 0.01
dt = 0.001
n_time_steps = 5000
plot_every = 50
n_poisson_pressure_iterations = 50

cell_length = 1.0/ (n_points_y -1)

n_points_x = (n_points_y-1) * aspect_ratio + 1

x = np. linspace(0.0,1.0*aspect_ratio,n_points_x)
y = np. linspace(0.0,1.0,n_points_y)

X,Y = np.meshgrid(x,y)

# INITIAL CONDITION

velocity_x_prev = np.ones((n_points_y+1,n_points_x))
# (outside + inside)/2 = 0...to prescribe 0 x-velocties on boundary at top and bottom face
# outside = - inside

velocity_x_prev[0,:] = -velocity_x_prev[1,:] # @ top boundary
velocity_x_prev[-1,:] = -velocity_x_prev[-2,:] # @ bottom boundary

velocity_y_prev = np.ones((n_points_y,n_points_x+1))

pressure_prev = np.ones((n_points_y+1,n_points_x+1))

# Pre-Allocation of some arrays

velocity_x_tent = np.zeros_like(velocity_x_prev)
velocity_x_next = np.zeros_like(velocity_x_prev)

velocity_y_tent = np.zeros_like(velocity_y_prev)
velocity_y_next = np.zeros_like(velocity_y_prev)

plt.style.use("dark_background")
plt.figure(figsize=(1.5*aspect_ratio,6))

for iter in tqdm(range(n_time_steps)):
    
    ### Step 1 in algorithm: To predict the velocities according to NVS 
    ##Calculating for u velocity:
    diffusion_x = nu * (
                velocity_x_prev[1:-1, 2:] + velocity_x_prev[1:-1, :-2] + velocity_x_prev[2:, 1:-1] + velocity_x_prev[:-2,1:-1] 
                -4 * velocity_x_prev[1:-1,1:-1]) / (cell_length**2)
    convection_x = ( (velocity_x_prev[1:-1, 2:]**2 - velocity_x_prev[1:-1, :-2]**2) / (2*cell_length))\
                    + (velocity_x_prev[2:,1:-1] - velocity_x_prev[:-2,1:-1])/(2*cell_length)\
                    *0.25 * (velocity_y_prev[1:,1:-2] + velocity_y_prev[1:,2:-1] + velocity_y_prev[:-1,1:-2] + velocity_y_prev[:-1,2:-1])

    pressure_gradient_x = (pressure_prev[1:-1,2:-1] - pressure_prev[1:-1,1:-2])/cell_length

    velocity_x_tent[1:-1,1:-1] = velocity_x_prev[1:-1,1:-1] + dt * (-pressure_gradient_x + diffusion_x - convection_x)

    # Apply BC
    velocity_x_tent[1:-1,0] = 1.0 #inlet wall
    velocity_x_tent[1:-1,-1] = velocity_x_tent[1:-1,-2] # Outlet wall, gradient is 0
    velocity_x_tent[0,:] = - velocity_x_tent[1,:] # Bottom_wall (velocity is 0-->  dirichlet)
    velocity_x_tent[-1,:] = - velocity_x_tent[-2,:] # Top_wall (velocity is 0-->  dirichlet)

    ##Calculating for v velocity:

    diffusion_y = nu * (
                velocity_y_prev[1:-1, 2:] + velocity_y_prev[1:-1, :-2] + velocity_y_prev[2:, 1:-1] + velocity_y_prev[:-2,1:-1] 
                -4 * velocity_y_prev[1:-1,1:-1]) / (cell_length**2)
    
    convection_y = ( (velocity_y_prev[1:-1, 2:] - velocity_y_prev[1:-1, :-2]) / (2*cell_length))\
                    + (velocity_y_prev[2:,1:-1]**2 - velocity_y_prev[:-2,1:-1]**2)/(2*cell_length)\
                    *0.25 * (velocity_x_prev[2:-1,1:] + velocity_x_prev[2:-1,:-1] + velocity_x_prev[1:-2,1:] + velocity_x_prev[1:-2,:-1])

    pressure_gradient_y = (pressure_prev[2:-1,1:-1] - pressure_prev[1:-2,1:-1])/cell_length

    velocity_y_tent[1:-1,1:-1] = velocity_y_prev[1:-1,1:-1] + dt * (-pressure_gradient_y + diffusion_y - convection_y)

    # Apply BC
    velocity_y_tent[1:-1,0] = -velocity_y_tent[1:-1,1] #inlet wall
    velocity_y_tent[1:-1,-1] = velocity_y_tent[1:-1,-2] # Outlet wall, gradient is 0
    velocity_y_tent[0,:] = 0.0 # Bottom_wall (velocity is 0-->  dirichlet)
    velocity_y_tent[-1,:] = 0.0 # Top_wall (velocity is 0-->  dirichlet)

    ### Step-2: Check divergence of velocities to be 0-->Incompressible liquid

    divergence = (velocity_x_tent[1:-1,1:] - velocity_x_tent[1:-1,:-1]) / cell_length \
                + ( velocity_y_tent[1:,1:-1] - velocity_y_tent[:-1,1:-1])/ cell_length
    
    ### Step -3 RHS of poisson pressure

    pressure_poisson_rhs = divergence / dt

    ## Solve the pressure correction poisson problem (jacobi iteration)

    pressure_correction_prev = np.zeros_like(pressure_prev)

    for _ in range(n_poisson_pressure_iterations):
        pressure_correction_next = np.zeros_like(pressure_correction_prev)
        pressure_correction_next[1:-1,1:-1] = 0.25* (pressure_correction_prev[1:-1,2:] + pressure_correction_prev[2:,1:-1]\
                                                + pressure_correction_prev[1:-1,:-2] + pressure_correction_prev[:-2,1:-1]\
                                                -cell_length**2 * pressure_poisson_rhs)

        # Apply pressure BC: Homogeneous Neumann everywhere except ehere it is homogeneous dirichilet
        pressure_correction_next[1:-1,0] = pressure_correction_next[1:-1,1] #Inlet wall
        pressure_correction_next[1:-1,-1] = -pressure_correction_next[1:-1,-2] #Top wall 
        pressure_correction_next[0,:] = pressure_correction_next[1,:] # bottom wall
        pressure_correction_next[-1,:] = pressure_correction_next[-2,:] #Right wall

        # Advance in smoothing 
        pressure_correction_prev = pressure_correction_next

    ### Step-4: Update the pressure

    pressure_next = pressure_prev + pressure_correction_next

    ### Step-5: Correct the velocities

    pressure_correction_gradient_x = (pressure_correction_next[1:-1,2:-1] - pressure_correction_next[1:-1,1:-2]) / cell_length
    velocity_x_next[1:-1,1:-1] = velocity_x_tent[1:-1,1:-1] - dt * pressure_correction_gradient_x

    pressure_correction_gradient_y = (pressure_correction_next[2:-1,1:-1] - pressure_correction_next[1:-2,1:-1]) / cell_length
    velocity_y_next[1:-1,1:-1] = velocity_y_tent[1:-1,1:-1] - dt * pressure_correction_gradient_y

    # Enforce BC again

    velocity_y_next[1:-1,0] = -velocity_y_next[1:-1,1] #inlet wall
    velocity_y_next[1:-1,-1] = velocity_y_next[1:-1,-2] # Outlet wall, gradient is 0
    velocity_y_next[0,:] = 0.0 # Bottom_wall (velocity is 0-->  dirichlet)
    velocity_y_next[-1,:] = 0.0 # Top_wall (velocity is 0-->  dirichlet)

    velocity_x_next[1:-1,0] = 1.0 #inlet wall
    inflow_mass_rate_next = np.sum(velocity_x_next[1:-1,0])
    outflow_mass_rate_next = np.sum(velocity_x_next[1:-1,-2])

    velocity_x_next[1:-1,-1] = velocity_x_next[1:-1,-2] * inflow_mass_rate_next / outflow_mass_rate_next # Outlet wall, gradient is 0
    velocity_x_next[0,:] = - velocity_x_next[1,:] # Bottom_wall (velocity is 0-->  dirichlet)
    velocity_x_next[-1,:] = - velocity_x_next[-2,:] # Top_wall (velocity is 0-->  dirichlet)

    # Advance in time

    velocity_x_prev = velocity_x_next
    velocity_y_prev = velocity_y_next
    pressure_prev = pressure_next

    # Visulatization commands
    if iter % plot_every ==0:

        velocity_x_vertex_centred = (velocity_x_next[1:,:] + velocity_x_next[:-1,:] ) / 2
        velocity_y_vertex_centred = (velocity_y_next[:,1:] + velocity_y_next[:,:-1] ) / 2

        plt.contourf(X,Y, velocity_x_vertex_centred,levels = 50,cmap = cmr.amber,vmin =0.0,vmax =1.6) 
        plt.colorbar()

        
        plt.quiver(X[:,::6], Y[:,::6], velocity_x_vertex_centred[:,::6],velocity_y_vertex_centred[:,::6],alpha = 0.4)
        plt.plot(20*cell_length + velocity_x_vertex_centred[:,20], Y[:,20], color = "black", linewidth = 3)
        plt.plot(60*cell_length + velocity_x_vertex_centred[:,60], Y[:,60], color = "black", linewidth = 3)
        plt.plot(80*cell_length + velocity_x_vertex_centred[:,80], Y[:,80], color = "black", linewidth = 3)

        plt.draw()
        plt.pause(0.05)
        plt.clf()
plt.show()
        