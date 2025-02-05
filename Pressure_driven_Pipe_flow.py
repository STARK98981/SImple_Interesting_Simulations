# This code produces the Hagen_Poiseuille profile (parabolic profile) for velocity. In a 2D domain, BCs are fixed veloctites at top 
# and bottom boundaries with a constant pressure gradient in x-dir.
# Periodic BCs across west and east face of 2D domain is considered to simulate an 'infinitely long pipe'
# Navier-Stokes Equation (incompressible flow)  is solved using FDM (Central difference for spatial discretization
# and Forward Difference for time discretization)

# This code is a follow through of the channel ("https://www.youtube.com/@MachineLearningSimulation")

# IMPORTING ALL THE DEPENDENCIES
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# DEFINE THE REQUIRED CONSTANTS
n_points = 11 # number of points in the domain (same in x- and y-dir)
nu = 0.01 # Kinematic Viscosity
dt = 0.2 # time step size
n_time_steps = 100 # Total number of time steps

# DEFINE THE CONSTANT GRADIENT OF PRESSURE IN X-DIRECTION
grad_p = np.array([-1.0,0.0]) # Gradient of pressure in x and y-dir

# CALCULATE GRID SIZE AND NODE POINTS IN THE DOMAIN
n_grid_size = 1.0 / (n_points-1) # 1.0 is the domain size in x and y dir

x_nodes = np.linspace(0.0,1.0,n_points)
y_nodes = np.linspace(0.0,1.0,n_points)

# CREATE CO-ORDINATES FOR MESH GRID
co_ordinates_x, co_ordinates_y = np.meshgrid(x_nodes,y_nodes)

# FUNCTION FOR CENTRAL DIFFERENCING FOR A GRAIENT OPERATOR OF A FIELD WITH PERIODIC IN X-DIR
def CD_x_periodic(ux):
    diff = (np.roll(ux, shift=1, axis=1) - np.roll(ux, shift=-1, axis=1))/(2*n_grid_size)
    return diff

# FUNCTION FOR CENTRAL DIFFERENCING FOR LAPLACIAN OPERATIR OF A FIELD WITH PERIODIC IN X-DIR
def laplace_periodic(ux):
    diff = (
            (np.roll ( ux, shift=1, axis=1)
            +
            np.roll ( ux, shift=1, axis=0)
            +
            np.roll ( ux, shift=-1, axis=1)
            +
            np.roll ( ux, shift=-1, axis=0)
            -
            4*ux)/ (n_grid_size**2)
            )
    return diff

# DEFINE THE INITIAL CONDITIONS
u_initial = np.ones((n_points,n_points))
u_initial[0,:] = 0.0
u_initial[-1,:] = 0.0



# FUNCTION FOR ITERATING IN TIME
for iter in tqdm(range(n_time_steps)):
    convection = u_initial * CD_x_periodic(u_initial)
    diffusion = nu * laplace_periodic(u_initial)
    u_new = u_initial + dt*(-grad_p[0] + diffusion - convection)

    u_initial[0,:] = 0.0
    u_initial[-1,:] = 0.0

    u_initial = u_new

    # PLOTTING COMMANDS
    plt.style.use("dark_background")
    plt.contourf(co_ordinates_x,co_ordinates_y,u_new,levels = 50)
    plt.colorbar()
    plt.quiver(co_ordinates_x,co_ordinates_y,u_new,np.zeros_like(u_new))
    plt.xlabel("Position along the length of the Pipe")
    plt.ylabel("Position perpendicular to the length of the Pipe")
    

    plt.twiny()
    plt.plot(u_new[:,1], co_ordinates_y[:,1], color="white")
    plt.xlabel("Flow Velocity")


    plt.draw()
    plt.pause(0.05)
    plt.clf()

plt.show()