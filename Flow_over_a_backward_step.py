# This code simulates the flow over a backward-facing step. It is based on the same 'staggered-grid' as in the pipe-flow
# This code is a follow through of the channel ("https://www.youtube.com/@MachineLearningSimulation")


# IMPORT ALL THE DEPENDENCIES

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_POINTS_Y = 15
ASPECT_RATIO = 20
KINEMATIC_VISCOSITY = 0.01
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 6000
PLOT_EVERY = 100

STEP_HEIGHT_POINTS = 7
STEP_WIDTH_POINTS = 60

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    cell_length = 1.0 / (N_POINTS_Y - 1)

    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x_range = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x)
    y_range = np.linspace(0.0, 1.0, N_POINTS_Y)

    coordinates_x, coordinates_y = np.meshgrid(x_range, y_range)

    # Initial condition
    velocity_x_prev = np.ones((N_POINTS_Y + 1, n_points_x))
    velocity_x_prev[:(STEP_HEIGHT_POINTS + 1), :] = 0.0

    # Top Edge
    velocity_x_prev[-1, :] = - velocity_x_prev[-2, :]

    # Top Edge of the step
    velocity_x_prev[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
        - velocity_x_prev[(STEP_HEIGHT_POINTS + 1), 1:STEP_WIDTH_POINTS]
    
    # Right Edge of the step
    velocity_x_prev[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

    # Bottom Edge of the domain
    velocity_x_prev[0, (STEP_WIDTH_POINTS + 1):-1] =\
        - velocity_x_prev[1, (STEP_WIDTH_POINTS + 1):-1]
    
    # Values inside of the step
    velocity_x_prev[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

    velocity_y_prev = np.zeros((N_POINTS_Y, n_points_x+1))

    pressure_prev = np.zeros((N_POINTS_Y+1, n_points_x+1))

    # Pre-Allocate some arrays
    velocity_x_tent = np.zeros_like(velocity_x_prev)
    velocity_x_next = np.zeros_like(velocity_x_prev)

    velocity_y_tent = np.zeros_like(velocity_y_prev)
    velocity_y_next = np.zeros_like(velocity_y_prev)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6))

    for iter in tqdm(range(N_TIME_STEPS)):
        # Update interior of u velocity
        diffusion_x = KINEMATIC_VISCOSITY * (
            (
                +
                velocity_x_prev[1:-1, 2:  ]
                +
                velocity_x_prev[2:  , 1:-1]
                +
                velocity_x_prev[1:-1,  :-2]
                +
                velocity_x_prev[ :-2, 1:-1]
                - 4 *
                velocity_x_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )
        convection_x = (
            (
                velocity_x_prev[1:-1, 2:  ]**2
                -
                velocity_x_prev[1:-1,  :-2]**2
            ) / (
                2 * cell_length
            )
            +
            (
                velocity_y_prev[1:  , 1:-2]
                +
                velocity_y_prev[1:  , 2:-1]
                +
                velocity_y_prev[ :-1, 1:-2]
                +
                velocity_y_prev[ :-1, 2:-1]
            ) / 4
            *
            (
                velocity_x_prev[2:  , 1:-1]
                -
                velocity_x_prev[ :-2, 1:-1]
            ) / (
                2 * cell_length
            )
        )
        pressure_gradient_x = (
            (
                pressure_prev[1:-1, 2:-1]
                -
                pressure_prev[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        velocity_x_tent[1:-1, 1:-1] = (
            velocity_x_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_x
                +
                diffusion_x
                -
                convection_x
            )
        )

        # Apply BC

        # Inflow
        velocity_x_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        # Outflow
        inflow_mass_rate_tent = np.sum(velocity_x_tent[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_tent = np.sum(velocity_x_tent[1:-1, -2])
        velocity_x_tent[1:-1, -1] =\
            velocity_x_tent[1:-1, -2] * inflow_mass_rate_tent / outflow_mass_rate_tent
        
        # Top edge of the step
        velocity_x_tent[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
            - velocity_x_tent[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]
        
        # Right edge of the step
        velocity_x_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # Bottom edge of the domain
        velocity_x_tent[0, (STEP_WIDTH_POINTS + 1):-1] =\
            - velocity_x_tent[1, (STEP_WIDTH_POINTS + 1):-1]
        
        # Top edge of the domain
        velocity_x_tent[-1, :] = - velocity_x_tent[-2, :]

        # Set all u-velocities to zero inside the step
        velocity_x_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # Update interior of v velocity
        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                +
                velocity_y_prev[1:-1, 2:  ]
                +
                velocity_y_prev[2:  , 1:-1]
                +
                velocity_y_prev[1:-1,  :-2]
                +
                velocity_y_prev[ :-2, 1:-1]
                -
                4 * velocity_y_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )
        convection_y = (
            (
                velocity_x_prev[2:-1, 1:  ]
                +
                velocity_x_prev[2:-1,  :-1]
                +
                velocity_x_prev[1:-2, 1:  ]
                +
                velocity_x_prev[1:-2,  :-1]
            ) / 4
            *
            (
                velocity_y_prev[1:-1, 2:  ]
                -
                velocity_y_prev[1:-1,  :-2]
            ) / (
                2 * cell_length
            )
            +
            (
                velocity_y_prev[2:  , 1:-1]**2
                -
                velocity_y_prev[ :-2, 1:-1]**2
            ) / (
                2 * cell_length
            )
        )
        pressure_gradient_y = (
            (
                pressure_prev[2:-1, 1:-1]
                -
                pressure_prev[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        velocity_y_tent[1:-1, 1:-1] = (
            velocity_y_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_y
                +
                diffusion_y
                -
                convection_y
            )
        )

        # Apply BC

        # Inflow
        velocity_y_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
            - velocity_y_tent[(STEP_HEIGHT_POINTS + 1):-1, 1]
        
        # Outflow
        velocity_y_tent[1:-1, -1] = velocity_y_tent[1:-1, -2]

        # Top edge of the step
        velocity_y_tent[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        # Right edge of the step
        velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
            - velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
        
        # Bottom edge of the domain
        velocity_y_tent[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        # Top edge of the domain
        velocity_y_tent[-1, :] = 0.0

        # Set all v-velocities to zero inside of the edge
        velocity_y_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # Compute the divergence as it will be the rhs of the pressure poisson
        # problem
        divergence = (
            (
                velocity_x_tent[1:-1, 1:  ]
                -
                velocity_x_tent[1:-1,  :-1]
            ) / (
                cell_length
            )
            +
            (
                velocity_y_tent[1:  , 1:-1]
                -
                velocity_y_tent[ :-1, 1:-1]
            ) / (
                cell_length
            )
        )
        pressure_poisson_rhs = divergence / TIME_STEP_LENGTH

        # Solve the pressure correction poisson problem
        pressure_correction_prev = np.zeros_like(pressure_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            pressure_correction_next = np.zeros_like(pressure_correction_prev)
            pressure_correction_next[1:-1, 1:-1] = 1/4 * (
                +
                pressure_correction_prev[1:-1, 2:  ]
                +
                pressure_correction_prev[2:  , 1:-1]
                +
                pressure_correction_prev[1:-1,  :-2]
                +
                pressure_correction_prev[ :-2, 1:-1]
                -
                cell_length**2
                *
                pressure_poisson_rhs
            )

            # Apply pressure BC: Homogeneous Neumann everywhere except for the
            # right where is a homogeneous Dirichlet

            # Inflow
            pressure_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
                pressure_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 1]
            
            # Outflow
            pressure_correction_next[1:-1, -1] =\
                - pressure_correction_next[1:-1, -2]
            
            # Top edge of the step
            pressure_correction_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] =\
                pressure_correction_next[(STEP_HEIGHT_POINTS + 1), 1:(STEP_WIDTH_POINTS + 1)]
            
            # Right edge of the step
            pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
                pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
            
            # Bottom edge of the domain
            pressure_correction_next[0, (STEP_WIDTH_POINTS + 1):-1] =\
                pressure_correction_next[1, (STEP_WIDTH_POINTS + 1):-1]
            
            # Top edge of the domain
            pressure_correction_next[-1, :] = pressure_correction_next[-2, :]

            # Set all pressure (correction) values inside the step to zero
            pressure_correction_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

            # Advance in smoothing
            pressure_correction_prev = pressure_correction_next
        
        # Update the pressure
        pressure_next = pressure_prev + pressure_correction_next

        # Correct the velocities to be incompressible
        pressure_correction_gradient_x = (
            (
                pressure_correction_next[1:-1, 2:-1]
                -
                pressure_correction_next[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        velocity_x_next[1:-1, 1:-1] = (
            velocity_x_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            pressure_correction_gradient_x
        )

        pressure_correction_gradient_y = (
            (
                pressure_correction_next[2:-1, 1:-1]
                -
                pressure_correction_next[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        velocity_y_next[1:-1, 1:-1] = (
            velocity_y_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            pressure_correction_gradient_y
        )

        # Again enforce BC
        
        # Inflow
        velocity_x_next[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        # Outflow
        inflow_mass_rate_next = np.sum(velocity_x_next[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_next = np.sum(velocity_x_next[1:-1, -2])
        velocity_x_next[1:-1, -1] =\
            velocity_x_next[1:-1, -2] * inflow_mass_rate_next / outflow_mass_rate_next
        
        # Top edge of the step
        velocity_x_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
            - velocity_x_next[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]
        
        # Right edge of the step
        velocity_x_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # Bottom edge of the domain
        velocity_x_next[0, (STEP_WIDTH_POINTS + 1):-1] =\
            - velocity_x_next[1, (STEP_WIDTH_POINTS + 1):-1]
        
        # Top edge of the domain
        velocity_x_next[-1, :] = - velocity_x_next[-2, :]

        # Set all u-velocities to zero inside the step
        velocity_x_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0
        
        # Inflow
        velocity_y_next[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
            - velocity_y_next[(STEP_HEIGHT_POINTS + 1):-1, 1]
        
        # Outflow
        velocity_y_next[1:-1, -1] = velocity_y_next[1:-1, -2]

        # Top edge of the step
        velocity_y_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        # Right edge of the step
        velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
            - velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
        
        # Bottom edge of the domain
        velocity_y_next[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        # Top edge of the domain
        velocity_y_next[-1, :] = 0.0

        # Set all v-velocities to zero inside of the edge
        velocity_y_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0


        # Advance in time
        velocity_x_prev = velocity_x_next
        velocity_y_prev = velocity_y_next
        pressure_prev = pressure_next

        # inflow_mass_rate_next = np.sum(velocity_x_next[1:-1, 0])
        # outflow_mass_rate_next = np.sum(velocity_x_next[1:-1, -1])
        # print(f"Inflow: {inflow_mass_rate_next}")
        # print(f"Outflow: {outflow_mass_rate_next}")
        # print()

        # Visualization
        if iter % PLOT_EVERY == 0:
            velocity_x_vertex_centered = (
                (
                    velocity_x_next[1:  , :]
                    +
                    velocity_x_next[ :-1, :]
                ) / 2
            )
            velocity_y_vertex_centered = (
                (
                    velocity_y_next[:, 1:  ]
                    +
                    velocity_y_next[:,  :-1]
                ) / 2
            )

            velocity_x_vertex_centered[:(STEP_HEIGHT_POINTS + 1),:(STEP_WIDTH_POINTS + 1)] = 0.0
            velocity_y_vertex_centered[:(STEP_HEIGHT_POINTS + 1),:(STEP_WIDTH_POINTS + 1)] = 0.0

            plt.contourf(
                coordinates_x,
                coordinates_y,
                velocity_x_vertex_centered,
                levels=10,
                cmap=cmr.amber,
                vmin=-1.5,
                vmax=1.5,
            )
            plt.colorbar()

            plt.quiver(
                coordinates_x[:, ::6],
                coordinates_y[:, ::6],
                velocity_x_vertex_centered[:, ::6],
                velocity_y_vertex_centered[:, ::6],
                alpha=0.4,
            )

            plt.plot(
                5 * cell_length + velocity_x_vertex_centered[:, 5],
                coordinates_y[:, 5], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                40 * cell_length + velocity_x_vertex_centered[:, 40],
                coordinates_y[:, 40], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                80 * cell_length + velocity_x_vertex_centered[:, 80],
                coordinates_y[:, 80], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                180 * cell_length + velocity_x_vertex_centered[:, 180],
                coordinates_y[:, 180], 
                color="black",
                linewidth=3,
            )

            plt.draw()
            plt.pause(0.15)
            plt.clf()
    
    plt.figure()
    plt.streamplot(
        coordinates_x,
        coordinates_y,
        velocity_x_vertex_centered,
        velocity_y_vertex_centered,
    )

    plt.show()

if __name__ == "__main__":
    main()