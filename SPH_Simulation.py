import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm

# ----- Constants -----
MAX_PARTICLES = 125
DOMAIN_WIDTH = 40
DOMAIN_HEIGHT = 80

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = -0.9
CONSTANT_FORCE = np.array([[0.0, -0.1]])  # gravity, applied to every particle

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2500
ADD_PARTICLES_EVERY = 50

FIGURE_SIZE = (4, 6)
PLOT_EVERY = 6
SCATTER_DOT_SIZE = 2000

DOMAIN_X_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH])
DOMAIN_Y_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH])

NORMALIZATION_DENSITY = (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9)
NORMALIZATION_PRESSURE_FORCE = - (45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)
NORMALIZATION_VISCOUS_FORCE = (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)

# ----- Main Simulation Function -----
def main():
    n_particles = 1

    positions = np.zeros((n_particles, 2))
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    for iter in tqdm(range(N_TIME_STEPS)):
        # --- (A) Add new particles at the top if needed ---
        if iter % ADD_PARTICLES_EVERY == 0 and n_particles < MAX_PARTICLES:
            new_positions = np.array([
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [15 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [20 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ])
            new_velocities = np.array([
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ])
            positions = np.concatenate((positions, new_positions), axis=0)
            velocities = np.concatenate((velocities, new_velocities), axis=0)
            n_particles += 3

        # --- (B) Compute neighbors ---
        kd_tree = neighbors.KDTree(positions)
        neighbor_ids, distances = kd_tree.query_radius(
            positions,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True,
        )

        # --- (C) Compute Densities ---
        densities = np.zeros(n_particles)
        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                densities[i] += NORMALIZATION_DENSITY * (
                    SMOOTHING_LENGTH**2 - distances[i][j_in_list]**2
                )**3

        # --- (D) Compute Pressure ---
        pressures = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)

        # --- (E) Compute Forces ---
        forces = np.zeros_like(positions)
        # Remove self from the neighbor lists
        neighbor_ids = [np.delete(x, 0) for x in neighbor_ids]
        distances = [np.delete(x, 0) for x in distances]

        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                # Pressure force contribution:
                forces[i] += NORMALIZATION_PRESSURE_FORCE * (
                    - (positions[j] - positions[i]) / distances[i][j_in_list] *
                    (pressures[j] + pressures[i]) / (2 * densities[j]) *
                    (SMOOTHING_LENGTH - distances[i][j_in_list])**2
                )
                # Viscous force contribution:
                forces[i] += NORMALIZATION_VISCOUS_FORCE * (
                    (velocities[j] - velocities[i]) / densities[j] *
                    (SMOOTHING_LENGTH - distances[i][j_in_list])
                )

        # Add gravity (constant force)
        forces += CONSTANT_FORCE
        # Alternative: forces += CONSTANT_FORCE * densities[:, np.newaxis]
        # Uncomment the above and comment the plain addition if desired.

        # --- (F) Integrate (Euler Step) ---
        velocities = velocities + TIME_STEP_LENGTH * forces / densities[:, np.newaxis]
        positions = positions + TIME_STEP_LENGTH * velocities

        # --- (G) Enforce Boundary Conditions ---
        out_of_left_boundary   = positions[:, 0] < DOMAIN_X_LIM[0]
        out_of_right_boundary  = positions[:, 0] > DOMAIN_X_LIM[1]
        out_of_bottom_boundary = positions[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary    = positions[:, 1] > DOMAIN_Y_LIM[1]

        velocities[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

        velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

        velocities[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

        # --- (H) Plotting ---
        if iter % PLOT_EVERY == 0:
            plt.scatter(
                positions[:, 0],
                positions[:, 1],
                s=SCATTER_DOT_SIZE,
                c=positions[:, 1],
                cmap="Wistia_r",
            )
            plt.ylim(DOMAIN_Y_LIM)
            plt.xlim(DOMAIN_X_LIM)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


if __name__ == "__main__":
    main()
