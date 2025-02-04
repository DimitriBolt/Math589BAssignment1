import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Initialization
# -----------------------------
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` parameter, if non-zero, adds a small spiral twist to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i * i)
    return positions

# -----------------------------
# Potential Energy Functions
# -----------------------------
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute Lennard-Jones potential between two beads.
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def bond_potential(r, b=1.0, k_b=100.0):
    """
    Compute harmonic bond potential between two bonded beads.
    """
    return k_b * (r - b)**2

# -----------------------------
# Total Energy and Analytic Gradient
# -----------------------------
def total_energy_with_grad(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation and its analytic gradient.
    
    Parameters:
      x        : flattened numpy array containing the positions.
      n_beads  : number of beads in the protein.
      epsilon, sigma : parameters for the Lennard-Jones potential.
      b, k_b   : parameters for the bond potential.
    
    Returns:
      f : the total energy (a scalar)
      g : the gradient of the energy with respect to x (a flattened array)
    """
    # Reshape the flattened vector into an array of positions.
    positions = x.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    energy = 0.0
    grad = np.zeros_like(positions)

    # ---- Bond Energy and its Gradient ----
    # For each pair of adjacent beads
    for i in range(n_beads - 1):
        d_vec = positions[i+1] - positions[i]
        r = np.linalg.norm(d_vec)
        if r == 0:
            continue  # safeguard against division by zero
        # Bond energy: f_bond = k_b * (r - b)**2
        energy += bond_potential(r, b, k_b)
        # Derivative: df/dr = 2 * k_b * (r - b)
        dE_dr = 2 * k_b * (r - b)
        # The gradient with respect to positions[i] and positions[i+1]
        # Note: d(r)/d(positions[i]) = -(d_vec)/r, and for positions[i+1] it is + (d_vec)/r.
        d_grad = (dE_dr / r) * d_vec
        grad[i]   -= d_grad
        grad[i+1] += d_grad

    # ---- Lennard-Jones Energy and its Gradient ----
    # For each pair of non-bonded beads (i < j)
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            d_vec = positions[i] - positions[j]
            r = np.linalg.norm(d_vec)
            # Skip if too close to avoid singularity
            if r < 1e-2:
                continue
            # LJ energy: f_LJ = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
            lj_energy = lennard_jones_potential(r, epsilon, sigma)
            energy += lj_energy
            # Derivative: df/dr = 4*epsilon*( -12*sigma**12/r**13 + 6*sigma**6/r**7 )
            dE_dr = 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)
            # Gradient contribution: (df/dr) * (d_vec/r) for positions[i] and the negative for positions[j]
            d_grad = (dE_dr / r) * d_vec
            grad[i] += d_grad
            grad[j] -= d_grad

    # Flatten the gradient to match the shape of x
    return energy, grad.flatten()

# -----------------------------
# Optimization Function
# -----------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein to minimize total energy.
    
    Parameters:
    ----------
    positions : np.ndarray
        A 2D NumPy array of shape (n_beads, d) representing the initial
        positions of the protein's beads in d-dimensional space.
    
    n_beads : int
        The number of beads (or units) in the protein model.
    
    write_csv : bool, optional (default=False)
        If True, the final optimized positions are saved to a CSV file.
    
    maxiter : int, optional (default=1000)
        The maximum number of iterations for the BFGS optimization algorithm.
    
    tol : float, optional (default=1e-6)
        The tolerance level for convergence in the optimization.
    
    Returns:
    -------
    result : scipy.optimize.OptimizeResult
        The result of the optimization process, containing information
        such as the optimized positions and convergence status.
    
    trajectory : list of np.ndarray
        A list of intermediate configurations during the optimization,
        where each element is an (n_beads, d) array representing the
        positions of the beads at that step.
    """
    trajectory = []

    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Iteration {len(trajectory)}")

    # Call minimize using our analytic energy+gradient function.
    result = minimize(
        fun=total_energy_with_grad,
        x0=positions.flatten(),
        args=(n_beads,),
        method='BFGS',
        jac=True,
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")

    return result, trajectory

# -----------------------------
# 3D Visualization
# -----------------------------
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# -----------------------------
# Animation Function
# -----------------------------
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D with autoscaling.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    n_beads = 100
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)

    print("Initial Energy:", total_energy_with_grad(initial_positions.flatten(), n_beads)[0])
    plot_protein_3d(initial_positions, title="Initial Configuration")

    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=3000, tol=1e-6)

    optimized_positions = result.x.reshape((n_beads, dimension))
    print("Optimized Energy:", total_energy_with_grad(optimized_positions.flatten(), n_beads)[0])
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process
    animate_optimization(trajectory)
