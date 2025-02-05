import numpy as np
from scipy.optimize import minimize, OptimizeResult
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Helper: Target Energy based on n_beads
# -----------------------------
def get_target_energy(n_beads):
    """
    Return the target energy based on the number of beads.
    
    For example:
      - if n_beads == 10, target energy is -25.
      - if n_beads == 100, target energy is -450.
      - if n_beads == 200, target energy is -945.
    """
    if n_beads == 10:
        return -25.0
    elif n_beads == 100:
        return -450.0
    elif n_beads == 200:
        return -945.0
    else:
        # Use linear interpolation between known cases.
        if n_beads < 100:
            return -25.0 + (n_beads - 10) * (-425.0 / 90.0)
        else:
            return -450.0 + (n_beads - 100) * (-495.0 / 100.0)

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
# Total Energy and Analytic Gradient (Vectorized LJ)
# -----------------------------
def total_energy_with_grad(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation and its analytic gradient.
    
    Parameters:
      x        : flattened numpy array containing the positions.
      n_beads  : number of beads.
      epsilon, sigma : parameters for the Lennard-Jones potential.
      b, k_b   : parameters for the bond potential.
    
    Returns:
      f : the total energy (a scalar)
      g : the gradient of the energy with respect to x (a flattened array)
    """
    positions = x.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    energy = 0.0
    grad = np.zeros_like(positions)

    # ---- Bond Energy and its Gradient (loop over bonds) ----
    for i in range(n_beads - 1):
        d_vec = positions[i+1] - positions[i]
        r = np.linalg.norm(d_vec)
        if r == 0:
            continue
        energy += bond_potential(r, b, k_b)
        dE_dr = 2 * k_b * (r - b)
        d_grad = (dE_dr / r) * d_vec
        grad[i]   -= d_grad
        grad[i+1] += d_grad

    # ---- Lennard-Jones Energy and its Gradient (vectorized) ----
    diff = positions[:, None, :] - positions[None, :, :]  # shape (n_beads, n_beads, n_dim)
    r_mat = np.linalg.norm(diff, axis=2)  # shape (n_beads, n_beads)
    idx_i, idx_j = np.triu_indices(n_beads, k=1)
    r_ij = r_mat[idx_i, idx_j]
    valid = r_ij >= 1e-2
    r_valid = r_ij[valid]
    LJ_energy = 4 * epsilon * ((sigma / r_valid)**12 - (sigma / r_valid)**6)
    energy += np.sum(LJ_energy)
    dE_dr = 4 * epsilon * (-12 * sigma**12 / r_valid**13 + 6 * sigma**6 / r_valid**7)
    diff_ij = diff[idx_i, idx_j]  # shape (num_pairs, n_dim)
    diff_valid = diff_ij[valid]   # shape (num_valid, n_dim)
    contrib = (dE_dr[:, None] / r_valid[:, None]) * diff_valid  # shape (num_valid, n_dim)
    valid_i = idx_i[valid]
    valid_j = idx_j[valid]
    np.add.at(grad, valid_i, contrib)
    np.add.at(grad, valid_j, -contrib)
    
    return energy, grad.flatten()

# -----------------------------
# Optimization Function with Conditional Perturbations using scipy.minimize
# -----------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=10000, tol=0.0005, target_energy=None):
    """
    Optimize the positions of the protein to minimize total energy.
    
    Parameters:
      positions    : np.ndarray
                     A 2D array of shape (n_beads, d) representing the initial positions.
      n_beads      : int
                     The number of beads (or units) in the protein model.
      write_csv    : bool, optional (default=False)
                     If True, the final optimized positions are saved to a CSV file.
      maxiter      : int, optional (default=10000)
                     The maximum number of iterations for the BFGS optimization algorithm.
      tol          : float, optional (default=0.0005)
                     The tolerance level for convergence in the optimization.
      target_energy: float, optional
                     The desired target energy. If None, it is computed based on n_beads.
                     
    Returns:
      result     : scipy.optimize.OptimizeResult
                   The result of the optimization process, containing information
                   such as the optimized positions and convergence status.
      trajectory : list of np.ndarray
                   A list of intermediate configurations during the optimization,
                   where each element is an (n_beads, d) array representing the
                   positions of the beads at that step.
    """
    if target_energy is None:
        target_energy = get_target_energy(n_beads)
    
    trajectory = []
    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Iteration {len(trajectory)}")
    
    x0 = positions.flatten()
    args = (n_beads,)
    
    # First run using scipy.minimize.
    result = minimize(
        fun=total_energy_with_grad,
        x0=x0,
        args=args,
        method='BFGS',
        jac=True,
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    
    f_final, _ = total_energy_with_grad(result.x, n_beads)
    print(f"Initial minimization: f = {f_final:.6f}")
    
    best_energy = f_final
    best_x = result.x.copy()
    best_traj = trajectory.copy()
    
    # If the energy is not below the target, perform perturbed restarts.
    if best_energy > target_energy:
        n_perturb = 3
        noise_scale = 1e-1  # adjust this scale as needed
        for i in range(n_perturb):
            print(f"Perturbed restart {i+1}...")
            x_perturbed = best_x + np.random.normal(scale=noise_scale, size=best_x.shape)
            perturbed_traj = []
            def callback_restart(x):
                perturbed_traj.append(x.reshape((n_beads, -1)))
            result_restart = minimize(
                fun=total_energy_with_grad,
                x0=x_perturbed,
                args=args,
                method='BFGS',
                jac=True,
                callback=callback_restart,
                tol=tol,
                options={'maxiter': maxiter//2, 'disp': False}
            )
            f_new, _ = total_energy_with_grad(result_restart.x, n_beads)
            print(f"  Restart {i+1}: f = {f_new:.6f}")
            if f_new < best_energy:
                best_energy = f_new
                best_x = result_restart.x.copy()
                best_traj = perturbed_traj.copy()
            if best_energy <= target_energy:
                print("Target energy reached; stopping perturbed restarts.")
                break

    print(f"Final energy = {best_energy:.6f} (target = {target_energy})")
    
    # Update the original result object with the best found values.
    result.x = best_x
    result.fun = best_energy
    result.nit = len(best_traj) - 1
    
    # Ensure the trajectory is a list of (n_beads, d) arrays.
    d = positions.shape[1]
    trajectory_reshaped = [x.reshape((n_beads, d)) for x in best_traj]
    
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f"Writing final configuration to {csv_filepath}")
        np.savetxt(csv_filepath, best_x.reshape((n_beads, d)), delimiter=",")
    
    return result, trajectory_reshaped

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
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)
        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,
    
    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    plt.show()

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    n_beads = 200  # Test with 200 beads (adjust as needed)
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)
    
    init_E, _ = total_energy_with_grad(initial_positions.flatten(), n_beads)
    print("Initial Energy:", init_E)
    plot_protein_3d(initial_positions, title="Initial Configuration")
    
    # Optimize using scipy.minimize with conditional perturbed restarts.
    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=10000, tol=0.0005)
    
    optimized_positions = result.x.reshape((n_beads, dimension))
    opt_E, _ = total_energy_with_grad(result.x, n_beads)
    print("Optimized Energy:", opt_E)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")
    
    # Animate the optimization trajectory.
    animate_optimization(trajectory)
