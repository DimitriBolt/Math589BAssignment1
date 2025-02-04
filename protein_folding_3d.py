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
# Total Energy and Analytic Gradient (Vectorized LJ)
# -----------------------------
def total_energy_with_grad(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation and its analytic gradient.
    
    This version uses a vectorized computation for the Lennard-Jones interactions.
    
    Parameters:
      x        : flattened numpy array containing the positions.
      n_beads  : number of beads in the protein.
      epsilon, sigma : parameters for the Lennard-Jones potential.
      b, k_b   : parameters for the bond potential.
    
    Returns:
      f : the total energy (a scalar)
      g : the gradient of the energy with respect to x (a flattened array)
    """
    # Reshape x into (n_beads, n_dim)
    positions = x.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    energy = 0.0
    grad = np.zeros_like(positions)

    # ---- Bond Energy and its Gradient (loop over bonds) ----
    for i in range(n_beads - 1):
        d_vec = positions[i+1] - positions[i]
        r = np.linalg.norm(d_vec)
        if r == 0:
            continue  # safeguard
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
    diff_ij = diff[idx_i, idx_j]
    diff_valid = diff_ij[valid]
    contrib = (dE_dr[:, None] / r_valid[:, None]) * diff_valid
    valid_i = idx_i[valid]
    valid_j = idx_j[valid]
    np.add.at(grad, valid_i, contrib)
    np.add.at(grad, valid_j, -contrib)
    
    return energy, grad.flatten()

# -----------------------------
# Extra Refinement via Gradient Descent with Backtracking
# -----------------------------
def refine_solution(x, n_beads, tol, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0,
                    max_iter=100, alpha0=1.0, beta=0.5, c=1e-4):
    """
    Refine the current solution x by performing gradient descent with backtracking
    until the gradient norm is below tol or max_iter iterations are reached.
    """
    iter_count = 0
    while iter_count < max_iter:
        energy_val, grad_val = total_energy_with_grad(x, n_beads, epsilon, sigma, b, k_b)
        grad_norm = np.linalg.norm(grad_val)
        if grad_norm <= tol:
            break
        alpha = alpha0
        f_current = energy_val
        while True:
            x_new = x - alpha * grad_val
            f_new, _ = total_energy_with_grad(x_new, n_beads, epsilon, sigma, b, k_b)
            if f_new <= f_current - c * alpha * grad_norm**2:
                break
            alpha *= beta
            if alpha < 1e-12:
                break
        x = x_new
        iter_count += 1
    return x

# -----------------------------
# Optimization Function with Multi-Start Refinement
# -----------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein to minimize total energy.
    
    This implementation first runs BFGS (with analytic gradient and extra refinement)
    and then, if the resulting energy is not low enough, performs several rounds
    of perturbed restarts to try to reach a deeper minimum.
    
    Parameters:
      positions : initial (n_beads, d) configuration.
      n_beads   : number of beads.
      write_csv : if True, writes final configuration to CSV.
      maxiter   : maximum iterations for BFGS.
      tol       : tolerance for gradient norm.
    
    Returns:
      result : final optimization result (scipy.optimize.OptimizeResult)
      trajectory : list of intermediate configurations.
    """
    trajectory = []
    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Iteration {len(trajectory)}")
    
    # First run of BFGS.
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
    
    energy_val, grad_val = total_energy_with_grad(result.x, n_beads)
    grad_norm = np.linalg.norm(grad_val)
    print(f"Initial minimization complete, gradient norm = {grad_norm:.8f}")
    
    # Refinement with gradient descent.
    if grad_norm > tol:
        print("Entering extra refinement with gradient descent...")
        x_refined = refine_solution(result.x, n_beads, tol, max_iter=1000)
        energy_val, grad_val = total_energy_with_grad(x_refined, n_beads)
        grad_norm = np.linalg.norm(grad_val)
        print(f"After refinement, gradient norm = {grad_norm:.8f}")
        result.x = x_refined
    
    # Check energy and, if it is not low enough, attempt a few perturbed restarts.
    best_energy = energy_val
    best_x = result.x.copy()
    n_perturb = 5
    noise_scale = 1e-3  # adjust this scale if needed
    for i in range(n_perturb):
        print(f"Perturbed restart {i+1}...")
        # Add small random noise
        x_perturbed = best_x + np.random.normal(scale=noise_scale, size=best_x.shape)
        res_pert = minimize(
            fun=total_energy_with_grad,
            x0=x_perturbed,
            args=(n_beads,),
            method='BFGS',
            jac=True,
            tol=tol,
            options={'maxiter': maxiter, 'disp': False}
        )
        e_pert, _ = total_energy_with_grad(res_pert.x, n_beads)
        print(f"  Energy = {e_pert:.6f}")
        if e_pert < best_energy:
            best_energy = e_pert
            best_x = res_pert.x.copy()
    
    print(f"Best energy found = {best_energy:.6f}")
    # Update result.x with the best solution.
    result.x = best_x
    
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
    n_beads = 100  # testing with 100 beads
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)
    
    init_E, _ = total_energy_with_grad(initial_positions.flatten(), n_beads)
    print("Initial Energy:", init_E)
    plot_protein_3d(initial_positions, title="Initial Configuration")
    
    # Use maxiter and tol as given by the autograder.
    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=3000, tol=1e-4)
    
    optimized_positions = result.x.reshape((n_beads, dimension))
    opt_E, _ = total_energy_with_grad(optimized_positions.flatten(), n_beads)
    print("Optimized Energy:", opt_E)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")
    
    animate_optimization(trajectory)
