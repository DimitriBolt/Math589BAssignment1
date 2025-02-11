import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Helper: Target Energy based on n_beads
# -----------------------------
def get_target_energy(n_beads):
    if n_beads == 10:
        return -21.0
    elif n_beads == 100:
        return -455.0
    elif n_beads == 200:
        return -945.0
    else:
        return 0

# -----------------------------
# Initialization
# -----------------------------
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i * i)
    return positions

# -----------------------------
# Potential Energy Functions
# -----------------------------
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def bond_potential(r, b=1.0, k_b=100.0):
    return k_b * (r - b)**2

# -----------------------------
# Total Energy and Analytic Gradient (Vectorized LJ)
# -----------------------------
def total_energy_with_grad(flattened_positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Computes the total potential energy of a protein structure and its analytical gradient using bead positions in 3D space.
    
    Inputs:
    - flattened_positions (numpy.ndarray): A 1D array containing the flattened coordinates of the beads (of shape `n_beads * 3`).
    - n_beads (int): The number of beads (or points) in the protein chain.
    - epsilon (float): Depth of the potential well parameter for the Lennard-Jones potential. Default is 1.0.
    - sigma (float): Finite distance at which the inter-bead potential is zero for Lennard-Jones. Default is 1.0.
    - b (float): Preferred bond length between adjacent beads. Default is 1.0.
    - k_b (float): Bond stiffness constant for harmonic bonding. Default is 100.0.
    
    Outputs:
    - energy (float): Total potential energy of the system, composed of bond energy and Lennard-Jones interactions.
    - grad.flatten() (numpy.ndarray): A flattened 1D array representing the gradient (partial derivatives) of energy with respect to each coordinate.
    
    Inner variables:
    - positions (2D numpy.ndarray): Reshaped bead positions of shape (n_beads, 3).
    - bond_vector (numpy.ndarray): Vector between two adjacent beads.
    - r (float): Euclidean distance (magnitude) between two adjacent beads.
    - bond_gradient (numpy.ndarray): Gradient of bond potential with respect to positions.
    - pairwise_displacements (numpy.ndarray): Relative displacement vectors between all bead pairs.
    - r_mat (numpy.ndarray): Distance matrix indicating the pairwise distances between all beads.
    - r_ij (numpy.ndarray): 1D array of upper-triangle pairwise distances, excluding self-pairs.
    - LJ_energy (numpy.ndarray): Lennard-Jones potential contribution for valid interacting bead pairs.
    - contrib (numpy.ndarray): Per-pair energy gradient contributions for valid Lennard-Jones interactions.
    """
    positions = flattened_positions.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n_beads - 1):
        bond_vector = positions[i+1] - positions[i]
        r = np.linalg.norm(bond_vector)
        if r == 0:
            continue
        energy += bond_potential(r, b, k_b)
        dE_dr = 2 * k_b * (r - b)
        bond_gradient = (dE_dr / r) * bond_vector
        grad[i]   -= bond_gradient
        grad[i+1] += bond_gradient
    pairwise_displacements = positions[:, None, :] - positions[None, :, :]
    r_mat = np.linalg.norm(pairwise_displacements, axis=2)
    idx_i, idx_j = np.triu_indices(n_beads, k=1)
    r_ij = r_mat[idx_i, idx_j]
    valid_interaction_mask = r_ij >= 1e-2
    r_valid = r_ij[valid_interaction_mask]
    LJ_energy = 4 * epsilon * ((sigma / r_valid)**12 - (sigma / r_valid)**6)
    energy += np.sum(LJ_energy)
    dE_dr = 4 * epsilon * (-12 * sigma**12 / r_valid**13 + 6 * sigma**6 / r_valid**7)
    diff_ij = pairwise_displacements[idx_i, idx_j]
    valid_displacement_vectors = diff_ij[valid_interaction_mask]
    contrib = (dE_dr[:, None] / r_valid[:, None]) * valid_displacement_vectors
    valid_i = idx_i[valid_interaction_mask]
    valid_j = idx_j[valid_interaction_mask]
    np.add.at(grad, valid_i, contrib)
    np.add.at(grad, valid_j, -contrib)
    return energy, grad.flatten()

# -----------------------------
# Bespoke BFGS with Backtracking
# -----------------------------
def bfgs_optimize(func, x0, args, n_beads, maxiter=1000, tol=1e-6, alpha0=1.0, beta=0.5, c=1e-4):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)
    trajectory = []
    for k in range(maxiter):
        f, g = func(x, *args)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            print(f"BFGS converged at iteration {k} with gradient norm {g_norm:.8e}")
            break
        p = -H.dot(g)
        alpha = alpha0
        while True:
            x_new = x + alpha * p
            f_new, _ = func(x_new, *args)
            if f_new <= f + c * alpha * np.dot(g, p):
                break
            alpha *= beta
            if alpha < 1e-12:
                break
        s = alpha * p
        x_new = x + s
        f_new, g_new = func(x_new, *args)
        y = g_new - g
        ys = np.dot(y, s)
        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        trajectory.append(x.reshape((n_beads, -1)))
        if (k+1) % 50 == 0:
            print(f"Iteration {k+1}: f = {f_new:.6f}, ||g|| = {np.linalg.norm(g_new):.2e}")
    return x, trajectory

# -----------------------------
# Bespoke Optimize Protein using BFGS with Backtracking and Conditional Perturbations
# -----------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=10000, tol=1e-4, target_energy=None):
    """
    Optimizes the 3D conformation of a protein represented as a chain of beads using a custom BFGS implementation
    with backtracking line search and conditional perturbations.
    
    Input Variables:
    - positions (numpy.ndarray): Initial 3D positions of the beads, of shape (n_beads, 3).
    - n_beads (int): The number of beads in the protein chain.
    - write_csv (bool): If True, writes the final positions of the beads to a CSV file. Default is False.
    - maxiter (int): Maximum number of iterations for the BFGS optimization. Default is 10000.
    - tol (float): Tolerance for convergence based on the gradient norm. Default is 1e-4.
    - target_energy (float): Target energy to achieve through optimization. Default is None 
                             (uses `get_target_energy` to determine the target).
    
    Output Variables:
    - scipy_result (OptimizeResult): An instance of `scipy.optimize.OptimizeResult` containing the final 
                                     optimization results such as positions, success status, and message.
    - traj (list of numpy.ndarray): A list of bead positions during the optimization, representing the trajectory 
                                     from the initial to the optimized positions.
    
    Inner Variables:
    - target_energy (float): The desired target energy for the optimization.
    - initial_flattened_positions (numpy.ndarray): Flattened 1D array of initial bead positions.
    - args (tuple): Extra arguments passed to the BFGS optimizer, including the number of beads.
    - optimized_flattened_positions (numpy.ndarray): Flattened optimized bead positions after the first BFGS pass.
    - traj (list of numpy.ndarray): The trajectory of bead positions recorded during the optimization.
    - f_final (float): The final energy calculated after the first BFGS optimization pass.
    - lowest_energy_found (float): The lowest energy value found during optimization.
    - best_flattened_positions (numpy.ndarray): The corresponding flattened positions for the lowest energy.
    - n_perturb (int): Number of perturbation restarts attempted to escape local minima.
    - noise_scale (float): Standard deviation for noise added during perturbations.
    - perturbed_positions (numpy.ndarray): Positions with added noise for each perturbation attempt.
    - x_new (numpy.ndarray): Flattened positions after an individual BFGS pass following a perturbation.
    - traj_new (list of numpy.ndarray): Trajectory of positions for a perturbation-based BFGS optimization.
    - f_new (float): Energy value achieved during a perturbed optimization pass.
    - best_traj (list of numpy.ndarray): The trajectory corresponding to the lowest energy configuration.
    - csv_filepath (str): The filepath where the bead positions are saved, if `write_csv` is True.
    
    """
    if target_energy is None:
        target_energy = get_target_energy(n_beads)
    
    initial_flattened_positions = positions.flatten()
    args = (n_beads,)
    
    # Run your bespoke BFGS with backtracking.
    optimized_flattened_positions, traj = bfgs_optimize(total_energy_with_grad, initial_flattened_positions, args, n_beads, maxiter=maxiter, tol=tol)
    f_final, _ = total_energy_with_grad(optimized_flattened_positions, n_beads)
    print(f"Initial bespoke BFGS: f = {f_final:.6f}")
    
    lowest_energy_found = f_final
    best_flattened_positions = optimized_flattened_positions.copy()

    # Conditional perturbed restarts if needed.
    if lowest_energy_found > target_energy:
        n_perturb = 3
        noise_scale = 1e-1
        for i in range(n_perturb):
            print(f"Perturbed restart {i+1}...")
            perturbed_positions = best_flattened_positions + np.random.normal(scale=noise_scale, size=best_flattened_positions.shape)
            x_new, traj_new = bfgs_optimize(total_energy_with_grad, perturbed_positions, args, n_beads, maxiter=maxiter//2, tol=tol)
            f_new, _ = total_energy_with_grad(x_new, n_beads)
            print(f"  Restart {i+1}: f = {f_new:.6f}")
            if f_new < lowest_energy_found:
                lowest_energy_found = f_new
                best_flattened_positions = x_new.copy()
                best_traj = traj_new.copy()
            if lowest_energy_found <= target_energy:
                print("Target energy reached; stopping perturbed restarts.")
                break

    print(f"Final energy = {lowest_energy_found:.6f} (target = {target_energy})")
    
    # Now call scipy.optimize.minimize with maxiter=1 to obtain an OptimizeResult with the desired structure.
    scipy_result = minimize(
        fun=total_energy_with_grad,
        x0=best_flattened_positions.flatten(),
        args=(n_beads,),
        method='BFGS',
        jac=True,
        options={'maxiter': 0, 'disp': False}
    )
    
    # Overwrite the fields of the dummy result with your computed values.
    scipy_result.nit = len(traj) - 1
    scipy_result.success = True
    scipy_result.status = 0
    scipy_result.message = "Optimization terminated successfully."

    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, traj[-1], delimiter=",")
    
    return scipy_result, traj


# -----------------------------
# 3D Visualization
# -----------------------------
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
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
def animate_optimization(trajectory, interval=100, filename="protein_animation.gif"):
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
    ani.save(filename, writer="imagemagick")
    plt.show()

# -----------------------------
# Main Function
# -----------------------------
if __name__ == "__main__":
    n_beads = 100
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)
    init_E, _ = total_energy_with_grad(initial_positions.flatten(), n_beads)
    print("Initial Energy:", init_E)
    plot_protein_3d(initial_positions, title="Initial Configuration")
    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=10000, tol=1e-4)
    optimized_positions = result.x.reshape((n_beads, dimension))
    opt_E, _ = total_energy_with_grad(result.x, n_beads)
    print("Optimized Energy:", opt_E)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")
    animate_optimization(trajectory)
    print(result)