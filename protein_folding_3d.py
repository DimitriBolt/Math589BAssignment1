import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist, squareform

# Try to import numba. If not available, set flag to False.
try:
    import numba
    USE_NUMBA = False
except ImportError:
    USE_NUMBA = False

# ------------------------------
# 1. Protein Initialization
# ------------------------------
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` parameter adds a slight spiral structure.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1  # initial bond length of 1 unit along x
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i * i)
    return positions

# ------------------------------
# 2. Potential Functions
# ------------------------------
# Lennard-Jones potential and derivative (with increased attraction)
def lennard_jones_potential(r, epsilon=5.0, sigma=1.0):
    """Compute Lennard-Jones potential between two beads."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def lennard_jones_deriv(r, epsilon=5.0, sigma=1.0):
    """Compute the derivative dV/dr of the Lennard-Jones potential."""
    return 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)

# Bond potential and derivative (with lower equilibrium bond length and stiffness)
def bond_potential(r, b=0.9, k_b=10.0):
    """Compute harmonic bond potential between two bonded beads."""
    return k_b * (r - b)**2

def bond_deriv(r, b=0.9, k_b=10.0):
    """Compute the derivative dV/dr of the bond potential."""
    return 2 * k_b * (r - b)

# ------------------------------
# 3. Energy and Gradient for Bonded Interactions
# ------------------------------
def bond_energy_and_grad(positions, n_beads, b=0.9, k_b=10.0):
    energy_bond = 0.0
    grad_bond = np.zeros_like(positions)
    for i in range(n_beads - 1):
        diff = positions[i + 1] - positions[i]
        r = np.linalg.norm(diff)
        energy_bond += bond_potential(r, b, k_b)
        if r > 1e-12:  # avoid division by zero
            dV_dr = bond_deriv(r, b, k_b)
            grad_contrib = dV_dr * (diff / r)
        else:
            grad_contrib = 0 * diff
        grad_bond[i]   -= grad_contrib
        grad_bond[i + 1] += grad_contrib
    return energy_bond, grad_bond

# ------------------------------
# 4a. Vectorized LJ Energy and Gradient with Cutoff
# ------------------------------
def lj_energy_and_grad_vectorized(positions, epsilon=5.0, sigma=1.0, cutoff=5.0):
    """
    Compute Lennard-Jones energy and gradient in a vectorized manner using pdist/squareform.
    
    Parameters:
        positions : np.ndarray, shape (n_beads, 3)
        epsilon, sigma : float, LJ parameters.
        cutoff : float, interactions beyond this distance are ignored.
    
    Returns:
        energy_lj : float, total Lennard-Jones energy.
        grad_lj   : np.ndarray, shape (n_beads, 3), gradient.
    """
    n_beads = positions.shape[0]
    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(positions))
    # Exclude self-interactions by setting the diagonal to a large number
    np.fill_diagonal(dist_matrix, np.inf)
    # Create a boolean mask for pairs within cutoff
    mask = dist_matrix < cutoff

    # Compute energies for each pair; use np.where to avoid division by zero
    r = np.where(mask, dist_matrix, 1.0)  # dummy value where mask is False
    E_matrix = np.where(mask, 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6), 0.0)
    # Sum only over one triangle (each pair counted once)
    energy_lj = np.sum(np.triu(E_matrix, 1))
    
    # For the gradient, compute derivative dV/dr for pairs within the cutoff.
    dV_dr = np.zeros_like(r)
    dV_dr[mask] = 4 * epsilon * (-12 * sigma**12 / r[mask]**13 + 6 * sigma**6 / r[mask]**7)
    
    # Compute pairwise differences for each coordinate (broadcasting)
    diff = positions[:, None, :] - positions[None, :, :]  # shape: (n_beads, n_beads, 3)
    # Normalize differences (avoid division by zero using the mask)
    norm_diff = np.where(mask[..., None], diff / r[..., None], 0.0)
    
    # Each pair contributes a force dV_dr * (diff/r); sum contributions along axis 1.
    forces = np.sum(dV_dr[..., None] * norm_diff, axis=1)
    # The gradient is the force (for minimization, we use the gradient as computed)
    grad_lj = forces

    return energy_lj, grad_lj

# ------------------------------
# 4b. Numba-Accelerated LJ Energy and Gradient with Cutoff
# ------------------------------
if USE_NUMBA:
    @numba.njit
    def lj_energy_and_grad_numba(positions, epsilon, sigma, cutoff):
        n_beads = positions.shape[0]
        energy = 0.0
        grad = np.zeros_like(positions)
        for i in range(n_beads):
            for j in range(i+1, n_beads):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dz = positions[i, 2] - positions[j, 2]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r < cutoff and r > 1e-2:
                    sr = sigma / r
                    sr6 = sr**6
                    sr12 = sr6 * sr6
                    E = 4 * epsilon * (sr12 - sr6)
                    energy += E
                    dV_dr = 4 * epsilon * (-12 * (sigma**12) / (r**13) + 6 * (sigma**6) / (r**7))
                    fx = dV_dr * dx / r
                    fy = dV_dr * dy / r
                    fz = dV_dr * dz / r
                    grad[i, 0] += fx
                    grad[i, 1] += fy
                    grad[i, 2] += fz
                    grad[j, 0] -= fx
                    grad[j, 1] -= fy
                    grad[j, 2] -= fz
        return energy, grad

# ------------------------------
# 5. Total Energy and Gradient Function
# ------------------------------
def total_energy_grad(positions, n_beads, 
                      epsilon=5.0, sigma=1.0,  # stronger LJ attraction
                      b=0.9, k_b=10.0,       # bonds can contract more easily
                      cutoff=5.0,            # extended cutoff
                      use_numba=True):
    """
    Compute the total energy and its gradient for the protein configuration.
    
    Parameters:
        positions : np.ndarray
            A flat array of length (n_beads*3).
        n_beads : int
            The number of beads.
        epsilon, sigma : float, LJ parameters.
        b, k_b : float, bond parameters.
        cutoff : float, cutoff distance for LJ interactions.
        use_numba : bool, if True and numba is available, use the numba-accelerated LJ.
    
    Returns:
        total_energy : float, the total energy.
        total_grad : np.ndarray, a flat array containing the gradient.
    """
    positions = positions.reshape((n_beads, 3))
    energy_bond, grad_bond = bond_energy_and_grad(positions, n_beads, b, k_b)
    
    # Choose LJ computation method:
    if use_numba and USE_NUMBA:
        energy_lj, grad_lj = lj_energy_and_grad_numba(positions, epsilon, sigma, cutoff)
    else:
        energy_lj, grad_lj = lj_energy_and_grad_vectorized(positions, epsilon, sigma, cutoff)
        
    total_energy_val = energy_bond + energy_lj
    total_grad = grad_bond + grad_lj
    return total_energy_val, total_grad.flatten()

# Wrapper for energy only (used by the optimizer)
def total_energy_wrapper(positions, n_beads, 
                         epsilon=5.0, sigma=1.0, 
                         b=0.9, k_b=10.0):
    energy, _ = total_energy_grad(positions, n_beads, epsilon, sigma, b, k_b)
    return energy

# ------------------------------
# 6. Optimization Function Using L-BFGS-B
# ------------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    """
    Optimize the positions of the protein to minimize total energy.
    
    Returns:
        result : OptimizeResult from scipy.optimize.minimize.
        trajectory : list of intermediate configurations.
    """
    trajectory = []

    def callback(x):
        trajectory.append(x.reshape((n_beads, 3)))
        if len(trajectory) % 20 == 0:
            print(f"Trajectory length: {len(trajectory)}")

    result = minimize(
        fun=total_energy_wrapper,
        x0=positions.flatten(),
        args=(n_beads,),
        method='L-BFGS-B',
        jac=lambda x, n_beads=n_beads: total_energy_grad(x, n_beads)[1],
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing final positions to {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    
    return result, trajectory

# ------------------------------
# 7. 3D Visualization and Animation Functions
# ------------------------------
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
        # Autoscale axes
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

# ------------------------------
# 8. Main Function
# ------------------------------
if __name__ == "__main__":
    # For testing by the autograder, n_beads = 10
    n_beads = 200
    dimension = 3

    # Initialize configuration
    initial_positions = initialize_protein(n_beads, dimension)
    print("Initial Energy:", total_energy_wrapper(initial_positions.flatten(), n_beads))
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # Optimize using L-BFGS-B with analytical gradient
    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=1000, tol=1e-6)

    optimized_positions = result.x.reshape((n_beads, dimension))
    final_energy = total_energy_wrapper(optimized_positions.flatten(), n_beads)
    print("Optimized Energy:", final_energy)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process (optional)
    animate_optimization(trajectory)
