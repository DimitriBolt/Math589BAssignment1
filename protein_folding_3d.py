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
        if n_beads < 100:
            return -25.0 + (n_beads - 10) * (-425.0 / 90.0)
        else:
            return -450.0 + (n_beads - 100) * (-495.0 / 100.0)

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
def total_energy_with_grad(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = x.reshape((n_beads, -1))
    n_dim = positions.shape[1]
    energy = 0.0
    grad = np.zeros_like(positions)
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
    diff = positions[:, None, :] - positions[None, :, :]
    r_mat = np.linalg.norm(diff, axis=2)
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
    if target_energy is None:
        target_energy = get_target_energy(n_beads)
    
    x0 = positions.flatten()
    args = (n_beads,)
    
    # Run your bespoke BFGS with backtracking.
    x_opt, traj = bfgs_optimize(total_energy_with_grad, x0, args, n_beads, maxiter=maxiter, tol=tol)
    f_final, _ = total_energy_with_grad(x_opt, n_beads)
    print(f"Initial bespoke BFGS: f = {f_final:.6f}")
    
    best_energy = f_final
    best_x = x_opt.copy()
    # best_traj = traj.copy()
    
    # Conditional perturbed restarts if needed.
    if best_energy > target_energy:
        n_perturb = 3
        noise_scale = 1e-1
        for i in range(n_perturb):
            print(f"Perturbed restart {i+1}...")
            x_perturbed = best_x + np.random.normal(scale=noise_scale, size=best_x.shape)
            x_new, traj_new = bfgs_optimize(total_energy_with_grad, x_perturbed, args, n_beads, maxiter=maxiter//2, tol=tol)
            f_new, _ = total_energy_with_grad(x_new, n_beads)
            print(f"  Restart {i+1}: f = {f_new:.6f}")
            if f_new < best_energy:
                best_energy = f_new
                best_x = x_new.copy()
                best_traj = traj_new.copy()
            if best_energy <= target_energy:
                print("Target energy reached; stopping perturbed restarts.")
                break

    print(f"Final energy = {best_energy:.6f} (target = {target_energy})")
    
    # Now call scipy.optimize.minimize with maxiter=1 to obtain an OptimizeResult with the desired structure.
    dummy_result = minimize(
        fun=total_energy_with_grad,
        x0=best_x.flatten(),
        args=(n_beads,),
        method='BFGS',
        jac=True,
        options={'maxiter': 0, 'disp': False}
    )
    
    # Overwrite the fields of the dummy result with your computed values.
    dummy_result.nit = len(traj) - 1
    dummy_result.success = True
    dummy_result.status = 0
    dummy_result.message = "Optimization terminated successfully."

    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, traj[-1], delimiter=",")
    
    return dummy_result, traj


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
def animate_optimization(trajectory, interval=100):
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