# Math589BAssignment1

The topic is Protein Folding. A protein is a chain of beads. The bond between consecutive beads is spring-like. In addition, each two beads
attract each other when they are far apart, and repel each other when they are near, according to the Lennard-Jones potential.

## Report of Findings ##

This project tackles the problem of **Protein Folding**. The project models the bonds between consecutive beads
as spring-like and incorporates interactions between beads based on the **Lennard-Jones potential**, which governs their
attraction and repulsion depending on the distance.

The provided solution to this problem is implemented in the Python script *protein_folding_3d.py*. The solution aims to
minimize the total energy of the protein defined by the Lennard-Jones potential, consequently finding the optimal 3D
configuration of the protein chain.

### Key Functions in the Solution

#### 1. **`optimize_protein`**

This is the core function for solving the protein folding problem. It uses optimization techniques to determine the 3D
coordinates of the beads in the protein chain, such that the total energy of the system is minimized. This function
leverages the **gradient-based optimization methods**, ensuring efficient convergence to a low-energy configuration. The
specific optimization algorithm and its parameters are carefully chosen to balance speed and accuracy, making it
suitable for complex protein structures.

#### 2. **`total_energy_with_grad`**

This function computes the total energy of the protein and its gradient with respect to the positions of the beads. The
use of the energy gradient is critical because it provides the direction and magnitude of change needed to reduce the
energy efficiently. By evaluating the gradient of the Lennard-Jones potential, the optimization process systematically
adjusts the bead positions, ensuring faster convergence and better accuracy in finding the global energy minimum.

### Visualization of the Solution

The dynamic solution process and resulting 3D structure are illustrated in the animation below:

![Protein Animation](protein_animation.gif)

This animation visualizes how the protein chain evolves during the optimization process, eventually settling into its
stable, low-energy configuration.

---

This project demonstrates how computational techniques and optimization algorithms can address complex problems in
applied mathematics and biology.

