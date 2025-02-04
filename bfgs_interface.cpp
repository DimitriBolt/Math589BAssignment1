#include <vector>
#include <iostream>
#include <functional>
#include "bfgs_w_classes.cpp"  // Ensure this file contains the BFGS class

extern "C" {

// Function to optimize protein structure using BFGS
void optimize_protein(double *positions, int n_beads, int maxiter, double tol) {
    std::vector<std::vector<double> > x(n_beads, std::vector<double>(3));

    // Convert flat C-array into a 2D C++ vector
    for (int i = 0; i < n_beads; ++i) {
        x[i][0] = positions[3 * i];
        x[i][1] = positions[3 * i + 1];
        x[i][2] = positions[3 * i + 2];
    }

    // Define objective function (total energy)
    auto objective = [](const std::vector<double>& x_vec) -> double {
        return compute_energy(x_vec);  // Ensure compute_energy is implemented in bfgs_w_classes.cpp
    };

    // Define gradient function
    auto gradient = [](const std::vector<double>& x_vec) -> std::vector<double> {
        return compute_energy_gradient(x_vec);  // Ensure compute_energy_gradient exists in bfgs_w_classes.cpp
    };

    // Create BFGS optimizer instance
    BFGS optimizer(objective, gradient);
    std::vector<double> flat_x; 

    // Flatten the 2D vector for optimization
    for (const auto& bead : x) {
        flat_x.insert(flat_x.end(), bead.begin(), bead.end());
    }

    // Run optimization
    std::vector<double> optimized_x = optimizer.optimize(flat_x, maxiter, tol);

    // Copy results back into positions array
    for (int i = 0; i < n_beads; ++i) {
        positions[3 * i]     = optimized_x[3 * i];
        positions[3 * i + 1] = optimized_x[3 * i + 1];
        positions[3 * i + 2] = optimized_x[3 * i + 2];
    }
}

}
