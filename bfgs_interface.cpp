// bfgs_interface.cpp (New File)

#include "bfgs_w_classes.cpp"
#include <vector>
#include <iostream>
#include <cstdlib>

extern "C" {

// Define a function to run BFGS optimization
void optimize_protein(double *positions, int n_beads, int maxiter, double tol) {
    std::vector<std::vector<double>> x(n_beads, std::vector<double>(3));

    // Convert flat C-array into a C++ 2D vector
    for (int i = 0; i < n_beads; ++i) {
        x[i][0] = positions[3 * i];
        x[i][1] = positions[3 * i + 1];
        x[i][2] = positions[3 * i + 2];
    }

    // Create BFGS optimizer instance
    BFGS optimizer;
    optimizer.optimize(x, maxiter, tol);

    // Copy optimized positions back into positions array
    for (int i = 0; i < n_beads; ++i) {
        positions[3 * i] = x[i][0];
        positions[3 * i + 1] = x[i][1];
        positions[3 * i + 2] = x[i][2];
    }
}

}
