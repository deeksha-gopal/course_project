#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Linear algebra functions
double vector_norm(double *v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

void scalar_multiply(double *v, double a, int n) {
    for (int i = 0; i < n; i++) {
        v[i] *= a;
    }
}

void vector_addition(double *v, double *w, double a, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = v[i] + a * w[i];
    }
}

// Compute dot product between two vectors
double dot_product(double *v, double *w, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += v[i] * w[i];
    }
    return result;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming vectors x1, x2, ..., xr are of length n
    int n = 100;
    double **x = (double**)malloc(size * sizeof(double*));

    // Allocate memory for vectors
    for (int i = 0; i < size; i++) {
        x[i] = (double*)malloc(n * sizeof(double));
        // Initialize vectors
        for (int j = 0; j < n; j++) {
            x[i][j] = i + j + 1.0;
        }
    }

    // Gram-Schmidt orthogonalization
    for (int j = 0; j < size; j++) {
        // Compute rjj
        double rjj = vector_norm(x[j], n);

        // Check if rjj is zero, stop, else compute qj
        if (rjj != 0.0) {
            scalar_multiply(x[j], 1.0 / rjj, n);

            // Compute rij for i = 1 to j-1
            for (int i = 0; i < j; i++) {
                double rij = 0.0;
                for (int k = 0; k < n; k++) {
                    rij += x[j][k] * x[i][k];
                }
                vector_addition(x[j], x[i], -rij, n);
            }
        }
    }

    // Print the resulting orthogonal vectors on each process
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Process %d: Orthogonal vector (q%d): [", rank, i);
            for (int j = 0; j < n; j++) {
                printf("%.2f ", x[i][j]);
            }
            printf("]\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Check orthogonality by computing dot product between vectors
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            if (rank == i && k > i) {
                double dot_result = dot_product(x[i], x[k], n);
                printf("Process %d: Dot product between q%d and q%d: %.6f\n", rank, i, k, dot_result);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Clean up
    for (int i = 0; i < size; i++) {
        free(x[i]);
    }
    free(x);

    MPI_Finalize();
    return 0;
}

