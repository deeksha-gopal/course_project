#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define lx 100
#define ly 100
#define N (lx * ly)

double alpha = 1.0;
double beta = 1.0;
double gamma_value = 1.0;
double eta = 1.0;
double h = 1.0;
double residual_tolerance = 1e-6; 

// Structure to represent processor information
struct Processor {
    double* vector; // Local vector
    double* result; // Local result vector
    double* buffer; // Buffer for communication

    int rank;      // Processor rank
    int px, py;    // Coordinates in the 2D grid
    int forward_x; // Rank of the forward processor in the x-direction
    int backward_x; // Rank of the backward processor in the x-direction
    int forward_y; // Rank of the forward processor in the y-direction
    int backward_y; // Rank of the backward processor in the y-direction
};

// Function to initialize the processor structure
void initializeProcessor(struct Processor* proc, int rank, int px, int py, int npx, int npy) {

    proc->rank = rank;
    proc->px = px;
    proc->py = py;
    proc->forward_x = (px < npx - 1) ? rank + 1 : MPI_PROC_NULL;
    proc->backward_x = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    proc->forward_y = (py < npy - 1) ? rank + npx : MPI_PROC_NULL;
    proc->backward_y = (py > 0) ? rank - npx : MPI_PROC_NULL;

    // Initialize local vectors and buffers 
    proc->vector = (double*)malloc(N * sizeof(double));
    proc->result = (double*)malloc(N * sizeof(double));
    proc->buffer = (double*)malloc(N * sizeof(double));
}

// Function to exchange boundary fields in the x-direction
void exchangeBoundaryX(struct Processor* proc) {
    MPI_Sendrecv(proc->vector + (ly - 1) * lx, lx, MPI_DOUBLE, proc->forward_x, 0,
                 proc->buffer, lx, MPI_DOUBLE, proc->backward_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(proc->vector, lx, MPI_DOUBLE, proc->backward_x, 0,
                 proc->buffer + (ly - 1) * lx, lx, MPI_DOUBLE, proc->forward_x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Function to exchange boundary fields in the y-direction
void exchangeBoundaryY(struct Processor* proc) {
    // Send bottom row to forward processor and receive from backward processor
    MPI_Sendrecv(proc->vector + (lx - 1), 1, MPI_DOUBLE, proc->forward_y, 0,
                 proc->buffer, 1, MPI_DOUBLE, proc->backward_y, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send top row to backward processor and receive from forward processor
    MPI_Sendrecv(proc->vector, 1, MPI_DOUBLE, proc->backward_y, 0,
                 proc->buffer + (lx - 1), 1, MPI_DOUBLE, proc->forward_y, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Full matrix-vector multiplication
void fullMatrixVectorMultiply(struct Processor* proc) {
    // Using the provided stencil expressions for A
    for (int i = 0; i < N; i++) {
        proc->result[i] = (1 + 2 * h * (1 + eta)) * proc->vector[i] +
                           2 * alpha * h * (1 + eta) * gamma_value * (proc->vector[i - lx] + proc->vector[i - 1]) -
                           eta * proc->vector[i] + eta * proc->vector[i] +
                           alpha * h * h * proc->vector[i] * proc->vector[i];
    }
}

// Lower diagonal matrix-vector multiplication
void lowerDiagonalMatrixVectorMultiply(struct Processor* proc) {
    // Using the provided stencil expressions for F
    for (int i = 0; i < N; i++) {
        proc->result[i] = -2 * h * (1 + eta) * gamma_value * (proc->vector[i - lx] + proc->vector[i - 1]) -
                           eta * proc->vector[i];
    }
}

// Upper diagonal matrix-vector multiplication
void upperDiagonalMatrixVectorMultiply(struct Processor* proc) {
    // Using the provided stencil expressions for E
    for (int i = 0; i < N; i++) {
        proc->result[i] = -2 * h * (1 + eta) * gamma_value * (proc->vector[i - 1] - proc->vector[i - lx]) -
                           eta * proc->vector[i];
    }
}

// Function to perform Jacobi iteration with convergence check
void jacobiIterationConvergenceCheck(struct Processor* proc, double* b, int max_iter) {
    double* x_new = (double*)malloc(N * sizeof(double));
    double* Ax = (double*)malloc(N * sizeof(double));
    // Check for convergence
    double residual_norm = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        lowerDiagonalMatrixVectorMultiply(proc);
        upperDiagonalMatrixVectorMultiply(proc);

        // Jacobi update
        for (int i = 0; i < N; i++) {
            x_new[i] = (proc->buffer[i] + b[i] - proc->result[i]) / (1 + 2 * h * (1 + eta));
        }

        
        for (int i = 0; i < N; i++) {
            double diff = fabs(x_new[i] - proc->buffer[i]);
            residual_norm += diff * diff;
        }
        residual_norm = sqrt(residual_norm);

        // Update buffer for the next iteration
        for (int i = 0; i < N; i++) {
            proc->buffer[i] = x_new[i];
        }
        // Check for convergence
        if (residual_norm < 1e-6) {
            //printf("Converged after %d iterations.\n", iter +1);
        }
    }

    free(x_new);
    free(Ax);
}

// Function to perform steepest descent iteration with convergence check
void steepestDescentIteration(struct Processor* proc, double* b, int max_iter) {
    double* r = (double*)malloc(N * sizeof(double));
    double* p = (double*)malloc(N * sizeof(double));
    double* Ap = (double*)malloc(N * sizeof(double));

    // Initialize r = b - Ax and p = r
    fullMatrixVectorMultiply(proc); // A · v
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - proc->result[i];
        p[i] = r[i];
    }

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute Ap using the matrix-vector multiplication function
        fullMatrixVectorMultiply(proc); // A · v
        for (int i = 0; i < N; i++) {
            Ap[i] = proc->result[i];
        }

        // Compute alpha
        double rr = 0.0;
        double pAp = 0.0;
        for (int i = 0; i < N; i++) {
            rr += r[i] * r[i];
            pAp += p[i] * Ap[i];
        }

        double alpha;

        // Check if the matrix A is symmetric positive definite
        if (pAp > 0.0) {
            alpha = rr / pAp;
        } else {
            // Handle the case where A is not symmetric positive definite
            // set alpha to 0 and break the iteration
            alpha = 0.0;
            break;
        }

        // Update solution x and residual r
        for (int i = 0; i < N; i++) {
            proc->vector[i] = proc->vector[i] + alpha * p[i];
            r[i] = r[i] - alpha * Ap[i];
        }

        // Check for convergence
        double residual_norm = sqrt(rr);
        if (residual_norm < residual_tolerance) {
            // Converged
            break;
        }

        // Compute beta
        double rr_new = 0.0;
        for (int i = 0; i < N; i++) {
            rr_new += r[i] * r[i];
        }

        double beta = rr_new / rr;

        // Update search direction p
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    free(r);
    free(p);
    free(Ap);
}

// Function to update boundary conditions with Periodic Boundary Conditions
void update_bnd(double* vector, struct Processor* proc) {
    int i = 0;
    for (int j = 0; j < lx; j++) {
        int index = i * lx + j;

        double aij = vector[index];
        double aip1j = vector[((i + 1) % ly) * lx + j]; // Periodic boundary handling
        double aim1j = proc->buffer[j]; // Periodic boundary handling
        double aijp1 = vector[i * lx + (j + 1) % lx]; // Periodic boundary handling
        double aijm1 = vector[i * lx + (j - 1 + lx) % lx]; // Periodic boundary handling
    }

    i = ly - 1;
    for (int j = 0; j < lx; j++) {
        int index = i * lx + j;

        double aij = vector[index];
        double aip1j = proc->buffer[j]; // Periodic boundary handling
        double aim1j = vector[((i - 1 + ly) % ly) * lx + j]; // Periodic boundary handling
        double aijp1 = vector[i * lx + (j + 1) % lx]; // Periodic boundary handling
        double aijm1 = vector[i * lx + (j - 1 + lx) % lx]; // Periodic boundary handling
    }
}

// Function to free allocated memory
void freeProcessor(struct Processor* proc) {
    free(proc->vector);
    free(proc->result);
    free(proc->buffer);
}

// Function to perform weak scaling test
void weakScalingTest(int rank, int npx, int npy, int matrixSize) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and matrix-vector multiplication
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        fullMatrixVectorMultiply(&proc); // A · v
      
        lowerDiagonalMatrixVectorMultiply(&proc); // F · v

        upperDiagonalMatrixVectorMultiply(&proc); // E · v

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Weak Scaling Test: Matrix Size %dx%d, Processors %dx%d, Execution Time: %lf seconds\n",
           matrixSize, matrixSize, npx, npy, end_time - start_time);

    freeProcessor(&proc);
}

// Function to perform strong scaling test
void strongScalingTest(int rank, int npx, int npy, int baseMatrixSize, int totalProcessors) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and matrix-vector multiplication
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        fullMatrixVectorMultiply(&proc); // A · v
        
        lowerDiagonalMatrixVectorMultiply(&proc); // F · v

        upperDiagonalMatrixVectorMultiply(&proc); // E · v

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Strong Scaling Test: Matrix Size %dx%d, Processors %dx%d (out of %d), Execution Time: %lf seconds\n",
           baseMatrixSize, baseMatrixSize, npx, npy, totalProcessors, end_time - start_time);

    freeProcessor(&proc);
}

// Function to perform weak scaling test for Jacobi solver
void weakScalingJacobi(int rank, int npx, int npy, int matrixSize) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    // Define the right-hand side vector b
    double b[N];
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
    }
    b[2] = 1.0;

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and Jacobi iterations
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        jacobiIterationConvergenceCheck(&proc, b, num_steps);

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Weak Scaling Jacobi Test: Matrix Size %dx%d, Processors %dx%d, Execution Time: %lf seconds\n",
           matrixSize, matrixSize, npx, npy, end_time - start_time);

    freeProcessor(&proc);
}

// Function to perform strong scaling test for Jacobi solver
void strongScalingJacobi(int rank, int npx, int npy, int baseMatrixSize, int totalProcessors) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    // Define the right-hand side vector b
    double b[N];
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
    }
    b[2] = 1.0;

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and Jacobi iterations
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        jacobiIterationConvergenceCheck(&proc, b, num_steps);

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Strong Scaling Jacobi Test: Matrix Size %dx%d, Processors %dx%d (out of %d), Execution Time: %lf seconds\n",
           baseMatrixSize, baseMatrixSize, npx, npy, totalProcessors, end_time - start_time);

    freeProcessor(&proc);
}

// Function to perform weak scaling test for steepest descent solver
void weakScalingSteepestDescent(int rank, int npx, int npy, int matrixSize) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    // Define the right-hand side vector b
    double b[N];
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
    }
    b[2] = 1.0;

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and steepest descent iterations
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        steepestDescentIteration(&proc, b, num_steps);

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Weak Scaling Steepest Descent Test: Matrix Size %dx%d, Processors %dx%d, Execution Time: %lf seconds\n",
           matrixSize, matrixSize, npx, npy, end_time - start_time);

    freeProcessor(&proc);
}

// Function to perform strong scaling test for steepest descent solver
void strongScalingSteepestDescent(int rank, int npx, int npy, int baseMatrixSize, int totalProcessors) {
    struct Processor proc;
    int px = rank % npx;
    int py = rank / npx;

    initializeProcessor(&proc, rank, px, py, npx, npy);

    // Define the right-hand side vector b
    double b[N];
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
    }
    b[2] = 1.0;

    double start_time, end_time;
    start_time = MPI_Wtime();

    // Perform communication and steepest descent iterations
    int num_steps = 100;  
    for (int step = 0; step < num_steps; step++) {
        exchangeBoundaryX(&proc);
        exchangeBoundaryY(&proc);

        steepestDescentIteration(&proc, b, num_steps);

        // Update boundary conditions again after communication
        update_bnd(proc.vector, &proc);
    }

    end_time = MPI_Wtime();

    printf("Strong Scaling Steepest Descent Test: Matrix Size %dx%d, Processors %dx%d (out of %d), Execution Time: %lf seconds\n",
           baseMatrixSize, baseMatrixSize, npx, npy, totalProcessors, end_time - start_time);

    freeProcessor(&proc);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming a simple block distribution of processors
    int npx = 2;
    int npy = 2;

    // Perform weak scaling test for different matrix sizes
    for (int matrixSize = 100; matrixSize <= 400; matrixSize += 10) {
        weakScalingTest(rank, npx, npy, matrixSize);
    }

    // Perform strong scaling test for different numbers of processors
    int baseMatrixSize = 400;  
    for (int processors = 1; processors <= 4; processors++) {
        strongScalingTest(rank, npx, npy, baseMatrixSize, processors);
        
    }
    
    // Perform weak scaling test for different matrix sizes
    for (int matrixSize = 100; matrixSize <= 400; matrixSize += 10) {
        weakScalingJacobi(rank, npx, npy, matrixSize);
    }

    // Perform strong scaling test for different numbers of processors
    for (int processors = 1; processors <= 4; processors++) {
        strongScalingJacobi(rank, npx, npy, baseMatrixSize, processors);
    }

    // Perform weak scaling test for steepest descent solver
    for (int matrixSize = 100; matrixSize <= 400; matrixSize += 10) {
        weakScalingSteepestDescent(rank, npx, npy, matrixSize);
    }

    // Perform strong scaling test for steepest descent solver
    for (int processors = 1; processors <= 4; processors++) {
        strongScalingSteepestDescent(rank, npx, npy, baseMatrixSize, processors);
    }

    MPI_Finalize();
    return 0;
}

