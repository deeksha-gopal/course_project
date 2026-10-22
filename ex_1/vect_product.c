#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assuming vectors v and w are of length n
    int n = 100; // Adjust the size as needed
    double *v = (double*)malloc(n * sizeof(double));
    double *w = (double*)malloc(n * sizeof(double));

    // Initialize vectors v and w (you can replace this with your data)
    for (int i = 0; i < n; i++) {
        v[i] = i + 1;
        w[i] = 2 * i + 1;
    }

    // Compute local sum
    double local_sum = 0.0;
    for (int i = 0; i < n; i++) {
        local_sum += v[i] * w[i];
    }

    // Reduce local sums to get the global sum
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Print result on all processes
    printf("Result of vector product on process %d: %.2f\n", rank, global_sum);

    //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Use MPI Reduce to collect the final result on process 0

    
    //if (rank == 0) {
        //printf("The dot product is: %f\n", global_sum);
        // Print the result on process 0
    //}

    // Clean up
    free(v);
    free(w);

    MPI_Finalize();
    return 0;
}

