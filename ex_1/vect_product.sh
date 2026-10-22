#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=compute2011
#SBATCH --exclusive
#SBATCH --output=filename.out
#SBATCH --error=error.err

module load mpi/openmpi/4.1.0-no_ucx

mpirun ./vect_product


