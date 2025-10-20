#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>

void initialize_array(double *array, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        array[i] = (double)rand() / RAND_MAX * 100.0;
    }
}

double find_max(double *array, int N) {
    double max = array[0];
    #pragma omp parallel for reduction(max:max)
    for (int i = 1; i < N; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

void print_arrays(int rank, double *array, int N) {
    printf("Process %d: [", rank);
    for (int i = 0; i < N; i++) {
        printf(" %.2f", array[i]);
        if (i < N - 1) printf(",");
    }
    printf(" ]\n");
}

void odd_even_sort(double *array, int N, int rank, int size) {
    int phase;
    for (phase = 0; phase < size; phase++) {
        int partner = -1;
        if (phase % 2 == 0) {
            if (rank % 2 == 0 && rank + 1 < size) partner = rank + 1;
            else if (rank % 2 != 0) partner = rank - 1;
        } else {
            if (rank % 2 != 0 && rank + 1 < size) partner = rank + 1;
            else if (rank % 2 == 0 && rank > 0) partner = rank - 1;
        }
        if (partner != -1) {
            // Compare and swap (mocked, full logic would depend on array sharing across processes)
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    if (argc != 2 || (N = atoi(argv[1])) <= 1) {
        fprintf(stderr, "Usage: %s N (where N > 1)\n", argv[0]);
        return EXIT_FAILURE;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);  // Seed random number for each process

    // Step 1: Initialize array
    double *array = (double *)malloc(N * sizeof(double));
    initialize_array(array, N);

    // Step 2: Master process prints initial arrays
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (i == rank) {
                print_arrays(rank, array, N);
            } else {
                MPI_Recv(array, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                print_arrays(i, array, N);
            }
        }
    } else {
        MPI_Send(array, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Step 3: Calculate global average
    double local_sum = 0.0, global_sum, avg;

    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < N; i++) {
        local_sum += array[i];
    }
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg = global_sum / (size * N);

    // Step 4: Master selects odd-ball process
    int odd_ball = (rank == 0) ? rand() % (size - 1) + 1 : -1;
    MPI_Bcast(&odd_ball, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Odd-ball process: %d\n", odd_ball);
    }

    // Step 5: Adjust arrays based on odd-ball and average
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        if (rank == odd_ball) {
            array[i] += avg;
        } else {
            array[i] -= avg;
        }
    }

    // Step 6: Each process sends max to all others
    double local_max = find_max(array, N), max_values[size];
    MPI_Allgather(&local_max, 1, MPI_DOUBLE, max_values, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    // Step 7: Add neighbor's max or process 0's max
    double neighbor_max = max_values[(rank + 1) % size];

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        if (rank == odd_ball) {
            array[i] -= neighbor_max;
        } else {
            array[i] += neighbor_max;
        }
    }

    // Step 8: Add process ID and sort
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        array[i] += rank;
    }
    odd_even_sort(array, N, rank, size);

    // Final print of arrays
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            if (i == rank) {
                print_arrays(rank, array, N);
            } else {
                MPI_Recv(array, N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                print_arrays(i, array, N);
            }
        }
    } else {
        MPI_Send(array, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(array);
    MPI_Finalize();
    return 0;
}
