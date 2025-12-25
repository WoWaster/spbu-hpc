#! /usr/bin/env bash

MATRIX_SIZE=10000
BUILD_TYPE=Release
WARMUP_COUNT=3
ITER_COUNT=10

CORE_COUNT=$(nproc)

## Step 1. Build everything

echo "Building project" >&2

cmake . -B build/ -D CMAKE_BUILD_TYPE=$BUILD_TYPE >&2
cmake --build build/ >&2

echo "" >&2

## Step 2. Benchmark OpenMP

echo "Running benchmarks with OpenMP with n=${MATRIX_SIZE}" >&2
echo "OpenMP"

OMP_BINARY=./build/labA_01_omp

for ((i=1; i < CORE_COUNT; i=$((2 * i)))) do
    echo "Running with ${i} cores" >&2
    echo "n=$i"
    echo "Warming up..." >&2
    for ((j=0; j < WARMUP_COUNT; j++)) do
        OMP_NUM_THREADS=$i $OMP_BINARY $MATRIX_SIZE &> /dev/null
    done
    echo "Measuring..." >&2
    for ((j=0; j < ITER_COUNT; j++)) do
        OMP_NUM_THREADS=$i $OMP_BINARY $MATRIX_SIZE
    done
    echo "" >&2
done

echo "Running with ${CORE_COUNT} cores" >&2
echo "n=$CORE_COUNT"
echo "Warming up..." >&2
for ((j=0; j < WARMUP_COUNT; j++)) do
    OMP_NUM_THREADS=$CORE_COUNT $OMP_BINARY $MATRIX_SIZE &> /dev/null
done
echo "Measuring..." >&2
for ((j=0; j < ITER_COUNT; j++)) do
    OMP_NUM_THREADS=$CORE_COUNT $OMP_BINARY $MATRIX_SIZE
done
echo "" >&2
echo "" >&2

## Step 3. Benchmark MPI

echo "Running benchmarks with MPI with n=${MATRIX_SIZE}" >&2
echo "MPI"

MPI_BINARY=./build/labA_01_mpi

for ((i=1; i < CORE_COUNT; i=$((2 * i)))) do
    echo "Running with ${i} cores" >&2
    echo "n=$i"
    echo "Warming up..." >&2
    for ((j=0; j < WARMUP_COUNT; j++)) do
        mpirun -n $i $MPI_BINARY $MATRIX_SIZE &> /dev/null
    done
    echo "Measuring..." >&2
    for ((j=0; j < ITER_COUNT; j++)) do
        mpirun -n $i $MPI_BINARY $MATRIX_SIZE
    done
    echo "" >&2
done

# echo "Running with ${CORE_COUNT} cores" >&2
# echo "n=$CORE_COUNT"
# echo "Warming up..." >&2
# for ((j=0; j < WARMUP_COUNT; j++)) do
#     mpirun -n $i $MPI_BINARY $MATRIX_SIZE &> /dev/null
# done
# echo "Measuring..." >&2
# for ((j=0; j < ITER_COUNT; j++)) do
#     mpirun -n $i $MPI_BINARY $MATRIX_SIZE
# done
echo "" >&2
echo "" >&2
