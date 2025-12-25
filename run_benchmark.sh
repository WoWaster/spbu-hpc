#! /usr/bin/env bash

MATRIX_SIZE=10000
BUILD_TYPE=Release
WARMUP_COUNT=3
ITER_COUNT=10

CORE_COUNT=$(nproc)

## Step 1. Build everything

echo "Building project"

cmake . -B build/ -D CMAKE_BUILD_TYPE=$BUILD_TYPE
cmake --build build/

echo ""

## Step 2. Benchmark OpenMP

echo "Running benchmarks with OpenMP with n=${MATRIX_SIZE}"

OMP_BINARY=./build/labA_01_omp

for ((i=1; i < CORE_COUNT; i=$((2 * i)))) do
    echo "Running with ${i} cores"
    echo "Warming up..."
    for ((j=0; j < WARMUP_COUNT; j++)) do
        OMP_NUM_THREADS=$i $OMP_BINARY $MATRIX_SIZE &> /dev/null
    done
    echo "Measuring..."
    for ((j=0; j < ITER_COUNT; j++)) do
        OMP_NUM_THREADS=$i $OMP_BINARY $MATRIX_SIZE
    done
    echo ""
done

echo "Running with ${CORE_COUNT} cores"
echo "Warming up..."
for ((j=0; j < WARMUP_COUNT; j++)) do
    OMP_NUM_THREADS=$CORE_COUNT $OMP_BINARY $MATRIX_SIZE &> /dev/null
done
echo "Measuring..."
for ((j=0; j < ITER_COUNT; j++)) do
    OMP_NUM_THREADS=$CORE_COUNT $OMP_BINARY $MATRIX_SIZE
done
echo ""
echo ""

## Step 3. Benchmark MPI

echo "Running benchmarks with MPI with n=${MATRIX_SIZE}"

MPI_BINARY=./build/labA_01_mpi

for ((i=1; i < CORE_COUNT; i=$((2 * i)))) do
    echo "Running with ${i} cores"
    echo "Warming up..."
    for ((j=0; j < WARMUP_COUNT; j++)) do
        mpirun -n $i $MPI_BINARY $MATRIX_SIZE &> /dev/null
    done
    echo "Measuring..."
    for ((j=0; j < ITER_COUNT; j++)) do
        mpirun -n $i $MPI_BINARY $MATRIX_SIZE
    done
    echo ""
done

# echo "Running with ${CORE_COUNT} cores"
# echo "Warming up..."
# for ((j=0; j < WARMUP_COUNT; j++)) do
#     mpirun -n $i $MPI_BINARY $MATRIX_SIZE &> /dev/null
# done
# echo "Measuring..."
# for ((j=0; j < ITER_COUNT; j++)) do
#     mpirun -n $i $MPI_BINARY $MATRIX_SIZE
# done
echo ""
echo ""
