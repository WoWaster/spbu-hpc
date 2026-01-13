cmake_minimum_required(VERSION 3.25)

project(spbu-hpc CXX)

find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)

add_executable(labA_01 labA_01.cpp)
target_compile_options(labA_01 PRIVATE -Wall -Wextra -pedantic -Werror -g)

add_executable(labA_01_omp labA_01_omp.cpp)
target_compile_options(labA_01_omp PRIVATE -Wall -Wextra -pedantic -Werror -g)
target_link_libraries(labA_01_omp PRIVATE OpenMP::OpenMP_CXX)

add_executable(labA_01_mpi labA_01_mpi.cpp)
target_compile_options(labA_01_mpi PRIVATE -Wall -Wextra -pedantic -Werror -g)
target_link_libraries(labA_01_mpi PRIVATE MPI::MPI_CXX)
