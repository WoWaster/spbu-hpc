cmake_minimum_required(VERSION 3.25)

project(spbu-hpc CXX)

find_package(OpenMP COMPONENTS CXX)

add_executable(labA_01_omp labA_01_omp.cpp)
target_compile_options(labA_01_omp PRIVATE -Wall -Wextra -pedantic -Werror -fsanitize=address -fsanitize=leak -fsanitize=undefined)
target_link_options(labA_01_omp PRIVATE -fsanitize=address -fsanitize=leak -fsanitize=undefined)
target_link_libraries(labA_01_omp PRIVATE OpenMP::OpenMP_CXX)
