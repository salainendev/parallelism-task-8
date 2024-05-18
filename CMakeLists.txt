cmake_minimum_required(VERSION 3.22)

project(Example_class VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER "nvc++")
set(MATRIX1D "main")

message(STATUS "Compile C++: " ${CMAKE_CXX_COMPILER})

find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED program_options)
add_executable(${MATRIX1D} "main.cu") 
target_compile_features(${MATRIX1D} PRIVATE cxx_std_20)
target_compile_options(${MATRIX1D} PRIVATE -arch=native)
target_link_libraries(${MATRIX1D} Boost::program_options)
target_include_directories(${MATRIX1D} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# add_executable(${MATRIX2D} "sum_matrix2D.cu") 
# target_compile_features(${MATRIX2D} PRIVATE cxx_std_20)
# target_compile_options(${MATRIX2D} PRIVATE -arch=native)
# target_include_directories(${MATRIX2D} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# add_executable(${sCUB} "cub_reduce_max.cu") 
# target_compile_features(${sCUB} PRIVATE cxx_std_20)
# target_compile_options(${sCUB} PRIVATE -arch=native)
# target_include_directories(${sCUB} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})