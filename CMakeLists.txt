# FOLDER: Root
set(CMAKE_C_COMPILER "gcc-11")
set(CMAKE_CXX_COMPILER "g++-11")      

cmake_minimum_required( VERSION 3.10 )

project(DNS_Simulution)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Wextra -Wpedantic")

find_package( Eigen3 REQUIRED )

add_subdirectory( libraries )

include_directories( ${CMAKE_SOURCE_DIR}/libraries/ )
add_executable( exec main.cpp )
target_link_libraries(exec Eigen3::Eigen)
