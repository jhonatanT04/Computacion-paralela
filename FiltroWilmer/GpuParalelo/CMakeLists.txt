cmake_minimum_required(VERSION 3.16)
project(vision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your source files
add_executable(${PROJECT_NAME}
    Operaciones.cpp
    Principal.cpp
)
