cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("fewbit" VERSION 0.0.0 LANGUAGES CXX)

option(USE_CUDA "Build CUDA kernels" OFF)
if (USE_CUDA)
    enable_language(CUDA)
endif ()

set(TORCH_CUDA_ARCH_LIST "Common" CACHE STRING "Target CUDA archetecture")

# Set common library dependencies.
find_package(Torch REQUIRED)

message(STATUS "FewBit: Torch version detected: ${Torch_VERSION}")

# There is an issue somewhere in either CMake scripts of Torch or CMake itself.
# The issue is that some CUDA flags are not copied. So, we copy flags manually
# for Torch 1.10.
if (Torch_VERSION VERSION_LESS "1.10")
    message(FATAL_ERROR "Torch version lesser than 1.10 is not supported.")
elseif (Torch_VERSION VERSION_LESS "1.11")
    foreach(FLAG ${CUDA_NVCC_FLAGS})
        string(FIND "${FLAG}" " " flag_space_position)
        if(NOT flag_space_position EQUAL -1)
            message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
        endif()
        string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
    endforeach()
endif()

# Set common C++ standard requirements.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Force to display colorised error messages.
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_compile_options(-fdiagnostics-color=always)
endif()

add_subdirectory(fewbit)
