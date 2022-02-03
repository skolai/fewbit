cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("fewbit" VERSION 0.0.0 LANGUAGES CXX)

option(USE_CUDA "Build CUDA kernels" OFF)
if (USE_CUDA)
    enable_language(CUDA)
endif ()

set(TORCH_CUDA_ARCH_LIST "Auto" CACHE STRING "Target CUDA archetecture" FORCE)

find_package(Torch REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Force to display colorised error messages.
if (CMAKE_GENERATOR STREQUAL "Ninja")
    add_compile_options(-fdiagnostics-color=always)
endif()

add_subdirectory(fewbit)
