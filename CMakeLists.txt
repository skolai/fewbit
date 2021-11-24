cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project("few-bit-backward" VERSION 0.0.0 LANGUAGES CXX)

find_package(Torch REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(fewbit)
