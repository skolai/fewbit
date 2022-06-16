# Guard condition: Add targets only if we are forced to use CUDA.
if (NOT USE_CUDA)
    return()
endif()

add_executable("fewbit-cuda-test" EXCLUDE_FROM_ALL)
set_target_properties("fewbit-cuda-test" PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_sources("fewbit-cuda-test" PRIVATE codec.cu codec.h codec_test.cu)

add_library("fewbit-cuda" OBJECT)
set_target_properties("fewbit-cuda" PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries("fewbit-cuda" PUBLIC torch)

target_sources("fewbit-cuda"
    PRIVATE
        activation.cc
        codec.cu
    PUBLIC
        activation.h
        codec.h)


# If CUDA is enabled then we force CUDA C++ standard and link primary target
# with new compiled CUDA kernels.
set_target_properties("fewbit" PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON)
target_link_libraries("fewbit" PUBLIC fewbit-cuda)
