# Guard condition: Add targets only if we are forced to use CUDA.
if (NOT USE_CUDA)
    return()
endif()

add_executable("fewbit-cuda-test" EXCLUDE_FROM_ALL)
target_compile_features("fewbit-cuda-test" PRIVATE cxx_std_17 cuda_std_17)
target_sources("fewbit-cuda-test" PRIVATE codec.cu codec.h codec_test.cu)

add_library("fewbit-cuda" OBJECT)
set_target_properties("fewbit-cuda" PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)
target_compile_features("fewbit-cuda"
    PRIVATE cuda_std_17
    PRIVATE cxx_std_17)
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
target_compile_features("fewbit" PRIVATE cuda_std_17)
target_link_libraries("fewbit" PUBLIC fewbit-cuda)
