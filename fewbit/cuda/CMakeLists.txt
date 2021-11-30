# Guard condition: Add targets only if we are forced to use CUDA.
if (NOT USE_CUDA)
    return()
endif()

add_executable("fewbit-cuda-test" EXCLUDE_FROM_ALL)
target_compile_features("fewbit-cuda-test" PRIVATE cxx_std_17 cuda_std_17)
target_sources("fewbit-cuda-test" PRIVATE codec.cu codec.h codec_test.cu)

add_library("fewbit-cuda" STATIC)

target_sources("fewbit-cuda"
    PRIVATE
        codec.cu
    PUBLIC
        codec.h)

set_target_properties("fewbit-cuda" PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      POSITION_INDEPENDENT_CODE ON)
