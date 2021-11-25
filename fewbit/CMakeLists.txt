add_executable("fewbit-codec-test" codec.h codec.cc codec_test.cc)
target_compile_features("fewbit-codec-test" PRIVATE cxx_std_20)

add_library("fewbit" SHARED)
set_property(TARGET "fewbit" PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features("fewbit"
    PRIVATE cuda_std_17
    PRIVATE cxx_std_20)
target_sources("fewbit"
    PRIVATE codec.cc
    PRIVATE fewbit.cc
    PRIVATE torch_library.cc
    PUBLIC codec.h
    PUBLIC fewbit.h)
target_link_libraries("fewbit" PRIVATE torch)

if (USE_CUDA)
    add_library("fewbit-kernel" OBJECT)
    set_property(TARGET "fewbit-kernel"
                 PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries("fewbit-kernel" PUBLIC torch)
    target_sources("fewbit-kernel"
        PRIVATE gelu.cc
        PRIVATE gelu.cu
        PUBLIC gelu.h)
    target_compile_features("fewbit-kernel"
        PRIVATE cuda_std_17
        PRIVATE cxx_std_17)
    target_link_libraries("fewbit" PUBLIC fewbit-kernel)
endif()
