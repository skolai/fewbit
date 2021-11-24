add_executable("fewbit-codec-test" codec.h codec.cc codec_test.cc)

target_compile_features("fewbit-codec-test" PRIVATE cxx_std_20)

add_library("fewbit-static" STATIC)

set_property(TARGET "fewbit-static"
             PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_features("fewbit-static" PRIVATE cxx_std_20)

target_sources("fewbit-static"
    PRIVATE codec.cc
    PRIVATE fewbit.cc
    PUBLIC codec.h
    PUBLIC fewbit.h)

target_link_libraries("fewbit-static" PRIVATE torch)

add_library("fewbit" SHARED)

target_compile_features("fewbit" PRIVATE cxx_std_20)

target_sources("fewbit"
    PRIVATE torch_library.cc)

target_link_libraries("fewbit"
    PRIVATE fewbit-static
    PRIVATE torch)
