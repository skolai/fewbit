add_executable("fewbit-cpu-codec-test" EXCLUDE_FROM_ALL)
target_compile_features("fewbit-cpu-codec-test" PRIVATE cxx_std_20)
target_sources("fewbit-cpu-codec-test" PRIVATE codec.cc codec.h codec_test.cc)

add_library("fewbit-cpu" OBJECT)
set_target_properties("fewbit-cpu" PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)
target_compile_features("fewbit-cpu" PRIVATE cxx_std_20)
target_link_libraries("fewbit-cpu" PUBLIC torch)
target_sources("fewbit-cpu"
    PRIVATE
        codec.cc
        gelu.cc
    PUBLIC
        codec.h
        gelu.h)
