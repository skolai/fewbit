add_executable("fewbit-cpu-codec-test" EXCLUDE_FROM_ALL)
set_target_properties("fewbit-cpu-codec-test" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)
target_sources("fewbit-cpu-codec-test" PRIVATE codec.cc codec.h codec_test.cc)

add_library("fewbit-cpu" OBJECT)
set_target_properties("fewbit-cpu" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries("fewbit-cpu" PUBLIC torch)
target_sources("fewbit-cpu"
    PRIVATE
        codec.cc
        gelu.cc
    PUBLIC
        codec.h
        gelu.h)
