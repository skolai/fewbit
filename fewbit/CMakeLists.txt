add_executable("fewbit-codec-test" EXCLUDE_FROM_ALL)
target_compile_features("fewbit-codec-test" PRIVATE cxx_std_20)
target_sources("fewbit-codec-test" PRIVATE codec.cc codec.h codec_test.cc)

add_library("fewbit" SHARED)
set_property(TARGET "fewbit" PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features("fewbit" PRIVATE cxx_std_20)
target_link_libraries("fewbit" PRIVATE torch)
target_sources("fewbit"
    PRIVATE
        codec.cc
        fewbit.cc
        torch_library.cc
    PUBLIC
        codec.h
        fewbit.h)

# NOTE CMake does not populate target with INTERFACE/PUBLIC headers but allows
# installation them with PUBLIC_HEADER destination. So, we populate
# PUBLIC_HEADER variable here manually.
get_target_property(PUBLIC_HEADER "fewbit" PUBLIC_HEADER)
if (NOT PUBLIC_HEADERS)
    get_target_property(PUBLIC_HEADERS "fewbit" INTERFACE_SOURCES)
    set_target_properties("fewbit" PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADER}")
endif()

if (USE_CUDA)
    add_library("fewbit-kernel" OBJECT)
    set_property(TARGET "fewbit-kernel"
                 PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries("fewbit-kernel" PUBLIC torch)
    target_sources("fewbit-kernel" PRIVATE gelu.cc gelu.cu gelu.h)
    target_compile_features("fewbit-kernel"
        PRIVATE cuda_std_17
        PRIVATE cxx_std_17)

    # If CUDA is enabled then we force CUDA C++ standard and link with new
    # compiled CUDA kernels.
    target_compile_features("fewbit" PRIVATE cuda_std_17)
    target_link_libraries("fewbit" PUBLIC fewbit-kernel)
endif()

# NOTE Install only shared object since we have no plans to use the library in
# other C++ code.
install(TARGETS "fewbit"
        LIBRARY DESTINATION "fewbit")
