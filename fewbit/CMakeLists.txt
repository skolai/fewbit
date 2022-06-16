add_library("fewbit" SHARED)
set_target_properties("fewbit" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries("fewbit"
    PRIVATE fewbit-cpu
    PUBLIC torch)
target_sources("fewbit" PRIVATE fewbit.cc PUBLIC fewbit.h)

# NOTE CMake does not populate target with INTERFACE/PUBLIC headers but allows
# installation them with PUBLIC_HEADER destination. So, we populate
# PUBLIC_HEADER variable here manually.
get_target_property(PUBLIC_HEADER "fewbit" PUBLIC_HEADER)
if (NOT PUBLIC_HEADERS)
    get_target_property(PUBLIC_HEADERS "fewbit" INTERFACE_SOURCES)
    set_target_properties("fewbit" PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADER}")
endif()

# NOTE Install only shared object since we have no plans to use the library in
# other C++ code.
install(TARGETS "fewbit"
        LIBRARY DESTINATION "fewbit")

add_subdirectory(cpu)
add_subdirectory(cuda)
