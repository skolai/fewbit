add_library("fewbit" SHARED)

target_compile_features("fewbit" PRIVATE cxx_std_20)

target_sources("fewbit"
    PUBLIC fewbit.h
    PRIVATE fewbit.cc
    PRIVATE torch_library.cc)

target_link_libraries("fewbit" PRIVATE torch)
