add_library(lazy LazySnapping.cpp SeedsRevised.cpp Tools.cpp watershedLabel.cpp)

target_include_directories(graphcut PUBLIC ../max_flow)