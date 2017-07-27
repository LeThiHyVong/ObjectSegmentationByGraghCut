cmake_minimum_required(VERSION 3.8)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(SOURCE_FILES main.cpp)

project(GraphCutSegmentation)

include(PrecompiledHeader.cmake)

add_library(target stdc++.h stdc++.cpp)
add_precompiled_header(target stdc++.h FORCEINCLUDE)

add_subdirectory(max_flow)
add_subdirectory(graphcut)
add_subdirectory(lazy)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(GraphCutSegmentation ${SOURCE_FILES})
target_link_libraries(GraphCutSegmentation ${OpenCV_LIBS} graphcut lazy maxflow)