cmake_minimum_required(VERSION 3.0)
project(sriid)

find_package(OpenCV REQUIRED)

add_executable(sriid main.cpp)
target_link_libraries(sriid ${OpenCV_LIBRARIES})
set_target_properties(sriid PROPERTIES DEBUG_POSTFIX _d)
