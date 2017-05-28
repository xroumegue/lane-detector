cmake_minimum_required(VERSION 3.7)
project(lane_vx)
include_directories(include)
find_package(OpenCV REQUIRED)
find_package(OpenVX REQUIRED)
include_directories(${OpenVX_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenVX_LIBS_DIR}) 
set(lane_SRCS lane.cpp)
add_executable(lane ${lane_SRCS})
target_link_libraries(lane ${OpenVX_LIBRARIES} ${OpenCV_LIBRARIES})


set(CMAKE_VERBOSE_MAKEFILE TRUE)
