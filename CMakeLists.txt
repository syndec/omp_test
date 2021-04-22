# examples for omp test

#check version of cmake
cmake_minimum_required(VERSION 3.0)

project(omp_test)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMKAE_C_COMPILER "/usr/bin/gcc")
set(CMKAE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE release)
add_definitions("-g -Wall")

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 
# Add OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)
set(TEST_MATH ${DIR_SRCS})
add_executable(${PROJECT_NAME} ${TEST_MATH})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin)
