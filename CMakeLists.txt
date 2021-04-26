# examples for omp test

#check version of cmake
cmake_minimum_required(VERSION 3.0)

project(omp_test)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMKAE_C_COMPILER "/usr/bin/gcc")
set(CMKAE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_FLAGS "-lstdc++")
set(CMAKE_BUILD_TYPE release)
add_definitions("-O2 -Wall")

if(${OMP} MATCHES "y")
	add_definitions(-DOMP_PARALLEL)
	FIND_PACKAGE( OpenMP REQUIRED)
	if(OPENMP_FOUND)
		message("OPENMP FOUND")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 
# Add OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

include_directories(./ commom/)
add_subdirectory(common)

set(TEST_MATH ${DIR_SRCS})
add_executable(${PROJECT_NAME} ${TEST_MATH})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} common)
install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin)
