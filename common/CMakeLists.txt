#check version of cmake
cmake_minimum_required(VERSION 3.0)

aux_source_directory(. DIR_COMMON_SRCS)
add_library(common ${DIR_COMMON_SRCS})
