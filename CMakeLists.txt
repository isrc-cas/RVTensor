set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

if(CMAKE_TOOLCHAIN_FILE)
    set(LIBRARY_OUTPUT_PATH_ROOT ${CMAKE_BINARY_DIR} CACHE PATH "root for library output, set this to change where android libs are compiled to")
    get_filename_component(CMAKE_TOOLCHAIN_FILE_NAME ${CMAKE_TOOLCHAIN_FILE} NAME)
    find_file(CMAKE_TOOLCHAIN_FILE ${CMAKE_TOOLCHAIN_FILE_NAME} PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
    message(STATUS "CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
endif()

cmake_minimum_required(VERSION 3.1.3)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

project(rvos)

option(RVOS_KENDRYTE "kendryte support" ON)
option(RVOS_GAP8     "gap8 support" OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src lib)
add_subdirectory(examples)
