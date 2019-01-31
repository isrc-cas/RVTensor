set(KENDRYTE_YOLOV3_NAME kendryte_yolov3)

FILE(GLOB_RECURSE KENDRYTE_YOLOV3_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/*.s"
    "${CMAKE_CURRENT_LIST_DIR}/*.S"
    )
message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_LIST_DIR})
link_directories(${SDK_LIBS_DIR}/bsp)
link_directories(${SDK_LIBS_DIR}/freertos)
link_directories(${SDK_LIBS_DIR}/drivers)
link_directories(${SDK_LIBS_DIR}/hal)

add_executable(${KENDRYTE_YOLOV3_NAME} ${KENDRYTE_YOLOV3_SRCS})
target_link_libraries(${KENDRYTE_YOLOV3_NAME} rvos
                -Wl,--start-group
                m freertos atomic bsp c stdc++ drivers hal
                -Wl,--end-group
                )
set_target_properties(${KENDRYTE_YOLOV3_NAME} PROPERTIES LINKER_LANGUAGE C)
