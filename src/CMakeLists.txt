
FILE(GLOB_RECURSE RVOS_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/*/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.s"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.S"
    "${CMAKE_CURRENT_LIST_DIR}/../compiled/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/../compiled/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/../compiled/*.cpp"
    )

add_library(rvos STATIC ${RVOS_SRCS})
