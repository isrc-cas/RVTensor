set (CMAKE_SYSTEM_NAME "Generic")
set (KENDRYTE True)

set(SDK_LIBS_DIR "${SDK_DIR}/build/SDK")

include_directories(${SDK_DIR}/lib/arch/include)
include_directories(${SDK_DIR}/lib/utils/include)
include_directories(${SDK_DIR}/lib/freertos/include)
include_directories(${SDK_DIR}/lib/freertos/portable)
include_directories(${SDK_DIR}/lib/freertos/conf)

set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-g++)

set(CMAKE_TOOLCHAIN_PREFIX riscv64-unknown-elf-)

set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR}
    ${TOOLCHAIN_DIR}/riscv64-unknown-elf/include/c++/8.2.0
    ${TOOLCHAIN_DIR}/riscv64-unknown-elf/lib)

set(CMAKE_AR  ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-ar CACHE FILEPATH "Archiver")
set(CMAKE_LD  ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-ld CACHE string "linker")
set(CMAKE_NM  ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-nm CACHE string "nm")
set(CMAKE_STRIP ${TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-strip CACHE string "strip")
 
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NERVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
 
link_directories(/home/jiageng/kendryte/kendryte-gnu-toolchain/kendryte-toolchain/bin)
link_directories(${SDK_DIR}/usr/include)

add_compile_options(-ffunction-sections)
add_compile_options(-fdata-sections)
add_compile_options(-mcmodel=medany)
add_compile_options(-march=rv64imafdc)
add_compile_options(-fno-common)

set(CMAKE_CXX_FLAGS "-O2 -std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "-std=gnu11 ${CMAKE_C_FLAGS}")
