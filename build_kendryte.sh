#!/bin/bash

##### KENDRYTE
if [ -d "build-kendryte-freertos" ];then
    rm -fr build-kendryte-freertos
fi
toolchain_dir="/home/jiageng/kendryte/kendryte-gnu-toolchain/kendryte-toolchain"
sdk_dir="/home/jiageng/kendryte/kendryte-freertos-sdk"
mkdir -p build-kendryte-freertos
pushd build-kendryte-freertos
cmake -DCMAKE_TOOLCHAIN_FILE=../environments/kendryte.toolchain.cmake\
      -DTOOLCHAIN_DIR=$toolchain_dir -DSDK_DIR=$sdk_dir ..
make VERBOSE=1
popd
