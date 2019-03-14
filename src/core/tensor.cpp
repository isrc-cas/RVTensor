/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <cassert>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <memory>
#include "include/core/tensor.hpp"

namespace rvos {

////////////////////// Tensor ////////////////////////////////
Tensor::sptr Tensor::create() {
    return std::make_shared<Tensor>();
}

Tensor::sptr Tensor::create(int n, int c, int h, int w, size_t elemsize) {
    return std::make_shared<Tensor>(n, c, h, w, elemsize);
}

Tensor::sptr Tensor::create(int n, int c, int h, int w,
                            void* data, size_t elemsize) {
    return std::make_shared<Tensor>(n, c, h, w, data, elemsize);
}

inline Tensor::Tensor() : data_ptr(nullptr), element_size(0), n_batch(0),
                          width(0), height(0), channel(0), cstep(0) {}

inline Tensor::Tensor(int n, int c, int h, int w, size_t elemsize)
    : data_ptr(nullptr), element_size(elemsize), n_batch(n), width(w),
                                                 height(h), channel(c) {
    cstep = (channel <= 1) ? width * height :
                             alignSize(width * height * element_size,
                                       MALLOC_ALIGN) / element_size;
}

inline Tensor::Tensor(int n, int c, int h, int w, void* data, size_t elemsize)
    : data_ptr(data), element_size(elemsize), n_batch(n), width(w),
                                                 height(h), channel(c) {
    cstep = (channel <= 1) ? width * height :
                             alignSize(width * height * element_size,
                                       MALLOC_ALIGN) / element_size;
}

inline Tensor::~Tensor() {
    element_size = 0;
    n_batch = 0;
    width = 0;
    height = 0;
    channel = 0;
    cstep = 0;
}

inline bool Tensor::empty() const {
    return data_ptr == nullptr || total() == 0;
}

inline size_t Tensor::total() const {
    return cstep * channel * n_batch;
}

inline size_t Tensor::count() const {
    return n_batch * channel * height * width;
}

inline void Tensor::setQuantizeParams(
               float min, float max, int64_t sc, float zero) {
    min_range = min;
    max_range = max;
    scale = sc;
    zero_point = zero;
}

inline void Tensor::setQuantizeRange(float min, float max) {
    min_range = min;
    max_range = max;
}

inline void Tensor::setQuantizer(int64_t sc, float zero) {
    scale = sc;
    zero_point = zero;
}

/////////////////// FlashTensor /////////////////////////////
FlashTensor::sptr FlashTensor::create() {
    return std::make_shared<FlashTensor>();
}

FlashTensor::sptr FlashTensor::create(int n, int c, int h, int w,
                                                           size_t elemsize) {
    return std::make_shared<FlashTensor>(n, c, h, w, elemsize);
}

FlashTensor::sptr FlashTensor::create(int n, int c, int h, int w, void* data,
                                                           size_t elemsize) {
    return std::make_shared<FlashTensor>(n, c, h, w, data, elemsize);
}

inline FlashTensor::FlashTensor() : Tensor() {}

inline FlashTensor::FlashTensor(int n, int c, int h, int w, size_t elemsize)
                   : Tensor(n, c, h, w, elemsize) {}

inline FlashTensor::FlashTensor(int n, int c, int h, int w,
                                void* data, size_t elemsize)
                   : Tensor(n, c, h, w, data, elemsize) {}

inline FlashTensor::~FlashTensor() {
    data_ptr = nullptr;
}

inline void FlashTensor::bindData(void* data, size_t size) {
    if (size == total() && data_ptr == nullptr)
        data_ptr = data;
    else
        throw std::runtime_error("FlashTensor duplicate copy of data_ptr!");
}

template <typename T>
inline const T* FlashTensor::grepBatchData(int n) const {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<const T*>(data_ptr) +
           n * channel * cstep * element_size;
}

template <typename T>
inline const T* FlashTensor::grepChannelData(int n, int c) const {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<const T*>(data_ptr) +
           (n * channel * cstep + c * cstep) * element_size;
}

template <typename T>
inline const T* FlashTensor::grepRowData(int n, int c, int h) const {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<const T*>(data_ptr) +
           (n * channel * cstep + c * cstep + h * width) * element_size;
}

template <typename T>
inline const T FlashTensor::grepElement(int n, int c, int h, int w) const {
    assert(sizeof(T) == element_size);
    return *(reinterpret_cast<const T*>(data_ptr) +
           (n * channel * cstep + c * cstep + h * width + w) * element_size);
}

template <typename T>
inline FlashTensor::operator const T*() const {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<const T*>(data_ptr);
}

////////////////// RamTensor ////////////////////////////////
RamTensor::sptr RamTensor::create() {
    return std::make_shared<RamTensor>();
}

RamTensor::sptr RamTensor::create(int n, int c, int h, int w, size_t elemsize) {
    return std::make_shared<RamTensor>(n, c, h, w, elemsize);
}

RamTensor::sptr RamTensor::create(int n, int c, int h, int w,
                                  void* data, size_t elemsize) {
    return std::make_shared<RamTensor>(n, c, h, w, data, elemsize);
}


inline RamTensor::RamTensor() : Tensor() {}

inline RamTensor::RamTensor(int n, int c, int h, int w, size_t elemsize)
                   : Tensor(n, c, h, w, elemsize) {
    cstep = (c <= 1) ? width * height :
                     alignSize(width * height * element_size,
                               MALLOC_ALIGN) / element_size;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data_ptr = tensorDataMalloc(totalsize);
    }
}

inline RamTensor::RamTensor(int n, int c, int h, int w,
                            void* data, size_t elemsize)
                   : Tensor(n, c, h, w, data, elemsize) {}

inline RamTensor::~RamTensor() {
    tensorDataFree();
}

template <typename T>
inline void RamTensor::fill(T _v) {
    int size = total();
    T* ptr = reinterpret_cast<T*>(data_ptr);
    for (int i = 0; i < size; i++) {
        ptr[i] = _v;
    }
}

inline RamTensor::sptr RamTensor::clone() const {
    if (empty())
        return create();

    RamTensor::sptr ts = std::make_shared<RamTensor>(
                               n_batch, width, height, channel, element_size);

    if (total() > 0) {
        memcpy(ts->data_ptr, data_ptr, total() * element_size);
    }

    return ts;
}

void* RamTensor::tensorDataMalloc(size_t size) {
    unsigned char* udata = (unsigned char*)malloc(size +
                            sizeof(void*) + MALLOC_ALIGN);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr(
                (unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void RamTensor::tensorDataFree() {
    if (data_ptr) {
        unsigned char* udata = ((unsigned char**)data_ptr)[-1];
        free(udata);
    }
}

void RamTensor::writeData(void* data, size_t size) {
    if (size == (total() * element_size) && data_ptr != nullptr)
        memcpy(data_ptr, data, size);
    else
        throw std::runtime_error("RamTensor error in write data!");
}

template <typename T>
inline T* RamTensor::grepBatchData(int n) {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr) +
           n * channel * cstep * element_size;
}

template <typename T>
inline T* RamTensor::grepChannelData(int n, int c) {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr) +
           (n * channel * cstep + c * cstep) * element_size;
}

template <typename T>
inline T* RamTensor::grepRowData(int n, int c, int h) {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr) +
           (n * channel * cstep + c * cstep + h * width) * element_size;
}

template <typename T>
inline T RamTensor::grepElement(int n, int c, int h, int w) {
    assert(sizeof(T) == element_size);
    return *(reinterpret_cast<T*>(data_ptr) +
           (n * channel * cstep + c * cstep + h * width + w) * element_size);
}

template <typename T>
inline RamTensor::operator T*() {
    assert(sizeof(T) == element_size);
    return reinterpret_cast<T*>(data_ptr);
}

}  // namespace rvos
