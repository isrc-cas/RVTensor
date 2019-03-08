/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/tensor.hpp"

namespace rvos {

////////////////////// Tensor ////////////////////////////////
Tensor::sptr Tensor::create() {
    return std::make_shared<Tensor>();
}

Tensor::sptr Tensor::create(int w, int h,
                            int c, size_t elemsize, int refcount) {
    return std::make_shared<Tensor>(w, h, c, elemsize, refcount);
}

Tensor::sptr Tensor::create(int w, int h,
                            int c, void* data, size_t elemsize, int refcount) {
    return std::make_shared<Tensor>(w, h, c, data, elemsize, refcount);
}

inline Tensor::Tensor() : data_ptr(nullptr), reference_count(0),
                          element_size(0), weight(0), height(0), channel(0),
                          cstep(0), is_static(false) {}

inline Tensor::Tensor(int w, int h, int c, size_t elemsize, int refcount)
                        : data_ptr(nullptr),
                          reference_count(refcount), is_static(false) {}

inline Tensor::Tensor(int w, int h, int c,
                      void* data, size_t elemsize, int refcount)
    : data_ptr(data), reference_count(refcount), element_size(elemsize),
      weight(w), height(h), channel(c), is_static(false) {
    cstep = (channel <= 1) ? weight * height :
                             alignSize(weight * height * element_size,
                                       MALLOC_ALIGN) / element_size;
}

inline Tensor::~Tensor() {
    release();
}

inline int Tensor::increaseReference() {
    if (is_static == false)
        return ++reference_count;
    else
        return 0;
}

inline int Tensor::decreaseReference() {
    if (is_static == false)
        return --reference_count;
    else
        return 0;
}

inline void Tensor::release() {
    data_ptr = nullptr;
    element_size = 0;
    weight = 0;
    height = 0;
    channel = 0;
    cstep = 0;
    reference_count = 0;
    is_static = false;
}

inline bool Tensor::empty() const {
    return data_ptr == nullptr || total() == 0;
}

inline size_t Tensor::total() const {
    return cstep * channel;
}

/////////////////// FlashTensor /////////////////////////////
FlashTensor::sptr FlashTensor::create() {
    return std::make_shared<FlashTensor>();
}

FlashTensor::sptr FlashTensor::create(int w, int h, int c,
                                      size_t elemsize, int refcount) {
    FlashTensor::sptr shared_ptr = std::make_shared<FlashTensor>(w, h, c,
                                                      elemsize, refcount);
    shared_ptr->is_static = true;
    return shared_ptr;
}

FlashTensor::sptr FlashTensor::create(int w, int h, int c, void* data,
                                      size_t elemsize, int refcount) {
    FlashTensor::sptr shared_ptr = std::make_shared<FlashTensor>(
                                       w, h, c, data, elemsize, refcount);
    shared_ptr->is_static = true;
    return shared_ptr;
}

inline FlashTensor::FlashTensor() : Tensor() {}

inline FlashTensor::FlashTensor(int w, int h, int c,
                                size_t elemsize, int refcount)
                   : Tensor(w, h, c, elemsize, refcount) {}

inline FlashTensor::FlashTensor(int w, int h, int c,
                                void* data, size_t elemsize, int refcount)
                   : Tensor(w, h, c, data, elemsize, refcount) {}

inline FlashTensor::~FlashTensor() {
    data_ptr = nullptr;
}

inline void FlashTensor::bindData(void* data, size_t size) {
    if (size == total() && data_ptr == nullptr)
        data_ptr = data;
    // else report wrong msg
}

inline const FlashTensor::sptr FlashTensor::grepChannel(int _c) const {
    int c = 1;
    return create(weight, height, c, (unsigned char*)data_ptr +
                       cstep * _c * element_size, element_size);
}

template <typename T>
inline const T* FlashTensor::grepRow(int y) const {
    return (const T*)data_ptr + weight * y;
}

inline const FlashTensor::sptr FlashTensor::grepChannelRange(
                                      int c, int channels) const {
    return create(weight, height, channels,
             (unsigned char*)data_ptr + cstep * c * element_size, element_size);
}

inline const FlashTensor::sptr FlashTensor::grepRowRange(
                                          int y, int rows) const {
    int c = 1;
    return create(weight, rows, c,
            (unsigned char*)data_ptr + weight * y * element_size, element_size);
}

inline const FlashTensor::sptr FlashTensor::grepRange(int x, int n) const {
    int h = 1;
    int c = 1;
    return create(n, h, c,
                  (unsigned char*)data_ptr + x * element_size, element_size);
}

template <typename T>
inline FlashTensor::operator const T*() const {
    return (const T*)data_ptr;
}

////////////////// RamTensor ////////////////////////////////
RamTensor::sptr RamTensor::create() {
    return std::make_shared<RamTensor>();
}

RamTensor::sptr RamTensor::create(int w, int h, int c, size_t elemsize,
                                  int refcount) {
    return std::make_shared<RamTensor>(w, h, c, elemsize, refcount);
}

RamTensor::sptr RamTensor::create(int w, int h, int c,
                                  void* data, size_t elemsize, int refcount) {
    return std::make_shared<RamTensor>(w, h, c, data, elemsize, refcount);
}


inline RamTensor::RamTensor() : Tensor() {}

inline RamTensor::RamTensor(int w, int h, int c, size_t elemsize, int refcount)
                   : Tensor(w, h, c, elemsize, refcount) {
    createResource(w, h, c, elemsize);
}

inline RamTensor::RamTensor(int w, int h, int c, void* data, size_t elemsize,
                            int refcount)
                   : Tensor(w, h, c, data, elemsize, refcount) {}

inline RamTensor::~RamTensor() {
    releaseResource();
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
                        weight, height, channel, element_size, reference_count);

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

void RamTensor::tensorDataFree(void* ptr) {
    if (ptr) {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

void RamTensor::writeData(void* data, size_t size) {
    if (size == (total() * element_size) && data_ptr != nullptr)
        memcpy(data_ptr, data, size);
    // else report wrong msg
}

inline RamTensor::sptr RamTensor::grepChannel(int _c) {
    int c = 1;
    return create(weight, height, c, (unsigned char*)data_ptr +
                  cstep * _c * element_size, element_size, increaseReference());
}

template <typename T>
inline T* RamTensor::grepRow(int y) {
    return reinterpret_cast<T*>(data_ptr) + weight * y;
}

inline RamTensor::sptr RamTensor::grepChannelRange(int _c, int channels) {
    return create(weight, height, channels, (unsigned char*)data_ptr +
                  cstep * _c * element_size, element_size, increaseReference());
}

inline RamTensor::sptr RamTensor::grepRowRange(int y, int rows) {
    int c = 1;
    return create(weight, rows, c, (unsigned char*)data_ptr +
                  weight * y * element_size, element_size, increaseReference());
}

inline RamTensor::sptr RamTensor::grepRange(int x, int n) {
    int h = 1;
    int c = 1;
    return create(n, h, c, (unsigned char*)data_ptr + x * element_size,
                  element_size, increaseReference());
}

template <typename T>
inline RamTensor::operator T*() {
    return reinterpret_cast<T*>(data_ptr);
}

inline void RamTensor::createResource(int w, int h, int c, size_t elemsize) {
    cstep = (c <= 1) ? weight * height :
                     alignSize(weight * height * element_size,
                               MALLOC_ALIGN) / element_size;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data_ptr = tensorDataMalloc(totalsize);
    }
}

inline void RamTensor::releaseResource() {
    if (decreaseReference() == 0) {
        tensorDataFree(data_ptr);
    }
}

}  // namespace rvos
