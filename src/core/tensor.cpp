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

Tensor::sptr Tensor::create(int w, int h, int c, size_t elemsize) {
    return std::make_shared<Tensor>(w, h, c, elemsize);
}

Tensor::sptr Tensor::create(int w, int h, int c, void* data, size_t elemsize) {
    return std::make_shared<Tensor>(w, h, c, data, elemsize);
}

inline Tensor::Tensor() : data_ptr(nullptr), reference_count(nullptr),
                          element_size(0), weight(0), height(0), channel(0),
                          cstep(0) {}

inline Tensor::Tensor(int w, int h, int c, size_t elemsize)
                        : data_ptr(nullptr), reference_count(nullptr) {}

inline Tensor::Tensor(int w, int h, int c, void* data, size_t elemsize)
    : data_ptr(data), reference_count(nullptr), element_size(elemsize),
      weight(w), height(h), channel(c) {
    cstep = alignSize(weight * height * element_size,
                                          MALLOC_ALIGN) / element_size;
}

inline Tensor::~Tensor() {
    release();
}

inline Tensor& Tensor::operator=(const Tensor& m) {
    if (this == &m)
        return *this;

    if (m.reference_count)
        (*(m.reference_count))++;

    release();

    data_ptr = m.data_ptr;
    reference_count = m.reference_count;
    element_size = m.element_size;

    weight = m.weight;
    height = m.height;
    channel = m.channel;

    cstep = m.cstep;

    return *this;
}

inline void Tensor::addReference() {
    if (reference_count)
        (*reference_count)++;
}

inline void Tensor::release() {
    data_ptr = nullptr;
    element_size = 0;
    weight = 0;
    height = 0;
    channel = 0;
    cstep = 0;
    reference_count = nullptr;
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

FlashTensor::sptr FlashTensor::create(int w, int h, int c, size_t elemsize) {
    return std::make_shared<FlashTensor>(w, h, c, elemsize);
}

FlashTensor::sptr FlashTensor::create(int w, int h, int c,
                                      void* data, size_t elemsize) {
    return std::make_shared<FlashTensor>(w, h, c, data, elemsize);
}

inline FlashTensor::FlashTensor() : Tensor() {}

inline FlashTensor::FlashTensor(int w, int h, int c, size_t elemsize)
                   : Tensor(w, h, c, elemsize) {}

inline FlashTensor::FlashTensor(int w, int h, int c,
                                void* data, size_t elemsize)
                   : Tensor(w, h, c, data, elemsize) {}

inline FlashTensor::~FlashTensor() {
    data_ptr = nullptr;
}

inline void FlashTensor::bindData(void* data, size_t size) {
    if (size == total() && data_ptr == nullptr)
        data_ptr = data;
    // else report wrong msg
}

inline const FlashTensor FlashTensor::grep_channel(int _c) const {
    int c = 1;
    return FlashTensor(weight, height, c, (unsigned char*)data_ptr +
                       cstep * _c * element_size, element_size);
}

template <typename T>
inline const T* FlashTensor::grep_row(int y) const {
    return (const T*)data_ptr + weight * y;
}

inline const FlashTensor FlashTensor::grep_channel_range(
                                      int c, int channels) const {
    return FlashTensor(weight, height, channels,
             (unsigned char*)data_ptr + cstep * c * element_size, element_size);
}

inline const FlashTensor FlashTensor::grep_row_range(int y, int rows) const {
    int c = 1;
    return FlashTensor(weight, rows, c,
            (unsigned char*)data_ptr + weight * y * element_size, element_size);
}

inline const FlashTensor FlashTensor::grep_range(int x, int n) const {
    int h = 1;
    int c = 1;
    return FlashTensor(n, h, c,
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

RamTensor::sptr RamTensor::create(int w, int h, int c, size_t elemsize) {
    return std::make_shared<RamTensor>(w, h, c, elemsize);
}

RamTensor::sptr RamTensor::create(int w, int h, int c,
                                  void* data, size_t elemsize) {
    return std::make_shared<RamTensor>(w, h, c, data, elemsize);
}


inline RamTensor::RamTensor() : Tensor() {}

inline RamTensor::RamTensor(int w, int h, int c, size_t elemsize)
                   : Tensor(w, h, c, elemsize) {
    create_resourse(w, h, c, elemsize);
}

inline RamTensor::RamTensor(int w, int h, int c, void* data, size_t elemsize)
                   : Tensor(w, h, c, data, elemsize) {}

inline RamTensor::~RamTensor() {
    release_resourse();
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
                        weight, height, channel, element_size);

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

inline RamTensor RamTensor::grep_channel(int _c) {
    int c = 1;
    return RamTensor(weight, height, c, (unsigned char*)data_ptr +
                     cstep * _c * element_size, element_size);
}

template <typename T>
inline T* RamTensor::grep_row(int y) {
    return reinterpret_cast<T*>(data_ptr) + weight * y;
}

inline RamTensor RamTensor::grep_channel_range(int _c, int channels) {
    return RamTensor(weight, height, channels,
            (unsigned char*)data_ptr + cstep * _c * element_size, element_size);
}

inline RamTensor RamTensor::grep_row_range(int y, int rows) {
    int c = 1;
    return RamTensor(weight, rows, c,
            (unsigned char*)data_ptr + weight * y * element_size, element_size);
}

inline RamTensor RamTensor::grep_range(int x, int n) {
    int h = 1;
    int c = 1;
    return RamTensor(n, h, c,
                     (unsigned char*)data_ptr + x * element_size, element_size);
}

template <typename T>
inline RamTensor::operator T*() {
    return reinterpret_cast<T*>(data_ptr);
}

inline void RamTensor::create_resourse(int w, int h, int c, size_t elemsize) {
    if (weight == w && height == h && channel == c && element_size == elemsize)
        return;

    release();

    element_size = elemsize;

    weight = w;
    height = h;
    channel = c;

    cstep = alignSize(weight * height * element_size,
                                              MALLOC_ALIGN) / element_size;

    if (total() > 0) {
        size_t totalsize = alignSize(total() * elemsize, 4);
        data_ptr = tensorDataMalloc(totalsize + sizeof(*reference_count));
        reference_count = reinterpret_cast<int*>(
                                  ((unsigned char*)data_ptr) + totalsize);
        *reference_count = 1;
    }
}

inline void RamTensor::release_resourse() {
    if (reference_count && (--(*reference_count)) == 1) {
        tensorDataFree(data_ptr);
    }
}

}  // namespace rvos
