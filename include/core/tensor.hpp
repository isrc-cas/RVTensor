/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_TENSOR_HPP_
#define INCLUDE_CORE_TENSOR_HPP_

#include <memory>
#include <string.h> //NOLINT

namespace rvos {

/**
 * Aligns a buffer size to the specified number of bytes
 */
static inline size_t alignSize(size_t sz, int n) {
    return (sz + n-1) & -n;
}

/**
 * the alignment of all the allocated buffers
 */
#define MALLOC_ALIGN    16

/**Aligns a pointer to the specified number of bytes
 * @param ptr: Aligned pointer
 * @param n: Alignment size that must be a power of two
 */
template<typename _Tp> static inline _Tp* alignPtr(
               _Tp* ptr, int n = sizeof(_Tp)) {
    return reinterpret_cast<_Tp*>(((size_t)ptr + n - 1) & -n);
}

/**
 * RVOS data descriptor
 *
 *                 Tensor
 *                   +
 *                   |
 *       +-----------+-----------+
 *       |                       |
 *       v                       v
 *  FlashTensor               RamTensor
 *
 * FlashTensor describes model data which is already loaded in flash.
 * RamTensor describes dynamic memory data.
 *
 */
class Tensor {
 public:
     using sptr = std::shared_ptr<Tensor>;
     static sptr create();
     static sptr create(int w, int h, int c, size_t elemsize = 4u,
                        int refcount = 0);
     static sptr create(int w, int h, int c, void* data, size_t elemsize = 4u,
                        int refcount = 0);

     /**
      * Constructor & Deconstructor
      */
     Tensor();
     Tensor(int w, int h, int c, size_t elemsize = 4u, int refcount = 0);
     Tensor(int w, int h, int c, void* data,
            size_t elemsize = 4u, int refcount = 0);
     ~Tensor();
     Tensor& operator=(const Tensor& m);

     /**
      * refcount operation
      */
     int increaseReference();
     int decreaseReference();
     /**
      * releae
      */
     void release();

     /**
      * judage if Tensor is empty
      */
     bool empty() const;
     /**
      * Number of the elements in Tensor
      */
     size_t total() const;

     /**
      * pointer to the data
      */
     void* data_ptr;

     /**
      * reference counter
      */
     int reference_count;
     bool is_static;

     /**
      * element size in bytes
      * 4 = float32/int32
      * 2 = float16
      * 1 = int8/uint8
      * 0 = empty
      */
     size_t element_size;

     /**
      * dimension of the Tensor
      */
     int weight;
     int height;
     int channel;

     /**
      * layout of the martrix(w, h)
      */
     size_t cstep;
};

class FlashTensor : public Tensor {
 public:
     using sptr = std::shared_ptr<FlashTensor>;
     static sptr create();
     static sptr create(int w, int h, int c, size_t elemsize = 4u,
                        int refcount = 0);
     static sptr create(int w, int h, int c, void* data, size_t elemsize = 4u,
                        int refcount = 0);

     /**
      * Constructor & Deconstructor
      */
     FlashTensor();
     FlashTensor(int w, int h, int c, size_t elemsize = 4u, int refcount = 0);
     FlashTensor(int w, int h, int c, void* data,
                 size_t elemsize = 4u, int refcount = 0);
     ~FlashTensor();
     FlashTensor& operator=(const FlashTensor& m);

     /**
      * bind model data to the Tensor
      */
     void bindData(void* data, size_t size);

     /**
      * get const data reference
      */
     const FlashTensor::sptr grepChannel(int c) const;
     template<typename T> const T* grepRow(int y) const;

     /**
      * get const range reference
      */
     const FlashTensor::sptr grepChannelRange(int c, int channels) const;
     const FlashTensor::sptr grepRowRange(int y, int rows) const;
     const FlashTensor::sptr grepRange(int x, int n) const;

     /**
      * access const raw data
      */
     template<typename T> operator const T*() const;
};

class RamTensor : public Tensor {
 public:
     using sptr = std::shared_ptr<RamTensor>;
     static sptr create();
     static sptr create(int w, int h, int c, size_t elemsize = 4u,
                        int refcount = 0);
     static sptr create(int w, int h, int c, void* data, size_t elemsize = 4u,
                        int refcount = 0);

     /**
      * Constructor & Deconstructor
      */
     RamTensor();
     RamTensor(int w, int h, int c, size_t elemsize, int refcount);
     RamTensor(int w, int h, int c, void* data, size_t elemsize, int refcount);
     ~RamTensor();
     RamTensor& operator=(const RamTensor& m);

     /**
      * set tensor data with v
      */
     template <typename T> void fill(T v);
     /**
      * deep copy of this Tensor
      */
     RamTensor::sptr clone() const;

     /**
      *  write data to data_ptr
      */
     void writeData(void* data, size_t size);

     /**
      * get data reference
      */
     RamTensor::sptr grepChannel(int c);
     template<typename T> T* grepRow(int y);

     /**
      * get range reference
      */
     RamTensor::sptr grepChannelRange(int c, int channels);
     RamTensor::sptr grepRowRange(int y, int rows);
     RamTensor::sptr grepRange(int x, int n);

     /**
      * access raw data
      */
     template<typename T> operator T*();

 private:
     /**
      *  interface of create&release malloced data
      */
     void createResource(int w, int h, int c, size_t elemsize);
     void releaseResource();
     /**
      *  memory malloc&free for data_ptr
      */
     void* tensorDataMalloc(size_t size);
     void tensorDataFree(void* ptr);
};

}  // namespace rvos

#endif  // INCLUDE_CORE_TENSOR_HPP_
