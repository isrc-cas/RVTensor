/*  The MIT License
 *  
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

#ifndef RVOS_MAT_H
#define RVOS_MAT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace rvos {

// the three dimension matrix
class Mat
{
 public:
    // empty
    Mat();
    // vec
    Mat(int w, size_t elemsize = 4u);
    // image
    Mat(int w, int h, size_t elemsize = 4u);
    // dim
    Mat(int w, int h, int c, size_t elemsize = 4u);
    // packed vec
    Mat(int w, size_t elemsize, int packing);
    // packed image
    Mat(int w, int h, size_t elemsize, int packing);
    // packed dim
    Mat(int w, int h, int c, size_t elemsize, int packing);
    // copy
    Mat(const Mat& m);
    // external vec
    Mat(int w, void* data, size_t elemsize = 4u);
    // external image
    Mat(int w, int h, void* data, size_t elemsize = 4u);
    // external dim
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u);
    // external packed vec
    Mat(int w, void* data, size_t elemsize, int packing);
    // external packed image
    Mat(int w, int h, void* data, size_t elemsize, int packing);
    // external packed dim
    Mat(int w, int h, int c, void* data, size_t elemsize, int packing);
    // release
    ~Mat();
    // assign
    Mat& operator=(const Mat& m);
    // set all
    void fill(float v);
    void fill(int v);
    template <typename T> void fill(T v);
    // deep copy
    Mat clone() const;
    // allocate vec
    void create(int w, size_t elemsize = 4u);
    // allocate image
    void create(int w, int h, size_t elemsize = 4u);
    // allocate dim
    void create(int w, int h, int c, size_t elemsize = 4u);
    // allocate packed vec
    void create(int w, size_t elemsize, int packing);
    // allocate packed image
    void create(int w, int h, size_t elemsize, int packing);
    // allocate packed dim
    void create(int w, int h, int c, size_t elemsize, int packing);
    // allocate like
    void create_like(const Mat& m);
    // refcount++
    void addref();
    // refcount--
    void release();

    bool empty() const;
    size_t total() const;

    // data reference
    Mat channel(int c);
    const Mat channel(int c) const;
    float* row(int y);
    const float* row(int y) const;
    template<typename T> T* row(int y);
    template<typename T> const T* row(int y) const;

    // range reference
    Mat channel_range(int c, int channels);
    const Mat channel_range(int c, int channels) const;
    Mat row_range(int y, int rows);
    const Mat row_range(int y, int rows) const;
    Mat range(int x, int n);
    const Mat range(int x, int n) const;

    // access raw data
    template<typename T> operator T*();
    template<typename T> operator const T*() const;

    // convenient access float vec element
    float& operator[](int i);
    const float& operator[](int i) const;

#if RVOS_PIXEL
    enum
    {
        PIXEL_CONVERT_SHIFT = 16,
        PIXEL_FORMAT_MASK = 0x0000ffff,
        PIXEL_CONVERT_MASK = 0xffff0000,

        PIXEL_RGB       = 1,
        PIXEL_BGR       = (1 << 1),
        PIXEL_GRAY      = (1 << 2),
        PIXEL_RGBA      = (1 << 3),

        PIXEL_RGB2BGR   = PIXEL_RGB | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGB2GRAY  = PIXEL_RGB | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_BGR2RGB   = PIXEL_BGR | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_BGR2GRAY  = PIXEL_BGR | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),

        PIXEL_GRAY2RGB  = PIXEL_GRAY | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_GRAY2BGR  = PIXEL_GRAY | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),

        PIXEL_RGBA2RGB  = PIXEL_RGBA | (PIXEL_RGB << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2BGR  = PIXEL_RGBA | (PIXEL_BGR << PIXEL_CONVERT_SHIFT),
        PIXEL_RGBA2GRAY = PIXEL_RGBA | (PIXEL_GRAY << PIXEL_CONVERT_SHIFT),
    };
    // convenient construct from pixel data
    static Mat from_pixels(const unsigned char* pixels, int type, int w, int h);
    // convenient construct from pixel data and resize to specific size
    static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height);

    // convenient export to pixel data
    void to_pixels(unsigned char* pixels, int type) const;
    // convenient export to pixel data and resize to specific size
    void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
#endif // RVOS_PIXEL

    // substract channel-wise mean values, then multiply by normalize values, pass 0 to skip
    void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

    // convenient construct from half precisoin floating point data
    static Mat from_float16(const unsigned short* data, int size);

    // pointer to the data
    void* data;

    // pointer to the reference counter
    // when points to user-allocated data, the pointer is NULL
    int* refcount;

    // element size in bytes
    // 4 = float32/int32
    // 2 = float16
    // 1 = int8/uint8
    // 0 = empty
    size_t elemsize;

    // packed count inside element
    // c/1-h-w-1  h/1-w-1  w/1-1  scalar
    // c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
    // c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
    int packing;

    // the dimensionality
    int dims;

    int w;
    int h;
    int c;

    size_t cstep;
};

// misc function
#if RVOS_PIXEL
// convert yuv420sp(nv21) to rgb, the fast approximate version
void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
// image pixel bilinear resize
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
// image pixel bilinear resize, convenient wrapper for yuv420sp(nv21)
void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
#endif // RVOS_PIXEL

// mat process
enum
{
    BORDER_CONSTANT = 0,
    BORDER_REPLICATE = 1,
};
void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, int num_threads = 1);
void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int num_threads = 1);
void resize_bilinear(const Mat& src, Mat& dst, int w, int h, int num_threads = 1);
void convert_packing(const Mat& src, Mat& dst, int packing, int num_threads = 1);

inline Mat::Mat()
    : data(0), refcount(0), elemsize(0), packing(0), dims(0), w(0), h(0), c(0), cstep(0)
{
}

inline Mat::Mat(int _w, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _elemsize);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _elemsize);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _c, _elemsize);
}

inline Mat::Mat(int _w, size_t _elemsize, int _packing)
    : data(0), refcount(0), dims(0)
{
    create(_w, _elemsize, _packing);
}

inline Mat::Mat(int _w, int _h, size_t _elemsize, int _packing)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _elemsize, _packing);
}

inline Mat::Mat(int _w, int _h, int _c, size_t _elemsize, int _packing)
    : data(0), refcount(0), dims(0)
{
    create(_w, _h, _c, _elemsize, _packing);
}

inline Mat::Mat(const Mat& m)
    : data(m.data), refcount(m.refcount), elemsize(m.elemsize), packing(m.packing), dims(m.dims), w(m.w), h(m.h), c(m.c), cstep(m.cstep)
{
    // if (refcount)
    //     RVOS_XADD(refcount, 1);
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), packing(1), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), packing(1), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize)
    : data(_data), refcount(0), elemsize(_elemsize), packing(1), dims(3), w(_w), h(_h), c(_c)
{
    // cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Mat::Mat(int _w, void* _data, size_t _elemsize, int _packing)
    : data(_data), refcount(0), elemsize(_elemsize), packing(_packing), dims(1), w(_w), h(1), c(1)
{
    cstep = w;
}

inline Mat::Mat(int _w, int _h, void* _data, size_t _elemsize, int _packing)
    : data(_data), refcount(0), elemsize(_elemsize), packing(_packing), dims(2), w(_w), h(_h), c(1)
{
    cstep = w * h;
}

inline Mat::Mat(int _w, int _h, int _c, void* _data, size_t _elemsize, int _packing)
    : data(_data), refcount(0), elemsize(_elemsize), packing(_packing), dims(3), w(_w), h(_h), c(_c)
{
    // cstep = alignSize(w * h * elemsize, 16) / elemsize;
}

inline Mat::~Mat()
{
    release();
}

inline Mat& Mat::operator=(const Mat& m)
{
    if (this == &m)
        return *this;

    // if (m.refcount)
    //     RVOS_XADD(m.refcount, 1);

    release();

    data = m.data;
    refcount = m.refcount;
    elemsize = m.elemsize;
    packing = m.packing;

    dims = m.dims;
    w = m.w;
    h = m.h;
    c = m.c;

    cstep = m.cstep;

    return *this;
}

inline void Mat::fill(float _v)
{
    int size = total();
    float* ptr = (float*)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
    float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
    if (nn > 0)
    {
    asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.f32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

inline void Mat::fill(int _v)
{
    int size = total();
    int* ptr = (int*)data;

#if __ARM_NEON
    int nn = size >> 2;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
    int32x4_t _c = vdupq_n_s32(_v);
#if __aarch64__
    if (nn > 0)
    {
    asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "subs       %0, #1              \n"
        "vst1.s32   {%e4-%f4}, [%1 :128]!\n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr++ = _v;
    }
}

template <typename T>
inline void Mat::fill(T _v)
{
    int size = total();
    T* ptr = (T*)data;
    for (int i=0; i<size; i++)
    {
        ptr[i] = _v;
    }
}

inline Mat Mat::clone() const
{
    if (empty())
        return Mat();

    Mat m;
    if (dims == 1)
        m.create(w, elemsize, packing);
    else if (dims == 2)
        m.create(w, h, elemsize, packing);
    else if (dims == 3)
        m.create(w, h, c, elemsize, packing);

    if (total() > 0)
    {
        memcpy(m.data, data, total() * elemsize);
    }

    return m;
}

inline void Mat::create(int _w, size_t _elemsize)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && packing == 1)
        return;

    release();

    elemsize = _elemsize;
    packing = 1;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        // size_t totalsize = alignSize(total() * elemsize, 4);
        // data = fastMalloc(totalsize + (int)sizeof(*refcount));
        // refcount = (int*)(((unsigned char*)data) + totalsize);
        // *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && packing == 1)
        return;

    release();

    elemsize = _elemsize;
    packing = 1;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        // size_t totalsize = alignSize(total() * elemsize, 4);
        // data = fastMalloc(totalsize + (int)sizeof(*refcount));
        // refcount = (int*)(((unsigned char*)data) + totalsize);
        // *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && packing == 1)
        return;

    release();

    elemsize = _elemsize;
    packing = 1;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    // cstep = alignSize(w * h * elemsize, 16) / elemsize;

    // if (total() > 0)
    // {
    //     size_t totalsize = alignSize(total() * elemsize, 4);
    //     data = fastMalloc(totalsize + (int)sizeof(*refcount));
    //     refcount = (int*)(((unsigned char*)data) + totalsize);
    //     *refcount = 1;
    // }
}

inline void Mat::create(int _w, size_t _elemsize, int _packing)
{
    if (dims == 1 && w == _w && elemsize == _elemsize && packing == _packing)
        return;

    release();

    elemsize = _elemsize;
    packing = _packing;

    dims = 1;
    w = _w;
    h = 1;
    c = 1;

    cstep = w;

    if (total() > 0)
    {
        // size_t totalsize = alignSize(total() * elemsize, 4);
        // data = fastMalloc(totalsize + (int)sizeof(*refcount));
        // refcount = (int*)(((unsigned char*)data) + totalsize);
        // *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, size_t _elemsize, int _packing)
{
    if (dims == 2 && w == _w && h == _h && elemsize == _elemsize && packing == _packing)
        return;

    release();

    elemsize = _elemsize;
    packing = _packing;

    dims = 2;
    w = _w;
    h = _h;
    c = 1;

    cstep = w * h;

    if (total() > 0)
    {
        // size_t totalsize = alignSize(total() * elemsize, 4);
        // data = fastMalloc(totalsize + (int)sizeof(*refcount));
        // refcount = (int*)(((unsigned char*)data) + totalsize);
        // *refcount = 1;
    }
}

inline void Mat::create(int _w, int _h, int _c, size_t _elemsize, int _packing)
{
    if (dims == 3 && w == _w && h == _h && c == _c && elemsize == _elemsize && packing == _packing)
        return;

    release();

    elemsize = _elemsize;
    packing = _packing;

    dims = 3;
    w = _w;
    h = _h;
    c = _c;

    // cstep = alignSize(w * h * elemsize, 16) / elemsize;

    // if (total() > 0)
    // {
    //     size_t totalsize = alignSize(total() * elemsize, 4);
    //     data = fastMalloc(totalsize + (int)sizeof(*refcount));
    //     refcount = (int*)(((unsigned char*)data) + totalsize);
    //     *refcount = 1;
    // }
}

inline void Mat::create_like(const Mat& m)
{
    if (m.dims == 1)
        create(m.w, m.elemsize, m.packing);
    else if (m.dims == 2)
        create(m.w, m.h, m.elemsize, m.packing);
    else if (m.dims == 3)
        create(m.w, m.h, m.c, m.elemsize, m.packing);
}

inline void Mat::addref()
{
    // if (refcount)
    //     RVOS_XADD(refcount, 1);
}

inline void Mat::release()
{
    // if (refcount && RVOS_XADD(refcount, -1) == 1)
    // {
    //     fastFree(data);
    // }

    data = 0;

    elemsize = 0;
    packing = 0;

    dims = 0;
    w = 0;
    h = 0;
    c = 0;

    cstep = 0;

    refcount = 0;
}

inline bool Mat::empty() const
{
    return data == 0 || total() == 0;
}

inline size_t Mat::total() const
{
    return cstep * c;
}

inline Mat Mat::channel(int _c)
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, packing);
}

inline const Mat Mat::channel(int _c) const
{
    return Mat(w, h, (unsigned char*)data + cstep * _c * elemsize, elemsize, packing);
}

inline float* Mat::row(int y)
{
    return (float*)data + w * y;
}

inline const float* Mat::row(int y) const
{
    return (const float*)data + w * y;
}

template <typename T>
inline T* Mat::row(int y)
{
    return (T*)data + w * y;
}

template <typename T>
inline const T* Mat::row(int y) const
{
    return (const T*)data + w * y;
}

inline Mat Mat::channel_range(int _c, int channels)
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, packing);
}

inline const Mat Mat::channel_range(int _c, int channels) const
{
    return Mat(w, h, channels, (unsigned char*)data + cstep * _c * elemsize, elemsize, packing);
}

inline Mat Mat::row_range(int y, int rows)
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, packing);
}

inline const Mat Mat::row_range(int y, int rows) const
{
    return Mat(w, rows, (unsigned char*)data + w * y * elemsize, elemsize, packing);
}

inline Mat Mat::range(int x, int n)
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, packing);
}

inline const Mat Mat::range(int x, int n) const
{
    return Mat(n, (unsigned char*)data + x * elemsize, elemsize, packing);
}

template <typename T>
inline Mat::operator T*()
{
    return (T*)data;
}

template <typename T>
inline Mat::operator const T*() const
{
    return (const T*)data;
}

inline float& Mat::operator[](int i)
{
    return ((float*)data)[i];
}

inline const float& Mat::operator[](int i) const
{
    return ((const float*)data)[i];
}

} // namespace ncnn

#endif // RVOS_MAT_H
