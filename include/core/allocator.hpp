/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_ALLOCATOR_HPP_
#define INCLUDE_CORE_ALLOCATOR_HPP_

#include <memory>
#include <cassert>
#include <stdexcept>

namespace eigen {

// #if defined(RISCV)
  const std::size_t default_alignment = 64;
// #else
//  const std::size_t default_alignment = 8;
// #endif

template<typename T>
inline void check_size_for_overflow(std::size_t size) {
  if (size > std::size_t(-1) / sizeof(T))
    throw std::runtime_error("bad alloc");
}

/* ----- Hand made implementations of aligned malloc/free and realloc ----- */

/** Like malloc, but the returned pointer is guaranteed to be 16-byte aligned.
  * Fast, but wastes 16 additional bytes of memory.
  * Does not throw any exception.
  */
inline void* handmade_aligned_malloc(std::size_t size) {
  void *original = std::malloc(size + default_alignment);
  if (original == 0)
    return 0;
  void *aligned = reinterpret_cast<void*>(
                  (reinterpret_cast<std::size_t>(original) &
                  ~(std::size_t(default_alignment - 1))) + default_alignment);
  *(reinterpret_cast<void**>(aligned) - 1) = original;
  return aligned;
}

/** \internal Frees memory allocated with handmade_aligned_malloc */
inline void handmade_aligned_free(void *ptr) {
  if (ptr)
    std::free(*(reinterpret_cast<void**>(ptr) - 1));
}


/** Allocates size bytes.
 *  The returned pointer is guaranteed to have 16 or 32 bytes alignment
 *  depending on the requirements.
 * On allocation error, the returned pointer is null,
 * and std::bad_alloc is thrown.
 */
inline void* aligned_malloc(std::size_t size) {
  void *result;

  if (default_alignment == 0) {
    result = std::malloc(size);
  } else if (default_alignment == 16) {
    if (!(size < 16 || (std::size_t(result) % 16) == 0))
      throw std::runtime_error("System's malloc returned an unaligned pointer");
  } else {
    result = handmade_aligned_malloc(size);
  }

  if (!result && size)
    throw std::runtime_error("bad alloc");

  return result;
}

/** Frees memory allocated with aligned_malloc. */
inline void aligned_free(void *ptr) {
  if (default_alignment == 0)
    std::free(ptr);
  else
    handmade_aligned_free(ptr);
}

/****************************************************************************/

/** aligned_allocator
*
* STL compatible allocator to use with types requiring a non standrad alignment.
*
* The memory is aligned as for dynamically aligned matrix/array types such as
* MatrixXd. By default, it will thus provide at least 16 bytes alignment and
* more in following cases:
*  - 32 bytes alignment if AVX is enabled.
*  - 64 bytes alignment if AVX512 is enabled.
*
*/
template<class T>
class aligned_allocator : public std::allocator<T> {
 public:
  typedef std::size_t     size_type;
  typedef std::ptrdiff_t  difference_type;
  typedef T*              pointer;
  typedef const T*        const_pointer;
  typedef T&              reference;
  typedef const T&        const_reference;
  typedef T               value_type;

  template<class U>
  struct rebind {
    typedef aligned_allocator<U> other;
  };

  aligned_allocator() : std::allocator<T>() {}

  aligned_allocator(const aligned_allocator& other)
                      : std::allocator<T>(other) {}

  template<class U>
  aligned_allocator(const aligned_allocator<U>& other)
                      : std::allocator<T>(other) {}

  ~aligned_allocator() {}

  pointer allocate(size_type num, const void* /*hint*/ = 0) {
    check_size_for_overflow<T>(num);
    return static_cast<pointer>(aligned_malloc(num * sizeof(T)));
  }

  void deallocate(pointer p, size_type /*num*/) {
    aligned_free(p);
  }
};

template<typename T, std::size_t align = std::alignment_of<T>::value, typename... Args>  // NOLINT
std::shared_ptr<T> make_shared(Args&&... args) {
  if (align > default_alignment) {
    typedef aligned_allocator<T> alloc_type;
    return std::allocate_shared<T, alloc_type>(
                 alloc_type(), std::forward<Args>(args)...);
  } else {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }
}

}  // namespace eigen

#endif  // INCLUDE_CORE_ALLOCATOR_HPP_
