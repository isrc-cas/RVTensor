/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_TYPES_HPP_
#define INCLUDE_CORE_TYPES_HPP_

namespace RVTensor {

struct ConvParam {
  /// stride
  int sw;
  int sh;
  /// dilated
  int dw;
  int dh;
  /// add pad
  int pw;
  int ph;
  /// quantization int8 or float
  bool quantized;
};

/// Quantizing deep convolutional networks for efficient inference: A whitepaper
/// https://arxiv.org/abs/1806.08342
enum QuantizeStrategy {
  AFFINE_QUANTIZE_FLOAT32TOUINT8      = 0,  // [0 - 255]
  AFFINE_QUANTIZE_FLOAT32TOUINT16     = 1,
  SYMMETRIC_QUANTIZE_FLOAT32TOINT8    = 2,  // [-127, 127]
  SYMMETRIC_QUANTIZE_FLOAT32TOINT16   = 3,
  AFFINE_DEQUANTIZE_UINT8TOFLOAT32    = 4,
  AFFINE_DEQUANTIZE_UINT16TOFLOAT32   = 5,
  SYMMETRIC_DEQUANTIZE_INT8TOFLOAT32  = 6,
  SYMMETRIC_DEQUANTIZE_INT16TOFLOAT32 = 7,
  NONE                                = 8
};

struct QuantizeParam {
  int input_elemsize;
  int output_elemsize;
  QuantizeStrategy quant_type;
};

}  // namespace RVTensor

#endif  // INCLUDE_CORE_TYPES_HPP_
