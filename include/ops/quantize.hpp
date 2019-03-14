/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_QUANTIZE_HPP_
#define INCLUDE_OPS_QUANTIZE_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"

namespace rvos {

/// Quantizing deep convolutional networks for efficient inference: A whitepaper
/// https://arxiv.org/abs/1806.08342
enum QuantizeStrategy {
    AFFINE_QUANTIZE_FLOAT32TOUINT8      = 0, /// [0 - 255]
    AFFINE_QUANTIZE_FLOAT32TOUINT16     = 1,
    SYMMETRIC_QUANTIZE_FLOAT32TOINT8    = 2, /// [-127, 127]
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

class QuantizeOp: public Operation {
 public:
     using sptr = std::shared_ptr<QuantizeOp>;
     static sptr create();
     static sptr create(QuantizeParam qp,
                        RamTensor::sptr input,
                        RamTensor::sptr output);

     /**
      * Constructor & Deconstructor
      */
     QuantizeOp();
     QuantizeOp(QuantizeParam qp, RamTensor::sptr input, RamTensor::sptr output);
     ~QuantizeOp();
     QuantizeOp& operator=(const QuantizeOp& quant_op);

     /**
      * check output dims
      */
     void checkOutputDims() override;

     /**
      * inference
      */
     void forward_compute() override;

 private:
     /// quantize paramter
     QuantizeParam param_;
};

}  // namespace rvos

#endif  // INCLUDE_OPS_QUANTIZE_HPP_
