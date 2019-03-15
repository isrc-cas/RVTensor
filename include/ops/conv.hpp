/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_CONV_HPP_
#define INCLUDE_OPS_CONV_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"

namespace rvos {

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

class CPUConvOp: public Operation {
 public:
    using sptr = std::shared_ptr<CPUConvOp>;
    static sptr create();
    static sptr create(ConvParam conv_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        RamTensor::sptr weight,
        RamTensor::sptr bias = nullptr);

    /**
     * Constructor & Deconstructor
     */
    CPUConvOp();
    CPUConvOp(ConvParam conv_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        RamTensor::sptr weight,
        RamTensor::sptr bias = nullptr);
    ~CPUConvOp();
    CPUConvOp& operator=(const CPUConvOp& conv_op);

    /**
     * check output dims
     */
    void checkOutputDims() override;

    /**
     * inference
     */
    void forward_compute() override;

 private:
    /// conv paramter
    ConvParam param_;
    /// model data: weight
    RamTensor::sptr weight_;
    /// model data: bias
    RamTensor::sptr bias_;
};

}  // namespace rvos

#endif  // INCLUDE_OPS_CONV_HPP_
