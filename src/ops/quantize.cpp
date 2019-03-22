/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <cmath>
#include <algorithm>
#include "include/ops/quantize.hpp"
#include "include/core/tensor.hpp"

namespace RVTensor {

QuantizeOp::sptr QuantizeOp::create() {
  return std::make_shared<QuantizeOp>();
}

QuantizeOp::sptr QuantizeOp::create(QuantizeParam qp, RamTensor::sptr input,
                                    RamTensor::sptr output) {
  QuantizeOp::sptr ptr = std::make_shared<QuantizeOp>(qp, input, output);
  ptr->checkOutputDims();
  return ptr;
}

inline QuantizeOp::QuantizeOp() : Operation({}, {}),
       param_({0, 0, QuantizeStrategy::NONE}) {}

inline QuantizeOp::QuantizeOp(QuantizeParam qp, RamTensor::sptr input,
                              RamTensor::sptr output)
  : Operation({input}, {output}), param_(qp) {}

inline QuantizeOp::~QuantizeOp() {}

inline void QuantizeOp::checkOutputDims() {
  auto input = getInputs()[0];
  auto output = getOutputs()[0];
  if (input->n_batch != output->n_batch ||
      input->channel != output->channel ||
      input->height != output->height ||
      input->width != output->width) {
    throw std::runtime_error(
        "QuantizeOp shape of input or output is wrong!");
  }

  if (input->element_size != param_.input_elemsize ||
      output->element_size != param_.output_elemsize) {
    throw std::runtime_error("QuantizeOp Param is wrong!");
  }
}

inline void QuantizeOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  if (param_.quant_type == AFFINE_QUANTIZE_FLOAT32TOUINT8) {
    float* input = reinterpret_cast<float *>(input_tensor->data_ptr);
    uint8_t* output = reinterpret_cast<uint8_t *>(output_tensor->data_ptr);
    int element_num = input_tensor->count();
    float min = *std::min_element(input, input + element_num);
    float max = *std::max_element(input, input + element_num);
    input_tensor->setQuantizeRange(min, max);

    const float quant_min = std::min(static_cast<float>(0), 0.0f);
    const float quant_max = std::max(static_cast<float>(255), 0.0f);
    const float scale = (max - min) / (quant_max - quant_min);
    const float zero_point_from_min = min - min / scale;
    int64_t zero_point;
    if (zero_point_from_min < quant_min) {
      zero_point = static_cast<int64_t>(quant_min);
    } else if (zero_point_from_min > quant_max) {
      zero_point = static_cast<int64_t>(quant_max);
    } else {
      zero_point = static_cast<int64_t>(round(zero_point_from_min));
    }
    output_tensor->setQuantizer(scale, zero_point);

    const double inverse_scale = 1. / output_tensor->scale;
    for (int i = 0; i < element_num; i++) {
      const float src_val = input[i];
      double scaled_val;
      if (output_tensor->scale == 0) {
        scaled_val = output_tensor->zero_point;
      } else {
        scaled_val = output_tensor->zero_point + inverse_scale * src_val;
      }
      output[i] = static_cast<uint8_t>(round(scaled_val));
    }
    min = *std::min_element(output, output + element_num);
    max = *std::max_element(output, output + element_num);
    output_tensor->setQuantizeRange(min, max);
  } else if (param_.quant_type == AFFINE_DEQUANTIZE_UINT8TOFLOAT32) {
    uint8_t* input = reinterpret_cast<uint8_t *>(input_tensor->data_ptr);
    float* output = reinterpret_cast<float *>(output_tensor->data_ptr);
    int element_num = input_tensor->count();
    float min = *std::min_element(input, input + element_num);
    float max = *std::max_element(input, input + element_num);

    const int64_t number_of_steps = static_cast<int64_t>(1)
                                    << (sizeof(uint8_t) * 8);
    const float scale = (max - min) / (number_of_steps - 1.0);
    const float range_min_rounded = max == min ? min
                                    : round(min / scale) * scale;
    const float lowest = min;
    for (int i = 0; i < element_num; i++) {
      float val = static_cast<float>(input[i]);
      output[i] = (range_min_rounded - lowest * scale) + val * scale;
    }
  } else {
    throw std::runtime_error("QuantizeOP unsupport quantize strategy!");
  }
}

}  // namespace RVTensor
