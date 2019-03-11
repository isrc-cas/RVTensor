/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/ops/conv.hpp"

namespace rvos {

CPUConvOp::sptr CPUConvOp::create() {
    return std::make_shared<CPUConvOp>();
}

CPUConvOp::sptr CPUConvOp::create(ConvParam conv_param,
                                  RamTensor::sptr input,
                                  RamTensor::sptr output,
                                  RamTensor::sptr weight,
                                  RamTensor::sptr bias) {
    CPUConvOp::sptr ptr = std::make_shared<CPUConvOp>(conv_param, input,
                                                      output, weight, bias);
    ptr->checkOutputDims();
    return ptr;
}

inline CPUConvOp::CPUConvOp() : Operation({}, {}),
                                param_({0, 0, 0, 0, 0, 0}),
                                weight_(nullptr), bias_(nullptr) {}

inline CPUConvOp::CPUConvOp(ConvParam conv_param,
                            RamTensor::sptr input,
                            RamTensor::sptr output,
                            RamTensor::sptr weight,
                            RamTensor::sptr bias)
                           : Operation({input}, {output}), param_(conv_param),
                             weight_(weight), bias_(bias) {}

inline CPUConvOp::~CPUConvOp() {}

inline void CPUConvOp::checkOutputDims() {
    auto input = getInputs()[0];
    auto output = getOutputs()[0];
    if (input->channel != weight_->channel) {
        throw std::runtime_error("CPUConvOp channel of input is wrong!");
    }

    int input_h = input->height + param_.ph;
    int input_w = input->width + param_.pw;
    int kh = param_.dh > 1 ? (weight_->height - 1) * param_.dh + 1
                                                         : weight_->height;
    int kw = param_.dw > 1 ? (weight_->width - 1) * param_.dw + 1
                                                         : weight_->width;
    int output_h = (input_h - kh) / param_.sh + 1;
    int output_w = (input_w - kw) / param_.sw + 1;
    int output_c = weight_->n_batch;
    int output_n = input->n_batch;
    if (output->n_batch != output_n ||
            output->channel != output_c ||
                output->height != output_h ||
                output->width != output_w) {
        throw std::runtime_error("CPUConvOp output shape is wrong!");
    }

    if (input_h < kh) {
        throw std::runtime_error("CPUConvOp kernel_h is wrong");
    }

    if (input_w < kw) {
        throw std::runtime_error("CPUConvOp kernel_w is wrong!");
    }
}

inline void CPUConvOp::forward_compute() {
    auto input_tensor = getInputs()[0];
    auto output_tensor = getOutputs()[0];

    float* input = reinterpret_cast<float *>(input_tensor->data_ptr);
    float* output = reinterpret_cast<float *>(output_tensor->data_ptr);
    float* weight = reinterpret_cast<float *>(weight_->data_ptr);
    float* bias = bias_ ? reinterpret_cast<float *>(bias_->data_ptr) : nullptr;

    int ni = input_tensor->n_batch;
    int ci = input_tensor->channel;
    int hi = input_tensor->height;
    int wi = input_tensor->width;
    int co = output_tensor->channel;
    int ho = output_tensor->height;
    int wo = output_tensor->width;
    int sh = param_.sh;
    int sw = param_.sw;
    int kh = weight_->height;
    int kw = weight_->width;
    int dh = param_.dh;
    int dw = param_.dw;
    int ph = param_.ph;
    int pw = param_.pw;

    // dilated
    float* temp_weight = nullptr;
}

}  // namespace rvos
