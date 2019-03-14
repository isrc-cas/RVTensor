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
                                param_({0, 0, 1, 1, 0, 0, false}),
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
    int stepi = input_tensor->cstep;
    int co = output_tensor->channel;
    int ho = output_tensor->height;
    int wo = output_tensor->width;
    int stepo = output_tensor->cstep;
    int sh = param_.sh;
    int sw = param_.sw;
    int kh = weight_->height;
    int kw = weight_->width;
    int dh = param_.dh;
    int dw = param_.dw;
    int ph = param_.ph;
    int pw = param_.pw;

    float* temp_weight = nullptr;
    int x = 0, y = 0;
    if (dh > 1 || dw > 1) {
        kh = (kh - 1) * dh + 1;
        kw = (kw - 1) * dw + 1;
        temp_weight = (float *)malloc(sizeof(float) * kw * kh * ci * co);
        x = -1;
        y = -1;
        for (int coi = 0; coi < co; coi++) {
            for (int cii = 0; cii < ci; cii++) {
                for (int khi = 0; khi < kh; khi++) {
                    for (int kwi = 0; kwi < kw; kwi++) {
                        x++;
                        if (khi % dh != 0 || kwi % dw != 0) {
                            temp_weight[x] = 0;
                        } else {
                            y++;
                            temp_weight[x] = weight[y];
                        }
                    }
                }
            }
        }
    } else {
        temp_weight[x] = weight[y];
    }

    for (int n = 0; n < ni; n++) {
        for (int coo = 0; coo < co; coo++) {
            for (int hoo = 0; hoo < ho; hoo++) {
                for (int woo = 0; woo < wo; woo++) {
                    int start_w = sw * woo - pw / 2;
                    int start_h = sh * hoo - ph / 2;
                    int end_w = (std::min)(start_w + kw, wi);
                    int end_h = (std::min)(start_h + kh, hi);
                    int kernel_shift_w = (start_w < 0) ? -start_w : 0;
                    int kernel_shift_h = (start_h < 0) ? -start_h : 0;
                    int rem_dw = kernel_shift_w % dw;
                    int rem_dh = kernel_shift_h % dh;
                    int kernel_shift_dw = (rem_dw > 0) ? dw - rem_dw : 0;
                    int kernel_shift_dh = (rem_dh > 0) ? dh - rem_dh : 0;
                    start_w = (std::max)(start_w, kernel_shift_dw);
                    start_h = (std::max)(start_h, kernel_shift_dh);
                    output[n * co * stepo + coo * stepo + hoo * wo + woo] = 0; // NOPLINT
                    for (int cii = 0; cii < ci; cii++) {
                        for (int h = start_h; h < end_h; h += dh) {
                            for (int w = start_w; w < end_w; w += dw) {
                                output[n * co * stepo + coo * stepo + hoo * wo + woo] += // NOPLINT
                                    input[n * ci *  stepi + cii *  stepi + h * wi + w] * // NOPLINT
                                        temp_weight[coo * ci * kh * kw + cii * kh * kw + // NOPLINT
                                            (kernel_shift_h + kernel_shift_dh + h - start_h) * kw + // NOPLINT
                                            (kernel_shift_w + kernel_shift_dw + w - start_w)]; // NOPLINT
                            }
                        }
                    }
                    if (bias != nullptr) {
                        output[n * co * stepo + coo * stepo + hoo * wo + woo] += bias[coo]; // NOPLINT
                    }
                }
            }
        }
    }
    if (dh > 1 || dw > 1) {
        free(temp_weight);
    }
}

}  // namespace rvos
