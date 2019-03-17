/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_OPS_KPU_KPU_CONV_HPP_
#define INCLUDE_OPS_KPU_KPU_CONV_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"
#include "include/core/operation.hpp"
#include "include/ops/conv.hpp"
#include "include/ops/kpu/kpu.h"

namespace rvos {

class KPUConvOp: public Operation {
 public:
    using sptr = std::shared_ptr<KPUConvOp>;
    static sptr create();
    static sptr create(ConvParam conv_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        FlashTensor::sptr weight,
        FlashTensor::sptr bias = nullptr);

    /**
     * Constructor & Deconstructor
     */
    KPUConvOp();
    KPUConvOp(ConvParam conv_param,
        RamTensor::sptr input,
        RamTensor::sptr output,
        FlashTensor::sptr weight,
        FlashTensor::sptr bias = nullptr);
    ~KPUConvOp();
    KPUConvOp& operator=(const KPUConvOp& conv_op);

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
    FlashTensor::sptr weight_;
    /// model data: bias
    FlashTensor::sptr bias_;
    /// kpu active table
    kpu_activate_table_t kpu_act_table __attribute__((aligned(256)));
};

}  // namespace rvos

#endif  // INCLUDE_OPS_KPU_KPU_CONV_HPP_
