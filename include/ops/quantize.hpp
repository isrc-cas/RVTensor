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
#include "include/core/types.hpp"

namespace rvos {

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
