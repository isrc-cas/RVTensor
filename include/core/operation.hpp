/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_OPERATION_HPP_
#define INCLUDE_CORE_OPERATION_HPP_

#include <vector>
#include <memory>
#include "include/core/tensor.hpp"

namespace rvos {

/**
 * RVOS operation descriptor
 *
 *               Operation
 *                   +
 *                   |
 *       +-----------+-----------+
 *       |                       |
 *       v                       v
 * KPUOperations            CPUOperations
 *
 */
class Operation {
 public:
     using sptr = std::shared_ptr<Operation>;
     static sptr create();
     static sptr create(std::vector<RamTensor::sptr> inputs,
                        std::vector<RamTensor::sptr> outputs);

     /**
      * Constructor & Deconstructor
      */
     Operation();
     Operation(std::vector<RamTensor::sptr> inputs,
              std::vector<RamTensor::sptr> outputs);
     ~Operation();
     Operation& operator=(const Operation& op);

     /**
      * check output dims
      */
     virtual void checkOutputDims() {}

     /**
      * get inputs/outputs
      */
     std::vector<RamTensor::sptr> getInputs();
     std::vector<RamTensor::sptr> getOutputs();

     /**
      * inference
      */
     virtual void forward_compute() {}

 private:
     std::vector<RamTensor::sptr> inputs_;
     std::vector<RamTensor::sptr> outputs_;
};

}  // namespace rvos

#endif  // INCLUDE_CORE_OPERATION_HPP_
