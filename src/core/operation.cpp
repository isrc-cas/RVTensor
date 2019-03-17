/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/operation.hpp"

namespace rvos {

Operation::sptr Operation::create() {
    return std::make_shared<Operation>();
}

Operation::sptr Operation::create(std::vector<RamTensor::sptr> inputs,
                                  std::vector<RamTensor::sptr> outputs) {
    return std::make_shared<Operation>(inputs, outputs);
}

Operation::Operation() : inputs_({}), outputs_({}) {}

Operation::Operation(std::vector<RamTensor::sptr> inputs,
                     std::vector<RamTensor::sptr> outputs)
                    : inputs_(inputs), outputs_(outputs) {}

Operation::~Operation() {}

std::vector<RamTensor::sptr> Operation::getInputs() {
    return inputs_;
}

std::vector<RamTensor::sptr> Operation::getOutputs() {
    return outputs_;
}

}  // namespace rvos
