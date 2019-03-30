/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/tensor.hpp"
#include "compiled/model_execute.hpp"

namespace RVTensor {

#define MODEL_EXECUTE(model_name, ...) \
  model_name##_model_execute(__VA_ARGS__)

Executor::sptr Executor::create() {
  return std::make_shared<Executor>();
}

Executor::sptr Executor::create(std::string model_name, int thread_num) {
  return std::make_shared<Executor>(model_name, thread_num);
}

Executor::Executor() {}

Executor::Executor(std::string model_name, int thread_num)
                  : thread_num_(thread_num), model_name_(model_name),
                  image_ptr(nullptr), output_ptr(nullptr) {}

void Executor::loadImage(uint8_t* ai_buf, int channel, int height, int width) {
  image_ptr = RamTensor::create(1, channel, height, width,
                                reinterpret_cast<void*>(ai_buf), 1u);
}

// void Executor::loadImage(std::string image_path, int channel,
//                                                  int height, int width) {
// }

int Executor::compute() {
  if (model_name_.compare("yolov3") == 0) {
    // fill image_ptr with test data
    // RamTensor::sptr test_ptr = RamTensor::create(1, 3, 240, 320, 1u);
    // uint8_t v = 1;
    // test_ptr->fill(v);
    output_ptr = MODEL_EXECUTE(yolov3, test_ptr);
  } else {
    return -1;
  }
  return 0;
}

void Executor::copyOutputData(void* data_ptr, size_t size) {
  if (output_ptr->trueSize() != size) {
    throw std::runtime_error("copyOutputData data size is wrong!");
  }
  size_t surface_size = size / output_ptr->channel;
  for (int c = 0; c < output_ptr->channel; c++) {
    void* src_ptr = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(output_ptr->data_ptr) +
                    c * output_ptr->cstep * output_ptr->element_size);
    void* dst_ptr = reinterpret_cast<void*>(
                    reinterpret_cast<uint8_t*>(data_ptr) + c * surface_size);
    memcpy(dst_ptr, src_ptr, surface_size);
  }
}

int Executor::inferenceResult(void* result_buf, uint64_t size,
    callback_draw_box call) {
  return 0;
}

Executor::~Executor() {}

}  // namespace RVTensor
