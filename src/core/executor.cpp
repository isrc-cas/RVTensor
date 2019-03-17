/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/tensor.hpp"
#include "compiled/model_execute.hpp"

namespace rvos {

#define MODEL_EXECUTE(model_name, ...) \
  model_name##_model_execute(__VA_ARGS__)

Executor::sptr Executor::create() {
  return std::make_shared<Executor>();
}

Executor::sptr Executor::create(std::string model_name, int input_height,
                                int input_width, int thread_num) {
  return std::make_shared<Executor>(model_name, input_height,
                                    input_width, thread_num);
}

Executor::Executor() {}

Executor::Executor(std::string model_name, int input_height,
                   int input_width, int thread_num) : thread_num_(thread_num),
                   model_name_(model_name) {}

void Executor::loadImage(uint8_t* ai_buf, int height, int width) {
  return;
}

void Executor::loadImage(std::string image_path, int height, int width) {
  return;
}

int Executor::compute() {
  RamTensor::sptr output = MODEL_EXECUTE(yolov3, image_ptr);
  return 0;
}

int Executor::inferenceResult(void* result_buf, uint64_t size,
    callback_draw_box call) {
  return 0;
}

Executor::~Executor() {}

}  // namespace rvos
