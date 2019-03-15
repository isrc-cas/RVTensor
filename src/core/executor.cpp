/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/tensor.hpp"
// #include "compiled/yolov3_model_execute.hpp"

namespace rvos {

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

int Executor::analysisModel() {
  return 0;
}

void Executor::loadImage(uint8_t* ai_buf, int height, int width) {
  return;
}

void Executor::loadImage(std::string image_path, int height, int width) {
  return;
}

int Executor::copyData(void* host_buf, void* ai_base, uint64_t size,
    Direction_t dir) {
  return 0;
}

int Executor::compute() {
  // RamTensor::sptr output = yolov3_model_execute(image_ptr);
  return 0;
}

int Executor::inferenceResult(void* result_buf, uint64_t size,
    callback_draw_box call) {
  return 0;
}

Executor::~Executor() {}

}  // namespace rvos
