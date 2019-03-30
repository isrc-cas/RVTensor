/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <memory>
#include "include/core/executor.hpp"
#include "include/core/rvtensor_api.h"

RVTensor::Executor::sptr sp = nullptr;

void create_executor(void** pptr, char* model_name, int thread_num) {
  if (pptr == NULL)
    exit(0);
  std::string st = model_name;
  sp = RVTensor::Executor::create(st, thread_num);
  *pptr = reinterpret_cast<void*>(&sp);
}

void load_image_by_buf(void* ptr, uint8_t* ai_buf,
                       int channel, int height, int width) {
  (*(reinterpret_cast<RVTensor::Executor::sptr*>(ptr)))->loadImage(
                                        ai_buf, channel, height, width);
}

// void load_image_by_path(void* ptr, char* image_path,
//                         int channel, int height, int width) {
//   (*(reinterpret_cast<RVTensor::Executor::sptr*>(ptr)))->loadImage(
//                                         image_path, channel, height, width);
// }

void compute_model(void* ptr) {
  (*(static_cast<RVTensor::Executor::sptr*>(ptr)))->compute();
}

void copy_output_buf(void* ptr, void* data_ptr, size_t size) {
  (*(static_cast<RVTensor::Executor::sptr*>(ptr)))->copyOutputData(
                                                           data_ptr, size);
}

void inference_result(void* ptr, void* result_buf, uint64_t size, void* call) {
  (*(static_cast<RVTensor::Executor::sptr*>(ptr)))->inferenceResult(
      result_buf, size, (RVTensor::callback_draw_box)call);
}

void destroy_executor(void* ptr) {
  sp.reset();
}
