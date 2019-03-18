/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/rvos_api.h"

void create_executor(void** pptr, char* model_name, int thread_num) {
  if (pptr == NULL)
    exit(0);
  *pptr = reinterpret_cast<void*>((rvos::Executor::create(model_name,
                                   thread_num)).get());
}

void load_image_by_buf(void* ptr, uint8_t* ai_buf,
                       int channel, int height, int width) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->loadImage(
                                        ai_buf, channel, height, width);
}

void load_image_by_path(void* ptr, char* image_path,
                        int channel, int height, int width) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->loadImage(
                                        image_path, channel, height, width);
}

void compute_model(void* ptr) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->compute();
}

void inference_result(void* ptr, void* result_buf, uint64_t size, void* call) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->inferenceResult(
      result_buf, size, (rvos::callback_draw_box)call);
}
