/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include "include/core/executor.hpp"
#include "include/core/rvos_api.h"

void create_executor(void** pptr, char* model_name, int input_height,
                     int input_width, int thread_num) {
  if (pptr == NULL)
    exit(0);
  *pptr = reinterpret_cast<void*>((rvos::Executor::create(model_name,
                                input_height, input_width, thread_num)).get());
}

void analysis_model(void* ptr) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->analysisModel();
}

void load_image_by_buf(void* ptr, uint8_t* ai_buf, int height, int width) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->loadImage(
                                                        ai_buf, height, width);
}

void load_image_by_path(void* ptr, char* image_path, int height, int width) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->loadImage(
                                                    image_path, height, width);
}

void copy_data_to_ai(void* ptr, void* host_buf, void* ai_base, uint64_t size) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->copyData(
                                    host_buf, ai_base, size, rvos::HOST_TO_AI);
}

void copy_data_to_host(void* ptr, void* host_buf,
                       void* ai_base, uint64_t size) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->copyData(
                                    host_buf, ai_base, size, rvos::AI_TO_HOST);
}

void compute_model(void* ptr) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->compute();
}

void inference_result(void* ptr, void* result_buf, uint64_t size, void* call) {
  (*(reinterpret_cast<rvos::Executor::sptr*>(ptr)))->inferenceResult(
      result_buf, size, (rvos::callback_draw_box)call);
}

