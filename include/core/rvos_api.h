/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef INCLUDE_CORE_RVOS_API_H_
#define INCLUDE_CORE_RVOS_API_H_

extern "C"
void create_executor(void** pptr, char* model_name, int thread_num);

extern "C"
void analysis_model(void* ptr);

extern "C"
void load_image_by_buf(void* ptr, uint8_t* ai_buf, int channel, int height,
                       int width);

extern "C"
void load_image_by_path(void* ptr, char* image_path, int channel, int height,
                        int width);

extern "C"
void compute_model(void* ptr);

extern "C"
void copy_output_buf(void* ptr, void* data_ptr, size_t size);

extern "C"
void inference_result(void* ptr,
                      void* result_buf,
                      uint64_t size,
                      void* call);

#endif  // INCLUDE_CORE_RVOS_API_H_
