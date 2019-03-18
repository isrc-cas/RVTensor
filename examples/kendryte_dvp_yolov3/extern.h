/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#ifndef _EXTERN_H
#define _EXTERN_H

extern void create_executor(void** ptr, char* model_name, int thread_num);

extern void analysis_model(void* ptr);

extern void load_image_by_buf(void* ptr, uint8_t* ai_buf, int channel,
                              int height, int width);

extern void load_image_by_path(void* ptr, char* image_path, int channel,
                               int height, int width);

extern void compute_model(void* ptr);

extern void inference_result(void* ptr,
                      void* result_buf,
                      uint64_t size,
                      void* call);
#endif
