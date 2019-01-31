/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

extern void create_executor(void** ptr,
                     char* model_name,
                     int input_height,
                     int input_width,
                     int thread_num);

extern void analysis_model(void* ptr);

extern void load_image_by_buf(void* ptr,
                       uint8_t* ai_buf,
                       int height,
                       int width);

extern void load_image_by_path(void* ptr,
                         char* image_path,
                         int height,
                         int width);

extern void copy_data_to_ai(void* ptr,
                     void* host_buf,
                     void* ai_base,
                     uint64_t size);

extern void copy_data_to_host(void* ptr,
                     void* host_buf,
                     void* ai_base,
                     uint64_t size);

extern void compute_model(void* ptr);

extern void inference_result(void* ptr,
                      void* result_buf,
                      uint64_t size,
                      void* call);
