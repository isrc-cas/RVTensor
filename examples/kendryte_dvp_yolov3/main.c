/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <stdio.h>
#include <FreeRTOS.h>
#include <task.h>
#include "dvp_camera.h"
#include "lcd.h"
#include "ov5640.h"

#include "extern.h"

uint8_t g_ai_buf[320 * 240 *3] __attribute__((aligned(128)));
uint8_t g_ai_output[1 *3] __attribute__((aligned(128)));

void vTaskYolov3(void* param)
{
    while (1)
    {
        while (dvp_finish_flag == 0)
            ;
        // start to inference
        void* exe;
        create_executor(&exe, "yolov3", 320, 240, 1);
        analysis_model(exe);
        load_image_by_buf(exe, g_ai_buf, 320, 240);
        copy_data_to_ai(exe, (void *)g_ai_buf, (void*)0x40600000, 320 * 240 * 3);
        compute_model(exe);
        copy_data_to_host(exe, (void *)g_ai_output, (void*)0x40600000, 3);

        // display pic
        dvp_finish_flag = 0;
        lcd_draw_picture(0, 0, 320, 240, gram_mux ? lcd_gram1 : lcd_gram0);
        gram_mux ^= 0x01;

        // draw boxes
        inference_result(exe, (void*)g_ai_output, 3, NULL);
    }
}

int main(void)
{
    printf("lcd init\n");
    lcd_init();
    printf("DVP init\n");
    dvp_init();
    ov5640_init();

    vTaskSuspendAll();
    xTaskCreate(vTaskYolov3, "vTaskYolov3", 1024, NULL, 3, NULL);
    if(!xTaskResumeAll())
    {
        taskYIELD();
    }

    while (1)
        ;
}

