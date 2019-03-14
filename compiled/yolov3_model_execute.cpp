// Auto generated by rvos_compiler

#include "include/core/graph.hpp"
#include "include/operation/add.hpp"
#include "include/operation/conv.hpp"
#include "yolov3_model_execute.hpp"
// #include "graph_cpu_0.hpp"
// #include "graph_kpu_0.hpp"
// #include "graph_cpu_1.hpp"
#include "yolov3_model_data.hpp"

RamTensor::sptr yolov3_model_execute(RamTensor::sptr input_yolov3_0) {
{
    RamTensor::sptr graph_cpu_0_input0 = input_yolov3_0;
    RamTensor::sptr graph_kpu_0_input0 = input_yolov3_0;
}
{
    RamTensor::sptr graph_cpu_0_output0 = RamTensor::create(20, 30, 1, 4u);
    graph_cpu_0_execute(graph_cpu_0_input0, graph_cpu_0_output0);
}
{
    RamTensor::sptr graph_kpu_0_output0 = RamTensor::create(20, 30, 1, 4u);
    graph_kpu_0_execute(graph_kpu_0_input0, graph_kpu_0_output0);
}
{
    RamTensor::sptr graph_cpu_1_output0 = RamTensor::create(20, 30, 1, 4u);
    graph_cpu_1_execute(graph_cpu_0_output0, graph_kpu_0_output0,
                        graph_cpu_1_output0);
}
{
    return graph_cpu_1_output0;
}
}

void graph_cpu_0_execute(RamTensor::sptr input_0, RamTensor::sptr output_0) {
{
    RamTensor::sptr conv_cpu_0_input_0 = input_0;
    RamTensor::sptr conv_cpu_0_output_0 = output_0;
}
{
    FlashTensor::sptr conv_cpu_0_weight_fix8 =
            FlashTensor::create(20, 30, 1, conv_cpu_0_weight_fix8_data, 1u);
    FlashTensor::sptr conv_cpu_0_bias_fix8 =
            FlashTensor::create(20, 1, 1, conv_cpu_0_bias_fix8_data, 1u);
    ConvParam conv_cpu_0_param = {1, 1, 1, 1, 1, 1, 0, 0};
    CPUConvOp::sptr conv_cpu_0_fix8 =
                  CPUConvOp::create(conv_cpu_0_param, conv_cpu_0_input_0,
                                    conv_cpu_0_output_0, conv_cpu_0_weight_fix8,
                                    conv_cpu_0_weight_fix8);
    conv_cpu_0_fix8->compute();
    delete conv_cpu_0_fix8;
}
}

void graph_cpu_1_execute(RamTensor::sptr input_0, RamTensor::sptr input_1,
                                                  RamTensor::sptr output_0) {
{
    RamTensor::sptr add_cpu_1_input_0 = input_0;
    RamTensor::sptr add_cpu_1_input_1 = input_1;
    RamTensor::sptr add_cpu_1_output_0 = output_0;

}
{
    CPUAddOp::sptr add_cpu_1_fix8 = CPUAddOp::create(add_cpu_1_input_0
                                   add_cpu_1_input_1, add_cpu_1_output_0);
    add_cpu_1_fix8->compute();
    delete add_cpu_1_fix8;
}
}

void graph_kpu_0_execute(RamTensor::sptr input_0, RamTensor::sptr output_0) {
{
    RamTensor::sptr conv_kpu_0_input_0 = input_0;
    RamTensor::sptr conv_kpu_0_output_0 = output_0;
}
{
    FlashTensor::sptr conv_kpu_0_weight_fix8 =
        FlashTensor::create(20, 30, 1, conv_kpu_0_weight_fix8_data, 1u);
    FlashTensor::sptr conv_kpu_0_bias_fix8 =
        FlashTensor::create(20, 1, 1, conv_kpu_0_bias_fix8_data, 1u);
    ConvParam conv_kpu_0_param = {1, 1, 1, 1, 1, 1, 0, 0};
    KPUConvOp::sptr conv_kpu_0_fix8 =
            KPUConvOp::create(conv_kpu_0_param, conv_kpu_0_input_0,
                        conv_kpu_0_output_0, conv_kpu_0_weight_fix8,
                        conv_kpu_0_weight_fix8);
    conv_kpu_0_fix8->compute();
    delete conv_kpu_0_fix8;
}
}