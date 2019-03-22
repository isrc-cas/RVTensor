/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

#include <type_traits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <sys/time.h> // NOLINT
#include "include/ops/kpu/kpu_conv.hpp"
#include "include/ops/kpu/kpu_extern.h"

namespace RVTensor {

KPUConvOp::sptr KPUConvOp::create() {
  return std::make_shared<KPUConvOp>();
}

KPUConvOp::sptr KPUConvOp::create(ConvParam conv_param, RamTensor::sptr input,
                             RamTensor::sptr output, FlashTensor::sptr weight,
                             FlashTensor::sptr bias) {
  KPUConvOp::sptr ptr = std::make_shared<KPUConvOp>(conv_param, input,
                                                    output, weight, bias);
  ptr->checkOutputDims();
  return ptr;
}

inline KPUConvOp::KPUConvOp() : Operation({}, {}),
       param_({0, 0, 1, 1, 0, 0, false}),
       weight_(nullptr), bias_(nullptr) {}

inline KPUConvOp::KPUConvOp(ConvParam conv_param, RamTensor::sptr input,
                            RamTensor::sptr output, FlashTensor::sptr weight,
                            FlashTensor::sptr bias)
  : Operation({input}, {output}), param_(conv_param),
  weight_(weight), bias_(bias) {}

inline KPUConvOp::~KPUConvOp() {}

inline void KPUConvOp::checkOutputDims() {
  auto input = getInputs()[0];
  auto output = getOutputs()[0];
  if (input->channel != weight_->channel) {
    throw std::runtime_error("KPUConvOp channel of input is wrong!");
  }

  int input_h = input->height + param_.ph;
  int input_w = input->width + param_.pw;
  int kh = param_.dh > 1 ? (weight_->height - 1) * param_.dh + 1
           : weight_->height;
  int kw = param_.dw > 1 ? (weight_->width - 1) * param_.dw + 1
           : weight_->width;
  int output_h = (input_h - kh) / param_.sh + 1;
  int output_w = (input_w - kw) / param_.sw + 1;
  int output_c = weight_->n_batch;
  int output_n = input->n_batch;
  if (output->n_batch != output_n || output->channel != output_c ||
      output->height != output_h || output->width != output_w) {
    throw std::runtime_error("KPUConvOp output shape is wrong!");
  }

  if (input_h < kh) {
    throw std::runtime_error("KPUConvOp kernel_h is wrong");
  }

  if (input_w < kw) {
    throw std::runtime_error("KPUConvOp kernel_w is wrong!");
  }
}

// note: this implementation does not disable this overload for array types
template<class T>
std::unique_ptr<T> make_unique(size_t n) {
  typedef typename std::remove_extent<T>::type U;
  return std::unique_ptr<T>(new U[n]());
}

template<class T>
inline uint64_t to_ui64(const T& reg) {
  union {
    T reg;
    uint64_t data;
  } u;
  u.reg = reg;
  return u.data;
}

inline void KPUConvOp::forward_compute() {
  auto input_tensor = getInputs()[0];
  auto output_tensor = getOutputs()[0];

  auto quantize_multiplier = [&] (int32_t* pm, int* ps) {
    double real_multiplier = input_tensor->scale * weight_->scale /
                             output_tensor->scale;
    const double q = std::frexp(real_multiplier, ps);
    auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));
    if (q_fixed == (1ll << 31)) {
      q_fixed /= 2;
      ++*ps;
    }
    *pm = static_cast<int32_t>(q_fixed);
  };

  const int32_t input_offset = input_tensor->zero_point;
  const int32_t weight_offset = weight_->zero_point;
  const int32_t output_offset = 0;
  int32_t multiplier;
  int shift;
  quantize_multiplier(&multiplier, &shift);

  float* input = reinterpret_cast<float *>(input_tensor->data_ptr);
  float* output = reinterpret_cast<float *>(output_tensor->data_ptr);
  float* weight = reinterpret_cast<float *>(weight_->data_ptr);
  float* bias = bias_ ? reinterpret_cast<float *>(bias_->data_ptr) : nullptr;

  uint32_t ni = input_tensor->n_batch;
  uint32_t ci = input_tensor->channel;
  uint32_t hi = input_tensor->height;
  uint32_t wi = input_tensor->width;
  uint32_t stepi = input_tensor->cstep;
  uint32_t co = output_tensor->channel;
  uint32_t ho = output_tensor->height;
  uint32_t wo = output_tensor->width;
  uint32_t stepo = output_tensor->cstep;
  uint32_t sh = param_.sh;
  uint32_t sw = param_.sw;
  uint32_t kh = weight_->height;
  uint32_t kw = weight_->width;
  uint32_t stepw = weight_->cstep;
  uint32_t dh = param_.dh;
  uint32_t dw = param_.dw;
  uint32_t ph = param_.ph;
  uint32_t pw = param_.pw;

  uint32_t in_row_padding;
  uint32_t in_row_group;
  uint32_t in_row_length;
  if (wi <= 16) {
    in_row_padding = 16;
    in_row_group = 4;
    in_row_length = 1;
  } else if (wi <= 32) {
    in_row_padding = 32;
    in_row_group = 2;
    in_row_length = 1;
  } else {
    in_row_padding = 64;
    in_row_group = 1;
    in_row_length = (wi + 63) / 64;
  }

  uint32_t out_row_padding;
  uint32_t out_row_group;
  uint32_t out_row_length;
  if (wo <= 16) {
    out_row_padding = 16;
    out_row_group = 4;
    out_row_length = 1;
  } else if (wo <= 32) {
    out_row_padding = 32;
    out_row_group = 2;
    out_row_length = 1;
  } else {
    out_row_padding = 64;
    out_row_group = 1;
    out_row_length = (wo + 63) / 64;
  }

  const uint32_t in_channels_of_group = std::min(ci, in_row_group);
  const uint32_t out_channels_of_group = std::min(co, out_row_group);
  const uint32_t out_channel_kernel_size = kw * kh * ci;
  const uint32_t one_time_kernel_out_channels =
                    std::min(co, 16 * 1024 / out_channel_kernel_size);
  const uint32_t load_time = static_cast<uint32_t>(std::ceil(
        static_cast<double>(co) / one_time_kernel_out_channels));

#if KPU_DEBUG
  printk("kpu conv: x %d >> %d,  i: %dx%d o:%dx%d, %d, %d, %d\n",
      multiplier, -shift, wi, hi, wo, ho,
      out_channel_kernel_size, one_time_kernel_out_channels, load_time);
  printk("io: %d, oo: %d\n ", input_offset, output_offset);
#endif

  struct timeval tv, tv2;
  gettimeofday(&tv, NULL);
  kpu_layer_argument_t layer;
  layer.interrupt_enabe.data = {
    .int_en = 1,
    .ram_flag = 0,
    .full_add = 0,
    .depth_wise_layer = 0
  };
  layer.image_addr.data = {
    .image_src_addr = (uint64_t)0x0,
    .image_dst_addr = (uint64_t)(0x8000 - (64 * out_row_length * ho * co /
                      out_channels_of_group + 63) / 64)
  };
  layer.image_channel_num.data = {
    .i_ch_num = ci - 1,
    .o_ch_num = co - 1,
    .o_ch_num_coef = one_time_kernel_out_channels - 1
  };
  layer.image_size.data = {
    .i_row_wid = wi - 1,
    .i_col_high = hi - 1,
    .o_row_wid = wo - 1,
    .o_col_high = ho - 1
  };
  layer.kernel_pool_type_cfg.data = {
    .kernel_type = kw == 3 ? 1U : 0,
    .pad_type = 0,
    .pool_type = sw == 2 ? 6U : 0,
    .first_stride = 0,
    .bypass_conv = 0,
    .load_para = 1,
    .dma_burst_size = 15,
    .pad_value = reinterpret_cast<uint64_t>(input),
    .bwsx_base_addr = 0
  };
  layer.kernel_load_cfg.data = {
    .load_coor = 1,
    .load_time = load_time - 1,
    .para_size = out_channel_kernel_size * one_time_kernel_out_channels,
    .para_start_addr = 0
  };
  layer.kernel_offset.data = {
    .coef_column_offset = 0,
    .coef_row_offset = 0
  };
  layer.kernel_calc_type_cfg.data = {
    .channel_switch_addr = in_row_length * hi,
    .row_switch_addr = in_row_length,
    .coef_size = 0,
    .coef_group = in_row_group,
    .load_act = 1,
    .active_addr = 0
  };
  layer.write_back_cfg.data = {
    .wb_channel_switch_addr = out_row_length * ho,
    .wb_row_switch_addr = out_row_length,
    .wb_group = out_row_group
  };
  layer.conv_value.data = {
    .shr_w = 0,
    .shr_x = 0,
    .arg_w = static_cast<uint64_t>(input_offset),
    .arg_x = static_cast<uint64_t>(weight_offset)
  };
  layer.conv_value2.data = {
    .arg_add = static_cast<uint64_t>(input_offset * weight_offset * kw * kh)
  };
  layer.dma_parameter.data = {
    .send_data_out = 1,
    .channel_byte_num = wo * ho - 1,
    .dma_total_byte = (wo * ho * co) - 1
  };

  auto kpu_bn_table = make_unique<kpu_batchnorm_argument_t[]>(co);
  uint64_t mul = multiplier >> (12 - shift);
  for (int out_channel = 0; out_channel < co; ++out_channel) {
    int64_t add = bias ? bias[out_channel] : 0;
    add = (add * mul) >> 15;
    add += output_offset << 4;
    kpu_bn_table[out_channel].batchnorm.data = {
      .norm_mul = mul, .norm_add = static_cast<uint64_t>(add), .norm_shift = 15
    };
  }

  auto kernels = make_unique<uint8_t[]>(kw * kh * ci * co);
  auto ai_inputs = reinterpret_cast<uint8_t*>(AI_IO_BASE_ADDR);

  size_t output_size =
         ((layer.dma_parameter.data.dma_total_byte + 1) + 7) / 8 * 8;
  auto ai_outputs = make_unique<uint8_t[]>(output_size);

  layer.kernel_pool_type_cfg.data.bwsx_base_addr = (uint64_t)kpu_bn_table.get();
  layer.kernel_calc_type_cfg.data.active_addr = (uint64_t)&kpu_act_table;
  layer.kernel_load_cfg.data.para_start_addr = (uint64_t)kernels.get();

  // init act
  kpu_act_table.activate_para[0].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x800000000};
  kpu_act_table.activate_para[1].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xf7d4cf4b8};
  kpu_act_table.activate_para[2].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xf8ed5a20c};
  kpu_act_table.activate_para[3].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfa05e4f60};
  kpu_act_table.activate_para[4].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfb2e05baa};
  kpu_act_table.activate_para[5].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfc46908fe};
  kpu_act_table.activate_para[6].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfd5f1b652};
  kpu_act_table.activate_para[7].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfe77a63a6};
  kpu_act_table.activate_para[8].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xff9fc6ff0};
  kpu_act_table.activate_para[9].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0xfffd4a9b7};
  kpu_act_table.activate_para[10].data =
  { .shift_number = 4, .y_mul = 1, .x_start = 0};
  kpu_act_table.activate_para[11].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x1d0dca98};
  kpu_act_table.activate_para[12].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x2e9677ec};
  kpu_act_table.activate_para[13].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x401f253f};
  kpu_act_table.activate_para[14].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x52a1318a};
  kpu_act_table.activate_para[15].data =
  { .shift_number = 0, .y_mul = 0, .x_start = 0x6429dedd};
  kpu_act_table.activate_para_bias0.data = {
    .result_bias = {0, 0, 0, 0, 0, 0, 0, 0}
  };
  kpu_act_table.activate_para_bias1.data = {
    .result_bias = {0, 0, 0, 0, 0, 0, 0, 0}
  };

  // init kernels
  uint8_t *k_it = kernels.get();
  for (int out_channel = 0; out_channel < co; ++out_channel) {
    for (int in_channel = 0; in_channel < ci; ++in_channel) {
      for (int filter_y = 0; filter_y < kh; ++filter_y) {
        for (int filter_x = 0; filter_x < kw; ++filter_x) {
          *k_it++ = weight[out_channel * ci * kh * kw +
                           in_channel * kh * kw + filter_y * kw + filter_x];
        }
      }
    }
  }
#if KPU_DEBUG
  printk("kernels\n");
  for (int i = 0; i < 64; i++)
    printk("%p: %d ", kernels.get(), kernels[i]);
#endif

  for (int batch = 0; batch < ni; ++batch) {
    // init inputs
    for (int in_channel = 0; in_channel < ci; ++in_channel) {
      auto channel_origin = ai_inputs +
                            ci / in_row_group * in_row_length * hi * 64 +
                            ci % in_row_group * in_row_padding;
      for (int in_y = 0; in_y < hi; ++in_y) {
        auto y_origin = channel_origin + in_y * in_row_length * 64;
        for (int in_x = 0; in_x < wi; ++in_x) {
          y_origin[in_x] = input[batch * ci * hi * wi + in_channel * hi * wi +
                                 in_y * wi + in_x];
        }
      }
    }

#if KPU_DEBUG
    printk("ai_inputs\n");
    for (int i = 0; i < 64; i++)
      printk("%d ", input_data[i]);
#endif

    volatile kpu_config_t *const kpu = (volatile kpu_config_t *)AI_BASE_ADDR;
    kpu->interrupt_clear.reg = to_ui64(kpu_config_interrupt_t {
        .calc_done_int = 1,
        .layer_cfg_almost_empty_int = 1,
        .layer_cfg_almost_full_int = 1
    });
    kpu->eight_bit_mode.reg = to_ui64(kpu_config_eight_bit_mode_t {
        .eight_bit_mode = 1
    });
    kpu->fifo_threshold.reg = to_ui64(kpu_config_fifo_threshold_t {
        .fifo_full_threshold = 10, .fifo_empty_threshold = 1
    });
    kpu->interrupt_mask.reg = to_ui64(kpu_config_interrupt_t {
        .calc_done_int = 0,
        .layer_cfg_almost_empty_int = 1,
        .layer_cfg_almost_full_int = 1
    });

    kpu->layer_argument_fifo = layer.interrupt_enabe.reg;
    kpu->layer_argument_fifo = layer.image_addr.reg;
    kpu->layer_argument_fifo = layer.image_channel_num.reg;
    kpu->layer_argument_fifo = layer.image_size.reg;
    kpu->layer_argument_fifo = layer.kernel_pool_type_cfg.reg;
    kpu->layer_argument_fifo = layer.kernel_load_cfg.reg;
    kpu->layer_argument_fifo = layer.kernel_offset.reg;
    kpu->layer_argument_fifo = layer.kernel_calc_type_cfg.reg;
    kpu->layer_argument_fifo = layer.write_back_cfg.reg;
    kpu->layer_argument_fifo = layer.conv_value.reg;
    kpu->layer_argument_fifo = layer.conv_value2.reg;
    kpu->layer_argument_fifo = layer.dma_parameter.reg;

    handle_t dma = dma_open_free();
    dma_set_request_source(dma, SYSCTL_DMA_SELECT_AI_RX_REQ);
    dma_transmit(dma, (void*)(&kpu->fifo_data_out),  // NOLINT
                 ai_outputs.get(), false, true, 8,
                 (layer.dma_parameter.data.dma_total_byte + 8) / 8, 8);
    dma_close(dma);

    uint8_t *o_it = ai_outputs.get();
    for (int out_channel = 0; out_channel < co; ++out_channel) {
      for (int out_y = 0; out_y < ho; ++out_y) {
        for (int out_x = 0; out_x < wo; ++out_x) {
          output[batch * co * ho * wo + out_channel * ho * wo +
                 out_y * wo + out_x] = *o_it++;
        }
      }
    }
#if KPU_DEBUG
    printk("ai_outputs\n");
    for (size_t i = 0; i < 64; i++)
      printk("%d ", output[i]);
    gettimeofday(&tv2, NULL);
    printk("\nconv used %dms.\n",
        static_cast<int>((tv2.tv_sec * 1000 + tv2.tv_usec / 1e3) -
          (tv.tv_sec * 1000 + tv.tv_usec / 1e3)));
#endif
  }
}

}  // namespace RVTensor
