/*  The MIT License
 *
 *  Copyright (c) 2019, Institute of Software Chinese Academy of Sciences(ISCAS)
 *  All rights reserved.
 *
 */

extern handle_t dma_open_free();

extern void dma_set_request_source(handle_t file, uint32_t request);

extern void dma_transmit(handle_t file, const volatile void *src,
                         volatile void *dest, bool src_inc, bool dest_inc,
                         size_t element_size, size_t count, size_t burst_size);

extern void dma_close(handle_t file);

