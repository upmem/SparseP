/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 */

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <seqread.h>
#include <stdint.h>
#include <stdio.h>

#include "../support/common.h"
#include "../support/utils.h"
#include "common.h"
#include <alloc.h>
#include <assert.h>
#include <barrier.h>
#include <built_ins.h>
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <perfcounter.h>
#include <seqread.h>
#include <stdint.h>
#include <stdio.h>

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
uint32_t nnz_offset;

/**
 * @brief kernel time measurement
 */
perfcounter_t start_time;
__host perfcounter_t kernel_cycles = 0;

/**
 * @brief main function executed by each tasklet
 */

int main() {
  uint32_t tasklet_id = me();
  perfcounter_config(COUNT_CYCLES, true);

  start_time = perfcounter_get();

  if (tasklet_id == 0) {
    mem_reset(); // Reset the heap
  }

  // Barrier
  barrier_wait(&my_barrier);

  // Load parameters
  uint32_t nrows = DPU_INPUT_ARGUMENTS.nrows;
  uint32_t max_rows_per_tasklet = DPU_INPUT_ARGUMENTS.max_rows_per_tasklet;
  uint64_t rank_max_rows_per_tasklet = DPU_INPUT_ARGUMENTS.rank_max_rows_per_tasklet;
  uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
  uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
  uint32_t start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
  uint32_t start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];
  uint32_t nnz_per_tasklet = DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id];

  // Find start addresses in MRAM
  uint32_t mram_base_addr_y = (uint32_t)(DPU_MRAM_HEAP_POINTER);
  uint32_t mram_temp_addr_y;
  uint32_t mram_base_addr_x =
      (uint32_t)(DPU_MRAM_HEAP_POINTER +
                 (rank_max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt)));
  uint32_t mram_base_addr_elems =
      (uint32_t)(mram_base_addr_x + (tcols * sizeof(val_dt)));
  mram_base_addr_y =
      (uint32_t)(DPU_MRAM_HEAP_POINTER +
                 (tasklet_id * max_rows_per_tasklet * sizeof(val_dt)));

  uint32_t i;
  // Initialize input vector cache
  val_dt *cache_x = mem_alloc(8);
  // Initialize output vector cache
  val_dt *cache_y = mem_alloc(8);

  // Use cache_y cache to initialize the output vector elements in MRAM with
  // zeros
#if INT8
  cache_y[0] = 0;
  cache_y[1] = 0;
  cache_y[2] = 0;
  cache_y[3] = 0;
  cache_y[4] = 0;
  cache_y[5] = 0;
  cache_y[6] = 0;
  cache_y[7] = 0;
#elif INT16
  cache_y[0] = 0;
  cache_y[1] = 0;
  cache_y[2] = 0;
  cache_y[3] = 0;
#elif INT32
  cache_y[0] = 0;
  cache_y[1] = 0;
#elif INT64
  cache_y[0] = 0;
#elif FP32
  cache_y[0] = 0;
  cache_y[1] = 0;
#elif FP64
  cache_y[0] = 0;
#else
  cache_y[0] = 0;
  cache_y[1] = 0;
#endif

  if (tasklet_id == 0) {
    mram_temp_addr_y = mram_base_addr_y;
    uint32_t iter = 0;
#if INT8
    iter = ((NR_TASKLETS * max_rows_per_tasklet) >> 3);
#elif INT16
    iter = ((NR_TASKLETS * max_rows_per_tasklet) >> 2);
#elif INT32
    iter = ((NR_TASKLETS * max_rows_per_tasklet) >> 1);
#elif INT64
    iter = NR_TASKLETS * max_rows_per_tasklet;
#elif FP32
    iter = ((NR_TASKLETS * max_rows_per_tasklet) >> 1);
#elif FP64
    iter = NR_TASKLETS * max_rows_per_tasklet;
#else
    iter = ((NR_TASKLETS * max_rows_per_tasklet) >> 1);
#endif
    for (i = 0; i < iter; i++) {
      mram_write(cache_y, (__mram_ptr void *)(mram_temp_addr_y), 8);
      mram_temp_addr_y += 8;
    }
  }
  barrier_wait(&my_barrier);

  // If there is no work, return
  if (nnz_per_tasklet == 0) {
    goto EXIT;
  }

  // Initialize sequential reader for nnzs
  mram_base_addr_elems += (start_nnz * sizeof(struct elem_t));
  seqreader_buffer_t cache_elems = seqread_alloc();
  seqreader_t sr_elem;
  struct elem_t *cur_elem = seqread_init(
      cache_elems, (__mram_ptr void *)mram_base_addr_elems, &sr_elem);
  uint32_t prev_row = cur_elem->rowind;

  // Initialize help variables
  uint32_t diff;
  val_dt acc = 0;

  // Iterate over nnzs
  for (i = 0; i < nnz_per_tasklet; i++) {
    // If all nnzs of the same row have been traversed, store the final value
    // for the output vector element in MRAM (8-byte alignment to MRAM accesses)
    if (cur_elem->rowind != prev_row) {
      diff = prev_row - start_row - tstart_row;

      if ((diff & 1) == 0) {
        mram_temp_addr_y = (uint32_t)(mram_base_addr_y + (diff << 2));
        cache_y[0] = acc;
        cache_y[1] = 0;
        mram_write(cache_y, (__mram_ptr void *)(mram_temp_addr_y), 8);
      } else {
        diff -= 1;
        mram_temp_addr_y = (uint32_t)(mram_base_addr_y + (diff << 2));
        mram_read((__mram_ptr void *)(mram_temp_addr_y), cache_y, 8);
        cache_y[1] = acc;
        mram_write(cache_y, (__mram_ptr void *)(mram_temp_addr_y), 8);
      }

      acc = 0;
      prev_row = cur_elem->rowind;
    }

    if ((cur_elem->colind & 1) == 0) {
      mram_read((__mram_ptr void const *)(mram_base_addr_x +
                                          cur_elem->colind * sizeof(val_dt)),
                cache_x, 8);
      acc += cur_elem->val * cache_x[0];
    } else {
      mram_read(
          (__mram_ptr void const *)(mram_base_addr_x +
                                    (cur_elem->colind - 1) * sizeof(val_dt)),
          cache_x, 8);
      acc += cur_elem->val * cache_x[1];
    }

    // Get next nnz
    cur_elem = seqread_get(cur_elem, sizeof(struct elem_t), &sr_elem);
  }

  // Store output for the last output vector element in MRAM
  diff = prev_row - start_row - tstart_row;

  if ((diff & 1) == 0) {
    mram_temp_addr_y = (uint32_t)(mram_base_addr_y + (diff << 2));
    cache_y[0] = acc;
    cache_y[1] = 0;
    mram_write(cache_y, (__mram_ptr void *)(mram_temp_addr_y), 8);
  } else {
    diff -= 1;
    mram_temp_addr_y = (uint32_t)(mram_base_addr_y + (diff << 2));
    mram_read((__mram_ptr void *)(mram_temp_addr_y), cache_y, 8);
    cache_y[1] = acc;
    mram_write(cache_y, (__mram_ptr void *)(mram_temp_addr_y), 8);
  }
  kernel_cycles = (perfcounter_get() - start_time);
  // printf(" kern cycles %lu \n", kernel_cycles);
  // printf(" start time %lu \n", kernel_cycles);

EXIT:
  return 0;
}
