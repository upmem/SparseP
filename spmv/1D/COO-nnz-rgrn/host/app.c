/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 */

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>

#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
#include "../support/partition.h"
#include "../support/timer.h"
#include "../support/utils.h"

// Define the DPU Binary path as DPU_BINARY here.
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/spmv_dpu"
#endif

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MB

/*
 * Main Structures:
 * 1. Matrices
 * 2. Input vector
 * 3. Output vector
 * 4. Help structures for data partitioning
 */
static struct COOMatrix *A;
static val_dt **x;
static val_dt **y;
static struct partition_info_t **part_info;

/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
  uint32_t rows_per_dpu;
  uint32_t rows_per_dpu_pad;
  uint32_t prev_rows_dpu;
  uint32_t prev_nnz_dpu;
  uint32_t nnz;
  uint32_t nnz_pad;
};
struct dpu_info_t **dpu_info;

/**
 * @brief initialize input vector
 * @param pointer to input vector and vector size
 */
void init_vector(val_dt **vec, u_int32_t batch_size, uint32_t input_size) {
  for (unsigned int b = 0; b < batch_size; ++b) {
    for (unsigned int i = 0; i < input_size; ++i) {
      vec[b][i] = (val_dt)(i % 1000 + b);
    }
  }
}

/**
 * @brief compute output in the host CPU
 */
static void spmv_host(val_dt **y, struct COOMatrix *A, val_dt **x,
                      uint32_t batch_size) {
  uint32_t limit = 0;
#pragma omp parallel for
  for (unsigned int b = 0; b < batch_size; b++) {
#pragma omp parallel for
    for (unsigned int n = 0; n < A->nnz; n++) {
      y[b][A->nnzs[n].rowind] += x[b][A->nnzs[n].colind] * A->nnzs[n].val;
    }
  }
}

struct dpu_set_t *dpu_set = NULL;
// +1 is used to deal with ranks with disabled DPUS (NR_DPUS<64) (eg : RANK_0
// (60 DPUS), RANK_1(62 DPUS) ...)
struct dpu_set_t *ranks = NULL;

void free_dpu_pool(dpu_pool_t *this) {
  DPU_ASSERT(dpu_free(*dpu_set));
  free(this->rank_nr_dpus);
  free(this->rank_offset);
  free(this->rank_attached_dpu);
  free(this->rank_dpu_index);
  free(ranks);
  free(dpu_set);
}

void alloc_dpu_pool(dpu_pool_t *this, uint32_t nr_ranks) {

  this->nr_ranks = nr_ranks;
  // 1) allocate DPUs
  assert(dpu_set == NULL);

  dpu_set = calloc(1, sizeof(struct dpu_set_t));
  DPU_ASSERT(dpu_alloc_ranks(this->nr_ranks, NULL, dpu_set));
  DPU_ASSERT(dpu_get_nr_dpus(*dpu_set, &(this->nr_dpus)));
  printf("[INFO] Nr DPUs %u\n", this->nr_dpus);
  printf("[INFO] Alloc DPUs, NR_DPUS %u NR_RANKS %u\n", this->nr_dpus,
         this->nr_ranks);
  printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);

  // 2) load the program into the DPUs
  DPU_ASSERT(dpu_load(*dpu_set, DPU_BINARY, NULL));

  // allocate rank array
  ranks = malloc(this->nr_ranks * sizeof(struct dpu_set_t));

  // allocate dpu_pool array
  {
    this->rank_nr_dpus = malloc(this->nr_ranks * sizeof(uint32_t));
    this->rank_offset = malloc(this->nr_ranks * sizeof(uint32_t));
  }
  {
    // 3) create rank offset, useful for callback to merge results
    struct dpu_set_t rank;
    uint32_t each_rank = 0;
    DPU_RANK_FOREACH(*dpu_set, rank, each_rank) {
      DPU_ASSERT(dpu_get_nr_dpus(rank, &(this->rank_nr_dpus[each_rank])));
      ranks[each_rank] = rank;
    }

    this->rank_offset[0] = 0;
    for (uint64_t i = 1; i < this->nr_ranks; i++) {
      this->rank_offset[i] =
          this->rank_offset[i - 1] + this->rank_nr_dpus[i - 1];
    }
  }

  {
    this->rank_attached_dpu = malloc(this->nr_dpus * sizeof(uint32_t));
    this->rank_dpu_index = malloc(this->nr_dpus * sizeof(uint32_t));
    uint64_t rank_id = 0;
    uint64_t dpu_rank_index_ = 0;
    for (uint64_t i = 0; i < this->nr_dpus; i++) {
      assert(rank_id < this->nr_ranks);
      this->rank_attached_dpu[i] = rank_id;
      this->rank_dpu_index[i] = dpu_rank_index_;

      if (dpu_rank_index_ == (this->rank_nr_dpus[rank_id] - 1)) {
        dpu_rank_index_ = 0;
        rank_id += 1;
      } else
        dpu_rank_index_ += 1;
    }
  }
}
dpu_pool_t dpu_pool;

/** @brief numer of DPUs used to partition all NNZ element of one matrix */
#define MAX_NR_DPUS_PER_RANK 64

/** @brief dpu set global profiling symbol */
uint64_t clocks_per_sec;

/**
 * @brief main of the host application.
 */
int main(int argc, char **argv) {

  uint64_t max_nr_dpu_per_rank = 64;

  struct Params p = input_params(argc, argv);

  struct dpu_set_t dpu;

  // Allocate dpu set and fill dpu poool info
  alloc_dpu_pool(&dpu_pool, NR_RANKS);

  // split batch across ranks
  uint32_t batch_size = dpu_pool.nr_ranks;

  // Initialize input data
  A = readCOOMatrix(p.fileName);
  sortCOOMatrix(A);

  // Initialize partition data
  part_info = partition_init(&dpu_pool, NR_TASKLETS);

  // Load-balance nnz across DPUs
  partition_by_nnz(A, part_info, &dpu_pool);

  // Allocate input vector
  x = (val_dt **)malloc(dpu_pool.nr_ranks * sizeof(val_dt *));
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    x[rank_index] = (val_dt *)calloc(A->ncols, sizeof(val_dt));

  // Initialize input vector with arbitrary data
  init_vector(x, batch_size, A->ncols);

  // Initialize help data
  dpu_info = (struct dpu_info_t **)malloc(dpu_pool.nr_ranks *
                                          sizeof(struct dpu_info_t *));
  dpu_arguments_t **input_args =
      (dpu_arguments_t **)malloc(dpu_pool.nr_ranks * sizeof(dpu_arguments_t *));
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    dpu_info[rank_index] = (struct dpu_info_t *)malloc(
        dpu_pool.rank_nr_dpus[rank_index] * sizeof(struct dpu_info_t));
    input_args[rank_index] = (dpu_arguments_t *)malloc(
        dpu_pool.rank_nr_dpus[rank_index] * sizeof(dpu_arguments_t));
  }

  // Max limits for parallel transfers
  uint64_t *max_rows_per_dpu = calloc(dpu_pool.nr_ranks, sizeof(uint64_t));
  uint64_t *max_nnz_per_dpu = calloc(dpu_pool.nr_ranks, sizeof(uint64_t));
  uint64_t *max_rows_per_tasklet = calloc(dpu_pool.nr_ranks, sizeof(uint64_t));

  // Timer for measurements
  Timer timer;

  unsigned int i;
  // Find padding for rows and non-zero elements needed for CPU-DPU transfers
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    DPU_FOREACH(ranks[rank_index], dpu, i) {
      uint32_t rows_per_dpu = part_info[rank_index]->row_split[i + 1] -
                              part_info[rank_index]->row_split[i];
      uint32_t prev_rows_dpu = part_info[rank_index]->row_split[i];
      printf("nr dpus %u, dpu index %u, rows per dpu %u\n",dpu_pool.rank_nr_dpus[rank_index] ,  i, rows_per_dpu);

      if (rows_per_dpu > max_rows_per_dpu[rank_index])
        max_rows_per_dpu[rank_index] = rows_per_dpu;

      // Pad data to be transfered for nnzs
      unsigned int nnz = 0, nnz_pad;
      for (uint32_t r = 0; r < rows_per_dpu; r++)
        nnz += A->rows[prev_rows_dpu + r];
      if (nnz % (8 / byte_dt) != 0)
        nnz_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
      else
        nnz_pad = nnz;

      if (nnz_pad > max_nnz_per_dpu[rank_index])
        max_nnz_per_dpu[rank_index] = nnz_pad;

      uint32_t prev_nnz_dpu = 0;
      for (unsigned int r = 0; r < prev_rows_dpu; r++)
        prev_nnz_dpu += A->rows[r];

      // Keep information per DPU
      dpu_info[rank_index][i].rows_per_dpu = rows_per_dpu;
      dpu_info[rank_index][i].prev_rows_dpu = prev_rows_dpu;
      dpu_info[rank_index][i].prev_nnz_dpu = prev_nnz_dpu;
      dpu_info[rank_index][i].nnz = nnz;
      dpu_info[rank_index][i].nnz_pad = nnz_pad;

      // Find input arguments per DPU
      input_args[rank_index][i].nrows = rows_per_dpu;
      input_args[rank_index][i].tcols = A->ncols;
      input_args[rank_index][i].tstart_row =
          dpu_info[rank_index][i].prev_rows_dpu;

      // Load-balance rows across tasklets
      partition_tsklt_by_row(part_info[rank_index], rows_per_dpu, NR_TASKLETS);

      // Find max_rows_per_tasklet
      uint32_t t;
      for (t = 0; t < NR_TASKLETS; t++) {
        input_args[rank_index][i].start_row[t] =
            part_info[rank_index]->row_split_tasklet[t];
        input_args[rank_index][i].rows_per_tasklet[t] =
            part_info[rank_index]->row_split_tasklet[t + 1] -
            part_info[rank_index]->row_split_tasklet[t];
        if (input_args[rank_index][i].rows_per_tasklet[t] >
            max_rows_per_tasklet[rank_index])
          max_rows_per_tasklet[rank_index] =
              input_args[rank_index][i].rows_per_tasklet[t];
      }

      // Find input arguments per DPU
      uint32_t prev_nnz = 0;
      for (unsigned int tasklet_id = 0; tasklet_id < NR_TASKLETS;
           tasklet_id++) {
        uint32_t cur_nnz = 0;
        for (unsigned int r = dpu_info[rank_index][i].prev_rows_dpu +
                              input_args[rank_index][i].start_row[tasklet_id];
             r < dpu_info[rank_index][i].prev_rows_dpu +
                     input_args[rank_index][i].start_row[tasklet_id] +
                     input_args[rank_index][i].rows_per_tasklet[tasklet_id];
             r++)
          if (r < dpu_info[rank_index][i].prev_rows_dpu +
                      dpu_info[rank_index][i].rows_per_dpu) {
            cur_nnz += A->rows[r];
          }
        input_args[rank_index][i].start_nnz[tasklet_id] = prev_nnz;
        input_args[rank_index][i].nnz_per_tasklet[tasklet_id] = cur_nnz;
        prev_nnz += cur_nnz;
      }
    }
  }

  // Initializations for parallel transfers with padding needed
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    if (max_rows_per_dpu[rank_index] % 2 == 1)
      max_rows_per_dpu[rank_index]++;
    if (max_nnz_per_dpu[rank_index] % (8 / byte_dt) != 0)
      max_nnz_per_dpu[rank_index] +=
          ((8 / byte_dt) - (max_nnz_per_dpu[rank_index] % (8 / byte_dt)));
    if (max_rows_per_tasklet[rank_index] % (8 / byte_dt) != 0)
      max_rows_per_tasklet[rank_index] +=
          ((8 / byte_dt) - (max_rows_per_tasklet[rank_index] % (8 / byte_dt)));
  }

  uint32_t rank_max_rows_per_tasklet = max_rows_per_tasklet[0];
  uint32_t rank_max_rows_per_dpu = max_rows_per_dpu[0];
  uint32_t rank_max_nnz_per_dpu = max_nnz_per_dpu[0];
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    if (rank_max_rows_per_tasklet < max_rows_per_tasklet[rank_index])
      rank_max_rows_per_tasklet = max_rows_per_tasklet[rank_index];
    if (rank_max_rows_per_dpu < max_rows_per_dpu[rank_index])
      rank_max_rows_per_dpu = max_rows_per_dpu[rank_index];
    if (rank_max_nnz_per_dpu < max_nnz_per_dpu[rank_index])
      rank_max_nnz_per_dpu = max_nnz_per_dpu[rank_index];
  }
  // add output datas
  uint64_t ***y_dpu_row_index;
  uint64_t ***y_dpu_cur_rows;
  uint64_t ***y_dpu_dest_index;
  y_dpu_row_index = malloc(  dpu_pool.nr_ranks * sizeof(uint64_t**));
  y_dpu_cur_rows =  malloc(  dpu_pool.nr_ranks * sizeof(uint64_t**));
  y_dpu_dest_index =  malloc(  dpu_pool.nr_ranks * sizeof(uint64_t**));
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    unsigned int n, j, t;
    y_dpu_row_index[rank_index] = malloc( dpu_pool.rank_nr_dpus[rank_index] * sizeof(uint64_t*));
    y_dpu_cur_rows[rank_index] =  malloc( dpu_pool.rank_nr_dpus[rank_index] * sizeof(uint64_t*));
    y_dpu_dest_index[rank_index] =  malloc( dpu_pool.rank_nr_dpus[rank_index] * sizeof(uint64_t*));
    for (n = 0; n < dpu_pool.rank_nr_dpus[rank_index]; n++) {
      y_dpu_row_index[rank_index][n] = malloc( NR_TASKLETS * sizeof(uint64_t));
      y_dpu_cur_rows[rank_index][n] = malloc(NR_TASKLETS * sizeof(uint64_t));
      y_dpu_dest_index[rank_index][n] = malloc(NR_TASKLETS * sizeof(uint64_t));
      //for (t = 0; t < NR_TASKLETS; t++) {
      //   y_dpu_row_index[rank_index][n][t] = malloc( rank_max_rows_per_tasklet * sizeof(uint64_t));
      //}
    }
  }

  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    unsigned int n, j, t;
    for (n = 0; n < dpu_pool.rank_nr_dpus[rank_index]; n++) {
      for (t = 0; t < NR_TASKLETS; t++) {
        unsigned int cur_rows = input_args[rank_index][n].rows_per_tasklet[t];
        if ((cur_rows + input_args[rank_index][n].start_row[t] >
             dpu_info[rank_index][n].rows_per_dpu) &&
            (cur_rows != 0))
              cur_rows = dpu_info[rank_index][n].rows_per_dpu -
                     input_args[rank_index][n].start_row[t];

        y_dpu_cur_rows[rank_index][n][t] = cur_rows;
        uint32_t row_index =
            n * NR_TASKLETS * rank_max_rows_per_tasklet +
            t * max_rows_per_tasklet[rank_index];
        y_dpu_row_index[rank_index][n][t] = row_index;
        y_dpu_dest_index[rank_index][n][t]  = i;
        i+= cur_rows;
      }
    }
  }

  // Re-allocations
  A->nnzs = (struct elem_t *)realloc(
      A->nnzs,
      (rank_max_nnz_per_dpu)*max_nr_dpu_per_rank * sizeof(struct elem_t));

  y = (val_dt **)malloc(dpu_pool.nr_ranks * sizeof(val_dt *));
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    y[rank_index] = (val_dt *)calloc(dpu_pool.nr_dpus * NR_TASKLETS *
                                         max_rows_per_tasklet[rank_index],
                                     sizeof(val_dt));
  val_dt **y_dpu = (val_dt **)malloc(dpu_pool.nr_ranks * sizeof(val_dt *));
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    y_dpu[rank_index] = (val_dt *)calloc(A->nrows, sizeof(val_dt));


  // Count total number of bytes to be transfered in MRAM of DPU
  // unsigned long int total_bytes;
  // total_bytes = ((rank_max_nnz_per_dpu) * sizeof(struct elem_t)) +
  //               (A->ncols * sizeof(val_dt)) +
  //               (rank_max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt));
  // assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");


  // Copy input arguments to DPUs
  for (u_int32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    DPU_FOREACH(ranks[rank_index], dpu, i) {
      input_args[rank_index][i].max_rows_per_tasklet =
          max_rows_per_tasklet[rank_index];
      input_args[rank_index][i].rank_max_rows_per_tasklet = rank_max_rows_per_tasklet;
      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args[rank_index] + i));
    }
  }
  DPU_ASSERT(dpu_push_xfer(*dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                           sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

  // Copy input matrix to DPUs
  startTimer(&timer, 0);

  // Copy input array
  for (u_int32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    DPU_FOREACH(ranks[rank_index], dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(
          dpu, A->nnzs + dpu_info[rank_index][i].prev_nnz_dpu));
    }
  }
  DPU_ASSERT(dpu_push_xfer(
      *dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
      rank_max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt) +
          A->ncols * sizeof(val_dt),
      rank_max_nnz_per_dpu * sizeof(struct elem_t), DPU_XFER_ASYNC));

  dpu_sync(*dpu_set);
  stopTimer(&timer, 0);

  //-- measure process CPU usage ratio
  struct timespec start_time, start_process_time, stop_time, stop_process_time;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);



  // Copy input vector  to DPUs
  startTimer(&timer, 1);
  for (u_int32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    DPU_FOREACH(ranks[rank_index], dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, x[rank_index]));
    }
  }
  DPU_ASSERT(dpu_push_xfer(
      *dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
      rank_max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt),
      A->ncols * sizeof(val_dt), DPU_XFER_ASYNC));


#ifdef SYNC_RUN
    dpu_sync(*dpu_set);
#endif
  // Run kernel on DPUs
  DPU_ASSERT(dpu_launch(*dpu_set, DPU_ASYNCHRONOUS));

#ifdef SYNC_RUN
    dpu_sync(*dpu_set);
#endif

  // Retrieve results for output vector from DPUs
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    DPU_FOREACH(ranks[rank_index], dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(
          dpu, &(y[rank_index]
                  [i * NR_TASKLETS * rank_max_rows_per_tasklet])));
    }
  }
  DPU_ASSERT(dpu_push_xfer(
      *dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
      NR_TASKLETS * rank_max_rows_per_tasklet * sizeof(val_dt),
      DPU_XFER_ASYNC));
  dpu_sync(*dpu_set);
  stopTimer(&timer, 1);

  startTimer(&timer, 2);
  // #pragma omp parallel for
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    i = 0;
    unsigned int n, j, t;
    for (n = 0; n < dpu_pool.rank_nr_dpus[rank_index]; n++) {
      // #pragma omp parallel for
      for (t = 0; t < NR_TASKLETS; t++) {
        uint32_t cur_rows = y_dpu_cur_rows[rank_index][n][t] ;
        uint64_t row_index_start = y_dpu_row_index[rank_index][n][t];
        uint64_t y_dpu_dest_index_start  = y_dpu_dest_index[rank_index][n][t];
        memcpy(&(y_dpu[rank_index][y_dpu_dest_index_start]), &(y[rank_index][row_index_start]), cur_rows * sizeof(val_dt));
      }
    }
  }
  stopTimer(&timer, 2);


  clock_gettime(CLOCK_MONOTONIC_RAW, &stop_time);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_process_time);
  double time = (float)((stop_time.tv_sec - start_time.tv_sec) * 1e9 +
                        stop_time.tv_nsec - start_time.tv_nsec) /
                (1e9);
  double process_time =
      (float)((stop_process_time.tv_sec - start_process_time.tv_sec) * 1e9 +
              stop_process_time.tv_nsec - start_process_time.tv_nsec) /
      (1e9);

  // Kernel time measurement
  uint64_t kernel_cycles;
  DPU_FOREACH(*dpu_set, dpu) {
    DPU_ASSERT(dpu_copy_from(dpu, "kernel_cycles", 0, &kernel_cycles,
                             sizeof(uint64_t)));
    DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,
                             sizeof(uint32_t)));
    break;
  }

// #define LOG 1
#if LOG
  // Display DPU Logs (default: disabled)
  i = 0;
  DPU_FOREACH(*dpu_set, dpu, i) {
    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    break;
  }
#endif

  // Print timing results
  float kernel_time_sec = ((float)kernel_cycles) / clocks_per_sec;
  float kernel_time_msec = (1e3 * (float)kernel_time_sec);

  // Print timing results
  float input_datas_byte = (float)(NR_RANKS * 64 * A->ncols * sizeof(val_dt)) ;
  float vec_per_sec =  (float)(NR_RANKS)/timer.time_sec[1];
  float nr_op = (A->nnz);

  float input_x_time_sec = (float)(timer.time_sec[1]) - (float)(kernel_time_sec);
  float input_x_bw_gbps = (1e-9 * input_datas_byte) / input_x_time_sec ;


  printf("\n");
  printf("NR RANKS: ");
  printf("%u", NR_RANKS);
  printf("\n");
  printf("Load Matrix ");
  printTimer(&timer, 0);
  printf("IO xfer + kernel ");
  printTimer(&timer, 1);
  printf("Output merge ");
  printf("\t\t\t%f\n",(float)timer.time_sec[2]/timer.time_sec[1]);
  printf("Kernel Time msec \t\t\t%f\n", kernel_time_msec);
  printf("Load Input BW GB per sec \t\t%f\n", input_x_bw_gbps);
  printf("Load Matrix BW GB per sec \t\t%f\n",
       (float)(1e-9 * max_nnz_per_dpu[0] * sizeof(struct elem_t) * dpu_pool.nr_dpus) /
           (timer.time_sec[0]));
  printf("Kernel GOp/sec  \t\t\t%f\n", (1e-9 * nr_op) / (kernel_time_sec));
  printf("dpu clocks rate  \t\t\t%u\n", clocks_per_sec);
  printf("vec per sec \t\t\t\t%f\n", vec_per_sec);

  printf("CPU CLOCK RAW time %5.2f\n"
         "CPUTIME time %5.2f\n"
         "CPU PROCESS ratio %5.2f\n",
         time, process_time, process_time / time * 100.0);


#if CHECK_CORR
  // Check output
  val_dt **y_host = (val_dt **)malloc(dpu_pool.nr_ranks * sizeof(val_dt *));
  for  (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    y_host[rank_index] = (val_dt *)calloc(A->nrows, sizeof(val_dt));

  startTimer(&timer, 4);
  spmv_host(y_host, A, x, batch_size);
  stopTimer(&timer, 4);
  bool status = true;
  for (uint64_t b = 0; b < batch_size; b++) {
    for (uint64_t n = 0; n < A->nrows; n++) {
      if (y_host[b][n] != y_dpu[b][n]) {
        status = false;
      }
    }
  }

  if (status) {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  } else {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    return -1;
  }
  // dealloc host result vector
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    free(y_host[rank_index]);
  free(y_host);
  printf("\n");
  printf("CPU ");
  printTimer(&timer, 4);
#endif
  // Deallocation
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    unsigned int n, j, t;
    for (n = 0; n < dpu_pool.rank_nr_dpus[rank_index]; n++) {
      free(y_dpu_row_index[rank_index][n]);
      free(y_dpu_cur_rows[rank_index][n] );
    }
  }
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    unsigned int n, j, t;
    free(y_dpu_row_index[rank_index]);
    free(y_dpu_cur_rows[rank_index]);
  }
  free(y_dpu_row_index);
  free(y_dpu_cur_rows);














  free(max_rows_per_dpu);
  free(max_nnz_per_dpu);
  free(max_rows_per_tasklet);

  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++) {
    free(dpu_info[rank_index]);
    free(input_args[rank_index]);
  }

  free(dpu_info);
  free(input_args);
  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    free(y_dpu[rank_index]);
  free(y_dpu);

  freeCOOMatrix(A);

  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    free(x[rank_index]);
  free(x);

  for (uint32_t rank_index = 0; rank_index < dpu_pool.nr_ranks; rank_index++)
    free(y[rank_index]);
  free(y);

  partition_free(part_info, &dpu_pool);
  free_dpu_pool(&dpu_pool);

  return 0;
}
