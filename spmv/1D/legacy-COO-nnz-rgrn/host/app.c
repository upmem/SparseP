/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>
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
static struct COOMatrix* A;
static val_dt* x;
static val_dt* y;
static struct partition_info_t *part_info;


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
struct dpu_info_t *dpu_info;


/**
 * @brief initialize input vector
 * @param pointer to input vector and vector size
 */
void init_vector(val_dt* vec, uint32_t size) {
    for(unsigned int i = 0; i < size; ++i) {
        vec[i] = (val_dt) (i%4+1);
    }
}

/**
 * @brief compute output in the host CPU
 */
static void spmv_host(val_dt* y, struct COOMatrix *A, val_dt* x) {
    #pragma omp parallel for
    for(unsigned int n = 0; n < A->nnz; n++) {
        y[A->nnzs[n].rowind] += x[A->nnzs[n].colind] * A->nnzs[n].val;
    }
}
/** @brief dpu set global profiling symbol */
uint64_t clocks_per_sec;

/**
 * @brief main of the host application.
 */
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;



    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc_ranks(NR_RANKS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("[INFO] Allocated %d DPU(s)\n", nr_of_dpus);
    printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);
#ifdef SYNC_RUN
    printf("[INFO] sync Run : YES\n");
#else
    printf("[INFO] sync Run : NO\n");
#endif

    unsigned int i;

    // Initialize input data
    A = readCOOMatrix(p.fileName);
    sortCOOMatrix(A);

    // Initialize partition data
    part_info = partition_init(nr_of_dpus, NR_TASKLETS);

    // Load-balance nnz across DPUs
    partition_by_nnz(A, part_info, nr_of_dpus);

    // Allocate input vector
    x = (val_dt *) malloc(A->ncols * sizeof(val_dt));

    // Initialize input vector with arbitrary data
    init_vector(x, A->ncols);

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_rows_per_dpu = 0;
    uint64_t max_nnz_per_dpu = 0;
    uint64_t max_rows_per_tasklet = 0;

    // Timer for measurements
    Timer timer;

    i = 0;
    // Find padding for rows and non-zero elements needed for CPU-DPU transfers
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t rows_per_dpu = part_info->row_split[i+1] - part_info->row_split[i];
        uint32_t prev_rows_dpu = part_info->row_split[i];

        if (rows_per_dpu > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu;

        // Pad data to be transfered for nnzs
        unsigned int nnz=0, nnz_pad;
        for (uint32_t r = 0; r < rows_per_dpu; r++)
            nnz += A->rows[prev_rows_dpu + r];
        if (nnz % (8 / byte_dt) != 0)
            nnz_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
        else
            nnz_pad = nnz;

        if (nnz_pad > max_nnz_per_dpu)
            max_nnz_per_dpu = nnz_pad;

        uint32_t prev_nnz_dpu = 0;
        for(unsigned int r = 0; r < prev_rows_dpu; r++)
            prev_nnz_dpu += A->rows[r];

        // Keep information per DPU
        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].prev_nnz_dpu = prev_nnz_dpu;
        dpu_info[i].nnz = nnz;
        dpu_info[i].nnz_pad = nnz_pad;

        // Find input arguments per DPU
        input_args[i].nrows = rows_per_dpu;
        input_args[i].tcols = A->ncols;
        input_args[i].tstart_row = dpu_info[i].prev_rows_dpu;

#if BLNC_TSKLT_ROW
        // Load-balance rows across tasklets
        partition_tsklt_by_row(part_info, rows_per_dpu, NR_TASKLETS);
#else
        // Load-balance nnz across tasklets
        partition_tsklt_by_nnz(A, part_info, rows_per_dpu, nnz, prev_rows_dpu, NR_TASKLETS);
#endif

        // Find max_rows_per_tasklet
        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            input_args[i].start_row[t] = part_info->row_split_tasklet[t];
            input_args[i].rows_per_tasklet[t] = part_info->row_split_tasklet[t+1] - part_info->row_split_tasklet[t];
            if (input_args[i].rows_per_tasklet[t] > max_rows_per_tasklet)
                max_rows_per_tasklet = input_args[i].rows_per_tasklet[t];

        }

        // Find input arguments per DPU
        uint32_t prev_nnz = 0;
        for(unsigned int tasklet_id=0; tasklet_id < NR_TASKLETS; tasklet_id++) {
            uint32_t cur_nnz = 0;
            for(unsigned int r = dpu_info[i].prev_rows_dpu + input_args[i].start_row[tasklet_id]; r < dpu_info[i].prev_rows_dpu + input_args[i].start_row[tasklet_id] + input_args[i].rows_per_tasklet[tasklet_id]; r++)
                if(r < dpu_info[i].prev_rows_dpu + dpu_info[i].rows_per_dpu) {
                    cur_nnz += A->rows[r];
                }
            input_args[i].start_nnz[tasklet_id] = prev_nnz;
            input_args[i].nnz_per_tasklet[tasklet_id] = cur_nnz;
            prev_nnz += cur_nnz;
        }
    }

    // Initializations for parallel transfers with padding needed
    if (max_rows_per_dpu % 2 == 1)
        max_rows_per_dpu++;
    if (max_nnz_per_dpu % (8 / byte_dt) != 0)
        max_nnz_per_dpu += ((8 / byte_dt) - (max_nnz_per_dpu % (8 / byte_dt)));
    if (max_rows_per_tasklet % (8 / byte_dt) != 0)
        max_rows_per_tasklet += ((8 / byte_dt) - (max_rows_per_tasklet % (8 / byte_dt)));

    // Re-allocations
    A->nnzs = (struct elem_t *) realloc(A->nnzs, (max_nnz_per_dpu) * nr_of_dpus * sizeof(struct elem_t));
    y = (val_dt *) calloc((uint64_t) ((uint64_t) nr_of_dpus) * (uint64_t) NR_TASKLETS * ((uint64_t) max_rows_per_tasklet), sizeof(val_dt));

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_nnz_per_dpu) * sizeof(struct elem_t)) + (A->ncols * sizeof(val_dt)) + (max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt));
    assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");


    // Copy input arguments to DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        input_args[i].max_rows_per_tasklet = max_rows_per_tasklet;
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input matrix to DPUs
    startTimer(&timer, 0);
    // Copy input array
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->nnzs + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt) + A->ncols * sizeof(val_dt), max_nnz_per_dpu * sizeof(struct elem_t), DPU_XFER_ASYNC));
    dpu_sync(dpu_set);
    stopTimer(&timer, 0);

  //-- measure process CPU usage ratio
  struct timespec start_time, start_process_time, stop_time, stop_process_time;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);

    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, x));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_tasklet * NR_TASKLETS * sizeof(val_dt), A->ncols * sizeof(val_dt), DPU_XFER_ASYNC));

#ifdef SYNC_RUN
    dpu_sync(dpu_set);
#endif

    // Run kernel on DPUs
    DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));

#ifdef SYNC_RUN
    dpu_sync(dpu_set);
#endif

#if LOG
    // Display DPU Logs (default: disabled)
    DPU_FOREACH(dpu_set, dpu) {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

    // Retrieve results for output vector from DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * NR_TASKLETS * max_rows_per_tasklet)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, NR_TASKLETS * max_rows_per_tasklet * sizeof(val_dt), DPU_XFER_ASYNC));
    dpu_sync(dpu_set);
    stopTimer(&timer, 1);

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
  DPU_FOREACH(dpu_set, dpu) {
    DPU_ASSERT(dpu_copy_from(dpu, "kernel_cycles", 0, &kernel_cycles,
                             sizeof(uint64_t)));
    DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,
                             sizeof(uint32_t)));
    break;
  }

  val_dt *y_dpu = (val_dt *) calloc((A->nrows), sizeof(val_dt));

  // merge output results (remove padding)
  startTimer(&timer, 2);
  i = 0;
  unsigned int n,j,t;
  for (n = 0; n < nr_of_dpus; n++) {
      for (t = 0; t < NR_TASKLETS; t++) {
          unsigned int cur_rows = input_args[n].rows_per_tasklet[t];
          if ((cur_rows + input_args[n].start_row[t] > dpu_info[n].rows_per_dpu) && (cur_rows != 0))
              cur_rows = dpu_info[n].rows_per_dpu - input_args[n].start_row[t];
          for (j = 0; j < cur_rows; j++) {
                y_dpu[i] = y[n * NR_TASKLETS * max_rows_per_tasklet + t * max_rows_per_tasklet + j];
              i++;
          }
      }
  }
  stopTimer(&timer, 2);

  float kernel_time_sec = ((float)kernel_cycles) / clocks_per_sec;
  float kernel_time_msec = (1e3 * (float)kernel_time_sec);

  // Print timing results
  float input_datas_byte = (float)(NR_RANKS * 64 * A->ncols * sizeof(val_dt)) ;
  float vec_per_sec =  (float)(1)/timer.time_sec[1];
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
       (float)(1e-9 * max_nnz_per_dpu * sizeof(struct elem_t) * nr_of_dpus  ) /
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
  val_dt *y_host = (val_dt *) calloc((A->nrows), sizeof(val_dt));
  startTimer(&timer, 4);
  spmv_host(y_host, A, x);
  stopTimer(&timer, 4);
  bool status = true;
  for (i = 0; i< A->nrows; i++) {
    if (y_host[i] != y[i])
        status = false;
  }
  if (status) {
      printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  } else {
      printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
      return -1;
  }

  free(y_host);

  printf("\n");
  printf("CPU ");
  printTimer(&timer, 4);
#endif

  // Deallocation
  freeCOOMatrix(A);
  free(x);
  free(y);
  partition_free(part_info);
  DPU_ASSERT(dpu_free(dpu_set));

  return 0;
}
