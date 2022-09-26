/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 *
 * Partitioning and balancing across DPUs and tasklets
 */

#ifndef _PARTITION_H_
#define _PARTITION_H_

/**
 * @brief Specific information for data partitioning/balancing
 */
#include <stdlib.h>
struct partition_info_t {
    uint32_t *row_split;
    uint32_t *row_split_tasklet;
};

typedef struct dpu_pool {
  uint32_t nr_dpus;
  uint32_t nr_ranks;
  uint32_t *rank_nr_dpus;
  uint32_t *rank_offset;
  uint32_t *rank_attached_dpu;
  uint32_t *rank_dpu_index;
} dpu_pool_t;


/**
 * @brief allocate data structure for partitioning
 */
struct partition_info_t **partition_init(dpu_pool_t * dpu_pool, uint32_t nr_of_tasklets) {
    struct partition_info_t **part_info;
    part_info = (struct partition_info_t **) malloc( dpu_pool->nr_ranks *  sizeof(struct partition_info_t*));

    for (uint32_t rank_index = 0; rank_index < dpu_pool->nr_ranks;
         rank_index++) {
	    part_info[rank_index] = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));
	    part_info[rank_index]->row_split = (uint32_t *) malloc((dpu_pool->rank_nr_dpus[rank_index] + 1) * sizeof(uint32_t));
	    part_info[rank_index]->row_split_tasklet = (uint32_t *) malloc((nr_of_tasklets + 2) * sizeof(uint32_t));
    }

    return part_info;
}

/**
 * @brief load-balance nnz in row granularity across DPUs
 */
void partition_by_nnz(struct COOMatrix *cooMtx, struct partition_info_t **part_info, dpu_pool_t *dpu_pool) {

    if (dpu_pool->nr_dpus == 1) {
        part_info[0]->row_split[0] = 0;
        part_info[0]->row_split[1] = cooMtx->nrows;
        return;
    }

    for (uint32_t rank_index = 0; rank_index < dpu_pool->nr_ranks;
         rank_index++) {

        // Compute the matrix splits.
        uint32_t nnz_cnt = cooMtx->nnz;
        uint32_t nnz_per_split = nnz_cnt / dpu_pool->rank_nr_dpus[rank_index];
        uint32_t curr_nnz = 0;
        uint32_t row_start = 0;
        uint32_t split_cnt = 0;
        uint32_t i;
      printf("nnz per split %u\n",  nnz_per_split);

        part_info[rank_index]->row_split[0] = row_start;
        for (i = 0; i < cooMtx->nrows; i++) {
            curr_nnz += cooMtx->rows[i];
            if (curr_nnz >= nnz_per_split) {
                row_start = i + 1;
                ++split_cnt;
                if (split_cnt <= dpu_pool->rank_nr_dpus[rank_index])
                    part_info[rank_index]->row_split[split_cnt] = row_start;
                curr_nnz = 0;
            }
        }
        printf("end split count %u\n", split_cnt);

        // Fill the last split with remaining elements
        if (curr_nnz < nnz_per_split && split_cnt <= dpu_pool->rank_nr_dpus[rank_index]) {
            part_info[rank_index]->row_split[split_cnt] = cooMtx->nrows;
        }

        // If there are any remaining rows merge them in last partition
        if (split_cnt > dpu_pool->rank_nr_dpus[rank_index]) {
            part_info[rank_index]->row_split[dpu_pool->rank_nr_dpus[rank_index]] = cooMtx->nrows;
        }

        // If there are remaining threads create empty partitions
        for (i = split_cnt + 1; i <= dpu_pool->rank_nr_dpus[rank_index]; i++) {
            part_info[rank_index]->row_split[i] = cooMtx->nrows;
        }
    }
}


/**
 * @brief load-balance rows across tasklets
 */
void partition_tsklt_by_row(struct partition_info_t *part_info, int rows_per_dpu, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t chunks = rows_per_dpu / nr_of_tasklets;
    uint32_t rest_rows = rows_per_dpu % nr_of_tasklets;
    uint32_t rows_per_tasklet;
    uint32_t curr_row = 0;

    part_info->row_split_tasklet[0] = curr_row;
    for(unsigned int i=0; i < nr_of_tasklets; i++) {
        rows_per_tasklet = chunks;
        if (i < rest_rows)
            rows_per_tasklet += 1;
        curr_row += rows_per_tasklet;
        if (curr_row > rows_per_dpu)
            curr_row = rows_per_dpu;
        part_info->row_split_tasklet[i+1] = curr_row;
    }

}



/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t **part_info, dpu_pool_t *dpu_pool) {
    for (uint32_t rank_index = 0; rank_index < dpu_pool->nr_ranks;
         rank_index++) {
       free(part_info[rank_index]->row_split);
       free(part_info[rank_index]->row_split_tasklet);
    }
    free(part_info);
}


#endif
