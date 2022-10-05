/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 *
 * Partitioning and balancing across DPUs and tasklets
 */

#ifndef _PARTITION_H_
#define _PARTITION_H_

#define MAX_NR_DPUS_PER_RANK 64
#define MEGABYTE(_x) (_x * 1000000)

#define MRAM_SIZE_BYTE MEGABYTE(64)

/**
 * @brief Specific information for data partitioning/balancing
 */
#include <stdint.h>
#include <stdlib.h>
#include <dpu.h>
struct partition_info_t {
    uint32_t *row_split;
    uint32_t *row_split_tasklet;
};


typedef struct dpu_clusters_t {
  uint32_t nr_clusters;
  uint32_t cluster_size;
  // struct dpu_set_t ***cluster_dpus;
  uint32_t *cluster_nr_dpus;

} dpu_clusters_t;

typedef struct dpu_pool {
  uint32_t nr_dpus;
  uint32_t nr_ranks;
  uint32_t *rank_nr_dpus;
  uint32_t *rank_offset;
  uint32_t *rank_attached_dpu;
  uint32_t *rank_dpu_index;
  dpu_clusters_t dpu_clusters;
} dpu_pool_t;


/**
 * @brief allocate data structure for partitioning
 */
struct partition_info_t **partition_init(dpu_pool_t * dpu_pool, dpu_clusters_t *dpu_clusters, uint32_t nr_of_tasklets) {
    struct partition_info_t **part_info;
    part_info = (struct partition_info_t **) malloc( dpu_clusters->nr_clusters *  sizeof(struct partition_info_t*));

    for (uint32_t cluster_index = 0; cluster_index < dpu_clusters->nr_clusters;
         cluster_index++) {
	    part_info[cluster_index] = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));
	    part_info[cluster_index]->row_split = (uint32_t *) malloc((dpu_clusters->cluster_nr_dpus[cluster_index] + 1) * sizeof(uint32_t));
	    part_info[cluster_index]->row_split_tasklet = (uint32_t *) malloc((nr_of_tasklets + 2) * sizeof(uint32_t));
    }

    return part_info;
}

void free_dpu_clusters(dpu_clusters_t *this) {
  free (this->cluster_nr_dpus);
}

void alloc_dpu_clusters(dpu_clusters_t *dpu_clusters, dpu_pool_t *dpu_pool, struct dpu_set_t *dpu_set , uint32_t dpu_cluster_size) {

  assert("nr_dpus must be multiple of dpu_cluster_size" && (0 == (dpu_pool->nr_dpus % dpu_cluster_size)));
  uint32_t nr_clusters = dpu_pool->nr_dpus / dpu_cluster_size;

  dpu_clusters->cluster_nr_dpus = (uint32_t *) malloc(nr_clusters * sizeof(uint32_t));
  dpu_clusters->nr_clusters = nr_clusters;
  dpu_clusters->cluster_size = dpu_cluster_size;
  for (uint32_t cluster_index = 0; cluster_index < nr_clusters;
       cluster_index++) {
       dpu_clusters->cluster_nr_dpus[cluster_index] = dpu_cluster_size;
  }
}

/**
 * @brief load-balance nnz in row granularity across DPUs
 */
void partition_by_nnz(struct COOMatrix *cooMtx, struct partition_info_t **part_info, dpu_pool_t *dpu_pool, dpu_clusters_t *dpu_clusters) {

    if (dpu_pool->nr_dpus == 1) {
        part_info[0]->row_split[0] = 0;
        part_info[0]->row_split[1] = cooMtx->nrows;
        return;
    }

    uint32_t nr_clusters = dpu_clusters->nr_clusters;

    for (uint32_t cluster_index = 0; cluster_index < nr_clusters;
         cluster_index++) {

        uint32_t cluster_nr_dpus = dpu_clusters->cluster_nr_dpus[cluster_index];

        printf("\n");
        printf("cluster index %u\n", cluster_index);
        printf("cluster nr dpus %u\n", cluster_nr_dpus);


        // Compute the matrix splits.
        uint32_t nnz_cnt = cooMtx->nnz;
        uint32_t nnz_per_split = nnz_cnt / cluster_nr_dpus;
        uint32_t curr_nnz = 0;
        uint32_t row_start = 0;
        uint32_t split_cnt = 0;
        uint32_t i;

        printf("nnz per split %u\n",  nnz_per_split);

        part_info[cluster_index]->row_split[0] = row_start;
        for (i = 0; i < cooMtx->nrows; i++) {
            curr_nnz += cooMtx->rows[i];
            if (curr_nnz >= nnz_per_split) {
                row_start = i + 1;
                ++split_cnt;
                if (split_cnt <= cluster_nr_dpus)
                    part_info[cluster_index]->row_split[split_cnt] = row_start;
                curr_nnz = 0;
            }
        }
        printf("end split count %u\n", split_cnt);

        // Fill the last split with remaining elements
        if (curr_nnz < nnz_per_split && split_cnt <= cluster_nr_dpus) {
            part_info[cluster_index]->row_split[++split_cnt] = cooMtx->nrows;
        }
        printf("split cnt %u , rows nr dpu %u \n", split_cnt, part_info[cluster_index]->row_split[split_cnt]);

        // If there are any remaining rows merge them in last partition
        if (split_cnt > cluster_nr_dpus) {
            part_info[cluster_index]->row_split[cluster_nr_dpus] = cooMtx->nrows;
        }

        // If there are remaining threads create empty partitions
        for (i = split_cnt + 1; i <= cluster_nr_dpus; i++) {
            part_info[cluster_index]->row_split[i] = cooMtx->nrows;
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
void partition_free(struct partition_info_t **part_info, dpu_clusters_t *dpu_clusters, dpu_pool_t *dpu_pool) {
    for (uint32_t cluster_index = 0; cluster_index < dpu_clusters->nr_clusters ;
         cluster_index++) {
       free(part_info[cluster_index]->row_split);
       free(part_info[cluster_index]->row_split_tasklet);
    }
    free(part_info);
}


#endif
