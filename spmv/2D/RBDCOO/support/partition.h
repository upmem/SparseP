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
struct partition_info_t {
    uint32_t *row_split;
    uint32_t *nnz_split;
#if FG_TRANS
    uint32_t *max_rows_per_rank; // max rows among 64 DPUs per rank
    uint32_t *active_dpus_per_rank; // active DPUs per rank (needed when there are faulty DPUs in the system)
    uint32_t *accum_dpus_ranks; // accumulated active DPUs
#endif
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(struct RBDCOOMatrix *rbdcooMtx, uint32_t nr_of_dpus, uint32_t max_ranks, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->row_split = (uint32_t *) malloc(2 * rbdcooMtx->vert_partitions * (rbdcooMtx->horz_partitions + 2) * sizeof(uint32_t));
    part_info->nnz_split = (uint32_t *) malloc(rbdcooMtx->vert_partitions * (rbdcooMtx->horz_partitions + 2) * sizeof(uint32_t));

#if FG_TRANS
    part_info->max_rows_per_rank = (uint32_t *) calloc(max_ranks, sizeof(uint32_t));
    part_info->active_dpus_per_rank = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
    part_info->accum_dpus_ranks = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
#endif
    return part_info;
}

/** 
 * @brief load-balance nnz across DPUs
 */
void partition_by_nnz(struct RBDCOOMatrix *rbdcooMtx, struct partition_info_t *part_info) {

    uint32_t c, i;
    uint64_t nnz_offset = 0;
    for(c = 0; c < rbdcooMtx->vert_partitions; c++) {
        // Compute the matrix splits.
        uint32_t rptr_offset = 2 * c * rbdcooMtx->horz_partitions;
        uint32_t sptr_offset = c * (rbdcooMtx->horz_partitions + 1);
        uint32_t nnz_cnt = rbdcooMtx->nnzs_per_vert_partition[c];
        uint32_t nnz_per_split = nnz_cnt / rbdcooMtx->horz_partitions;
        uint32_t curr_nnz = 0;
        uint32_t row_start = 0;
        uint32_t nnz_start = 0;
        uint32_t split_cnt = 0;

        part_info->row_split[rptr_offset] = row_start;
        part_info->nnz_split[sptr_offset] = nnz_start;
        for (i = 0; i < nnz_cnt; i++) {
            curr_nnz++; 
            if (curr_nnz >= nnz_per_split) {
                row_start = rbdcooMtx->nnzs[nnz_offset + i].rowind + 1;
                ++split_cnt;
                if (split_cnt <= rbdcooMtx->horz_partitions) {
                    part_info->row_split[rptr_offset + 2 * (split_cnt - 1) + 1] = row_start; 
                    part_info->nnz_split[sptr_offset + split_cnt] = part_info->nnz_split[sptr_offset + split_cnt - 1] + curr_nnz;
                    if ((nnz_offset + i) == (rbdcooMtx->nnz - 1)) {
                        part_info->row_split[rptr_offset + 2 * split_cnt] = row_start;
                    } else if ((rbdcooMtx->nnzs[nnz_offset + i].rowind == rbdcooMtx->nnzs[nnz_offset + i + 1].rowind)) {
                        part_info->row_split[rptr_offset + 2 * split_cnt] = row_start - 1; 
                    } else {
                        part_info->row_split[rptr_offset + 2 * split_cnt] = row_start;
                    }
                }
                curr_nnz = 0;
            }
        }

        // Fill the last split with remaining elements
        if (curr_nnz < nnz_per_split && split_cnt <= rbdcooMtx->horz_partitions) {
            split_cnt++;
            part_info->row_split[rptr_offset + 2 * (split_cnt - 1) + 1] = rbdcooMtx->nrows;
            part_info->row_split[rptr_offset + 2 * split_cnt] = rbdcooMtx->nrows;
            part_info->nnz_split[sptr_offset + split_cnt] = rbdcooMtx->nnzs_per_vert_partition[c];
        }

        // If there are any remaining rows merge them in last partition
        if (split_cnt > rbdcooMtx->horz_partitions) {
            part_info->row_split[rptr_offset + 2 * (rbdcooMtx->horz_partitions - 1) + 1] = rbdcooMtx->nrows;
            part_info->row_split[rptr_offset + 2 * rbdcooMtx->horz_partitions] = rbdcooMtx->nrows;
            part_info->nnz_split[sptr_offset + rbdcooMtx->horz_partitions] = rbdcooMtx->nnzs_per_vert_partition[c];
        }

        // If there are remaining threads create empty partitions
        for (i = split_cnt + 1; i <= rbdcooMtx->horz_partitions; i++) {
            part_info->row_split[rptr_offset + 2 * (i - 1) + 1] = rbdcooMtx->nrows;
            part_info->row_split[rptr_offset + 2 * i] = rbdcooMtx->nrows;
            part_info->nnz_split[sptr_offset + i] = rbdcooMtx->nnzs_per_vert_partition[c];
        }

        nnz_offset += nnz_cnt;
    }


    // Sanity Check
    for (c = 0; c < rbdcooMtx->vert_partitions; c++) {
        uint32_t local_nnzs = 0;
        for (i = 0; i < rbdcooMtx->horz_partitions; i++) {
            local_nnzs += (part_info->nnz_split[c * (rbdcooMtx->horz_partitions + 1) + i+1] - part_info->nnz_split[c * (rbdcooMtx->horz_partitions + 1) + i]);
        }
        assert(rbdcooMtx->nnzs_per_vert_partition[c] == local_nnzs && "Wrong partitioning!");
        assert(rbdcooMtx->nnzs_per_vert_partition[c] == part_info->nnz_split[c * (rbdcooMtx->horz_partitions + 1) + rbdcooMtx->horz_partitions] && "Wrong partitioning!");
    }

}


/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->row_split);
    free(part_info->nnz_split);
#if FG_TRANS
    free(part_info->max_rows_per_rank); 
    free(part_info->active_dpus_per_rank); 
    free(part_info->accum_dpus_ranks);
#endif
    free(part_info);
}


#endif
