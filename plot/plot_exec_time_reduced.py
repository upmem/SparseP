import os
import sys
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from benchmark_defs import *
import benchmark_defs

dir_path = os.getcwd()
rootdir = dir_path.replace("scripts/camera_ready/1D/balance/transfers_coo", "")

NMAT = len(MATRICES)

matrix_names = [""] * NMAT
matrix_nnzs = [0] * NMAT
load_matrix = {}
load_x = {}
load_x_bw = {}
kernel = {}
retrieve = {}
reduce = {}
total = {}

load_matrix_norm = {}
load_x_norm = {}
kernel_norm = {}
retreive_norm = {}
reduce_norm = {}
total_norm = {}

load_matrix_100 = {}
load_x_100 = {}
kernel_100 = {}
retrieve_100 = {}
reduce_100 = {}
total_100 = {}

load_matrix_final = {}
load_x_final = {}
kernel_final = {}
retrieve_final = {}
reduce_final = {}
total_final = {}

resultStruct = {
    'numadefault': [0] * NMAT,
    'numaall': [0] * NMAT,
    'numa12': [0] * NMAT
}

for n in NR_DPUS:
    load_matrix[n] = copy.deepcopy(resultStruct)
    load_x[n] = copy.deepcopy(resultStruct)
    kernel[n] = copy.deepcopy(resultStruct)
    retrieve[n] = copy.deepcopy(resultStruct)
    reduce[n] = copy.deepcopy(resultStruct)
    total[n] = copy.deepcopy(resultStruct)

    load_matrix_norm[n] = copy.deepcopy(resultStruct)
    load_x_norm[n] = copy.deepcopy(resultStruct)
    kernel_norm[n] = copy.deepcopy(resultStruct)
    retreive_norm[n] = copy.deepcopy(resultStruct)
    reduce_norm[n] = copy.deepcopy(resultStruct)
    total_norm[n] = copy.deepcopy(resultStruct)

    load_matrix_100[n] = copy.deepcopy(resultStruct)
    load_x_100[n] = copy.deepcopy(resultStruct)
    kernel_100[n] = copy.deepcopy(resultStruct)
    retrieve_100[n] = copy.deepcopy(resultStruct)
    reduce_100[n] = copy.deepcopy(resultStruct)
    total_100[n] = copy.deepcopy(resultStruct)

    load_matrix_final[n] = copy.deepcopy(resultStruct)
    load_x_final[n] = copy.deepcopy(resultStruct)
    load_x_bw[n] = copy.deepcopy(resultStruct)
    kernel_final[n] = copy.deepcopy(resultStruct)
    retrieve_final[n] = copy.deepcopy(resultStruct)
    reduce_final[n] = copy.deepcopy(resultStruct)
    total_final[n] = copy.deepcopy(resultStruct)


def run_coo(path, dt_in):
    i = 0
    file_path = os.path.join(rootdir, path)
    with open(file_path, "r") as a_file:
        for line in a_file:
            if (i == 0):
                i += 1
                continue

            stripped_line = line.strip()
            split_line = stripped_line.split(",")

            numamod = split_line[0]
            mtx = split_line[1]
            datatype = split_line[2]
            tl = split_line[3]
            dpu = int(split_line[4])
            cpu_time = split_line[5]
            load_matrix_time = split_line[6]
            load_x_time = split_line[7]
            load_x_bw_ = split_line[8]
            kernel_time = split_line[9]
            retrieve_time = split_line[10]
            reduce_time = split_line[11]

            total_time = float(load_x_time) + float(kernel_time) + \
                float(retrieve_time) + float(reduce_time)
            # float(load_matrix_time)

            if ((datatype.find(dt_in) != -1) and (mtx in MATRIXDICT.keys())):
                load_matrix[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_matrix_time)
                load_x[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_x_time)
                load_x_bw[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_x_bw_)
                kernel[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(kernel_time)
                retrieve[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(retrieve_time)
                reduce[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(reduce_time)
                total[dpu][numamod][int(
                    MATRIXDICT.get(mtx))] = float(total_time)
                # matrix_names
                matrix_names[int(MATRIXDICT.get(mtx))] = str(
                    MATRIXDICT_NAMES[int(MATRIXDICT.get(mtx))])


def main():
    path = sys.argv[1]
    dt = sys.argv[2]

    run_coo(path, dt)

    for numamod in NUMA_MODES_ID:
        total_norm[64][numamod] = [1] * NMAT
    for numamod in NUMA_MODES_ID:
        for dpu in NR_DPUS:
            if (not dpu > 64):
                continue
            total_norm[dpu][numamod] = [
                j / i for i, j in zip(total[64][numamod], total[dpu][numamod])
            ]

    for numamod in NUMA_MODES_ID:
        for dpu in NR_DPUS:
            load_x_100[dpu][numamod] = [
                i / j
                for i, j in zip(load_x[dpu][numamod], total[dpu][numamod])
            ]
            kernel_100[dpu][numamod] = [
                i / j
                for i, j in zip(kernel[dpu][numamod], total[dpu][numamod])
            ]
            retrieve_100[dpu][numamod] = [
                i / j
                for i, j in zip(retrieve[dpu][numamod], total[dpu][numamod])
            ]
            reduce_100[dpu][numamod] = [
                i / j
                for i, j in zip(reduce[dpu][numamod], total[dpu][numamod])
            ]

            load_x_final[dpu][numamod] = [
                i * j for i, j in zip(load_x_100[dpu][numamod], total_norm[dpu]
                                      [numamod])
            ]
            kernel_final[dpu][numamod] = [
                i * j for i, j in zip(kernel_100[dpu][numamod], total_norm[dpu]
                                      [numamod])
            ]
            retrieve_final[dpu][numamod] = [
                i * j for i, j in zip(retrieve_100[dpu][numamod],
                                      total_norm[dpu][numamod])
            ]
            reduce_final[dpu][numamod] = [
                i * j for i, j in zip(reduce_100[dpu][numamod], total_norm[dpu]
                                      [numamod])
            ]
            #print('dpu', dpu, ':', reduce_100[dpu][numamod][0])
            #print('dpu', dpu, ':', total_norm[dpu][numamod][0])
            #print('dd', reduce_final[dpu][numamod])

    res = {}
    for dpu in NR_DPUS:
        res[dpu] = [
            load_x_100[dpu], load_x_bw[dpu], kernel_100[dpu], retrieve_100[dpu],
            reduce_100[dpu], total[dpu]
        ]

    # CSV dump
    for k, v in res.items():
        print('NPUS,', end='')
        print('')
        print('{},'.format(k), end='')
        print('')
        print('stage,', end='')
        [print(i + ',', end='') for i in matrix_names]
        for numamod in NUMA_MODES_ID:
            print('')
            print('load_input[%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[0][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('load_input_bw[GB/sec]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[1][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('kernel[%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[2][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('retrieve[%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[3][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('reduce[%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[4][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('total[ms]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[5][numamod]]
        print('')


if __name__ == "__main__":
    main()
