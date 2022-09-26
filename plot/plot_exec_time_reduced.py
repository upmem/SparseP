import os
import sys
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
import benchmark_defs
from benchmark_defs import *

dir_path = os.getcwd()
rootdir = dir_path.replace("scripts/camera_ready/1D/balance/transfers_coo", "")

NMAT = len(MATRICES)

matrix_names = [""] * NMAT
matrix_nnzs = [0] * NMAT
cpu_time = {}
load_matrix = {}
load_x_bw = {}
load_matrix_bw = {}
kernel = {}
gops = {}
total = {}

kernel_norm = {}
total_norm = {}

kernel_100 = {}
total_100 = {}

kernel_final = {}
total_final = {}
output_merge = {}
cpu_ratio = {}
vec_per_sec = {}


resultStruct = {
    'numadefault': [0] * NMAT,
    'numaall': [0] * NMAT,
    'numa01': [0] * NMAT
}

for nr_ranks in NR_RANKS:
    cpu_time[nr_ranks] = copy.deepcopy(resultStruct)
    load_matrix[nr_ranks] = copy.deepcopy(resultStruct)
    load_x_bw[nr_ranks] = copy.deepcopy(resultStruct)
    load_matrix_bw[nr_ranks] = copy.deepcopy(resultStruct)
    kernel[nr_ranks] = copy.deepcopy(resultStruct)
    gops[nr_ranks] = copy.deepcopy(resultStruct)
    total[nr_ranks] = copy.deepcopy(resultStruct)

    kernel_norm[nr_ranks] = copy.deepcopy(resultStruct)
    total_norm[nr_ranks] = copy.deepcopy(resultStruct)

    kernel_100[nr_ranks] = copy.deepcopy(resultStruct)
    total_100[nr_ranks] = copy.deepcopy(resultStruct)

    kernel_final[nr_ranks] = copy.deepcopy(resultStruct)
    total_final[nr_ranks] = copy.deepcopy(resultStruct)
    output_merge[nr_ranks] = copy.deepcopy(resultStruct)
    cpu_ratio[nr_ranks] = copy.deepcopy(resultStruct)
    vec_per_sec[nr_ranks] = copy.deepcopy(resultStruct)


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
            ranks = int(split_line[4])
            cpu_time_ = split_line[5]
            load_matrix_time_ = split_line[6]
            load_matrix_bw_ = split_line[7]
            load_x_bw_ = split_line[8]
            total_time_ = split_line[9]
            kernel_time_ = split_line[10]
            gops_ = split_line[11]
            output_merge_ = split_line[12]
            cpu_ratio_ = split_line[13]
            vec_per_sec_ = split_line[14]

            if ((datatype.find(dt_in) != -1) and (mtx in MATRIXDICT.keys())):
                cpu_time[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(cpu_time_)
                load_matrix[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_matrix_time_)
                load_x_bw[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_x_bw_)
                load_matrix_bw[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(load_matrix_bw_)
                kernel[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(kernel_time_)
                gops[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(gops_)
                total[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(total_time_)
                output_merge[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(output_merge_)
                cpu_ratio[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(cpu_ratio_)
                vec_per_sec[ranks][numamod][int(
                    MATRIXDICT.get(mtx))] = float(vec_per_sec_)
                # matrix_names
                matrix_names[int(MATRIXDICT.get(mtx))] = str(
                    MATRIXDICT_NAMES[int(MATRIXDICT.get(mtx))])


def main():
    path = sys.argv[1]
    dt = sys.argv[2]

    run_coo(path, dt)

    for numamod in NUMA_MODES_ID:
        total_norm[1][numamod] = [1] * NMAT

    for numamod in NUMA_MODES_ID:
        for ranks in NR_RANKS:
            if (not ranks > 1):
                continue
            total_norm[ranks][numamod] = [
                j / i for i, j in zip(total[ranks][numamod], total[1][numamod])
            ]

    for numamod in NUMA_MODES_ID:
        for ranks in NR_RANKS:
            kernel_100[ranks][numamod] = [
                i / j
                for i, j in zip(kernel[ranks][numamod], total[ranks][numamod])
            ]
            total_100[ranks][numamod] = [
                i / j
                for i, j in zip(total[ranks][numamod], total[ranks][numamod])
            ]

    res = {}
    for ranks in NR_RANKS:
        res[ranks] = [
            load_x_bw[ranks],
            load_matrix[ranks],
            load_matrix_bw[ranks],
            gops[ranks],
            total[ranks],
            kernel_100[ranks],
            cpu_time[ranks],
            output_merge[ranks],
            cpu_ratio[ranks],
            vec_per_sec[ranks],
        ]

    # CSV dump
    print('')
    print('[summary report]')
    for k, v in res.items():
        print('RANKS,', end='')
        print('')
        print('{},'.format(k), end='')
        print('')
        print('stage,', end='')
        [print(i + ',', end='') for i in matrix_names]
        for numamod in NUMA_MODES_ID:
            print('')
            print('input_vector/sec({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[9][numamod]]

        print('')
    print('')
    print('[detailed report]')
    for k, v in res.items():
        print('RANKS,', end='')
        print('')
        print('{},'.format(k), end='')
        print('')
        print('stage,', end='')
        [print(i + ',', end='') for i in matrix_names]
        for numamod in NUMA_MODES_ID:
            print('')
            print('load_x_bw [GB/sec]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[0][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('load_matrix [msec]]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[1][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('load_matrix_bw [GB/sec]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[2][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('Kernel GOps/sec({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[3][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('total[msec]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[4][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('kernel [%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[5][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('cpu time [msec]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[6][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('output merge [%total]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[7][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('cpu ratio [%]({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[8][numamod]]
        for numamod in NUMA_MODES_ID:
            print('')
            print('input_vector/sec({}),'.format(numamod), end='')
            [print(str(i) + ',', end='') for i in v[9][numamod]]

        print('')


if __name__ == "__main__":
    main()
