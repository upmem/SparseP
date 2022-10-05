import os
import sys
import pandas as pd
import numpy as np

applications = ["spmv"]


def get_data(rootdir, resultdir):
    os.chdir(rootdir)

    chart = open(resultdir + "_output.txt", "w")
    chart.write(
        "NumaMode,Matrix,Datatype,Tl,NrRanks,CpuTime,LoadMatrixTime,LoadMatrixBw,LoadInputBw,TotalTime,KernelTime,KernelGOpsec,OutputMerge,CpuRatio,VecPerSec,DpuClusterSize\n"
    )

    for file in os.listdir(rootdir + "/" + resultdir):
        if (not file.endswith("out")):
            continue
        numamode = file.split("_")[0]
        matrix = file.split("_")[1]
        datatype = file.split("_")[2]
        tl = file.split("_")[3].replace("tl", "")
        nr_ranks = file.split("_")[4].replace("ranks", "")
        dpu_cluster_size = file.split("_")[5].replace("dcs", "")
        clock_rate = 1  # in ms, f = 350MHz

        avg_cpu_time = [0]
        avg_load_matrix_time = [0]
        avg_load_mat_bw = [0]
        avg_load_x_bw = [0]
        avg_total_time = [0]
        avg_kernel_time = [0]
        avg_x_gops = [0]
        avg_vec_per_sec = [0]
        avg_output_merge = [0]
        avg_cpu_process_ratio = [0]

        tmp = rootdir + "/" + resultdir + "/" + file
        with open(tmp, "r") as ins:

            for line in ins:
                if (line.find("CPU Time (ms):") != -1):
                    value = (
                        float(line.split("CPU Time (ms): ")[1].split()[0]))
                    avg_cpu_time.append(value)
                if (line.find("Load Matrix Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Matrix Time (ms): ")
                         [1].split()[0]))
                    avg_load_matrix_time.append(value)
                if (line.find("IO xfer + kernel Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("IO xfer + kernel Time (ms):")
                         [1].split()[0]))
                    avg_total_time.append(value)
                if (line.find("Kernel Time msec") != -1):
                    value = 0.00001 + \
                        (float(line.split("Kernel Time msec")
                         [1].split()[0]))
                    avg_kernel_time.append(value)
                if (line.find("Load Matrix BW GB per sec") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Matrix BW GB per sec")
                         [1].split()[0]))
                    avg_load_mat_bw.append(value)
                if (line.find("Load Input BW GB per sec") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Input BW GB per sec")
                         [1].split()[0]))
                    avg_load_x_bw.append(value)
                if (line.find("Kernel GOp/sec") != -1):
                    value = 0.00001 + \
                        (float(line.split("GOp/sec")[1].split()[0]))
                    avg_x_gops.append(value)
                if (line.find("vec per sec") != -1):
                    value = 0.00001 + \
                        (float(line.split("vec per sec")[1].split()[0]))
                    avg_vec_per_sec.append(value)
                if (line.find("Output merge") != -1):
                    value = 0.00001 + \
                        (float(line.split("Output merge")[1].split()[0]))
                    avg_output_merge.append(value)
                if (line.find("CPU PROCESS ratio") != -1):
                    value = 0.00001 + \
                        (float(line.split("CPU PROCESS ratio")[1].split()[0]))
                    avg_cpu_process_ratio.append(value)

        try:
            assert (nr_ranks != None)
            chart.write(
                str(numamode) + "," +
                str(matrix) + "," +
                str(datatype) + "," +
                str(tl) + "," +
                str(nr_ranks) + "," +
                str(max(avg_cpu_time)) + "," +
                str(max(avg_load_matrix_time)) + "," +
                str(max(avg_load_mat_bw)) + "," +
                str(max(avg_load_x_bw)) + "," +
                str(max(avg_total_time)) + "," +
                str(max(avg_kernel_time)) + "," +
                str(max(avg_x_gops)) + "," +
                str(max(avg_output_merge)) + "," +
                str(max(avg_cpu_process_ratio)) + "," +
                str(max(avg_vec_per_sec)) + "," +
                str(dpu_cluster_size) + "," +
                "\n")
        except:
            print(tmp)

    chart.close()


def main():
    if (len(sys.argv) < 3):
        print("Usage: python run.py root_dir result_dir")
        for value in applications:
            print(value)
    else:
        get_data(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
