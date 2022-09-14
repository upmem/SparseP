import os
import sys
import pandas as pd
import numpy as np

applications = ["spmv"]


def get_data(rootdir, resultdir):
    os.chdir(rootdir)

    chart = open(resultdir + "_output.txt", "w")
    chart.write(
        "NumaMode,Matrix,Datatype,TL,DPUS,LoadMatrixTime,LoadInputTime,LoadInputBw,KernelTime,RetrieveOutputTime,ReductionTime\n"
    )

    for file in os.listdir(rootdir + "/" + resultdir):
        if (not file.endswith("out")):
            continue
        numamode = file.split("_")[0]
        matrix = file.split("_")[1]
        datatype = file.split("_")[2]
        tl = file.split("_")[3].replace("tl", "")
        dpus = file.split("_")[4].replace("dpus", "")
        clock_rate = 1  # in ms, f = 350MHz

        avg_cpu_time = [0]
        avg_load_matrix_time = [0]
        avg_load_x_time = [0]
        avg_load_x_bw = [0]
        avg_kernel_time = [0]
        avg_retrieve_time = [0]
        avg_reduction_time = [0]
        tmp = rootdir + "/" + resultdir + "/" + file
        with open(tmp, "r") as ins:

            for line in ins:
                # if(line.find("CPU Time (ms):")!=-1):
                #    value =  (float(line.split("CPU Time (ms): ")[1].split()[0]))
                #    avg_cpu_time.append(value)
                if (line.find("Load Matrix Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Matrix Time (ms): ")
                         [1].split()[0]))
                    avg_load_matrix_time.append(value)
                if (line.find("Load Matrix Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Matrix Time (ms): ")
                         [1].split()[0]))
                    avg_load_matrix_time.append(value)
                if (line.find("Load Input Vector Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Input Vector Time (ms): ")
                         [1].split()[0]))
                    avg_load_x_time.append(value)
                if(line.find("Load Matrix BW GB per sec") != -1):
                    value = 0.00001 + \
                        (float(line.split("Load Matrix BW GB per sec")
                         [1].split()[0]))
                    avg_load_x_bw.append(value)
                if (line.find("Kernel Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Kernel Time (ms): ")[1].split()[0]))
                    avg_kernel_time.append(value)
                if (line.find("Retrieve Output Vector Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split(
                            "Retrieve Output Vector Time (ms): ")[1].split()[0]))
                    avg_retrieve_time.append(value)
                if (line.find("Merge Partial Result Time (ms):") != -1):
                    value = 0.00001 + \
                        (float(line.split("Merge Partial Result Time (ms): ")
                         [1].split()[0]))
                    avg_reduction_time.append(value)
        try:
            chart.write(
                str(numamode) + "," + str(matrix) + "," + str(datatype) + "," +
                str(tl) + "," + str(dpus) + "," +
                str(clock_rate * max(avg_cpu_time)) + "," +
                str(max(avg_load_matrix_time)) + "," +
                str(max(avg_load_x_time)) + "," +
                str(max(avg_load_x_bw)) +
                 "," + str(max(avg_kernel_time)) +
                "," + str(max(avg_retrieve_time)) + "," +
                str(max(avg_reduction_time)) + "\n")
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
