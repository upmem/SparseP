import os
import sys
import glob
import getpass
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import benchmark_defs
from benchmark_defs import *

result_path = "results/"


def run(input_path):

    pwd = os.getcwd()
    path = pwd.replace("scripts/run_1D", "") + "spmv/1D/COO-nnz-rgrn"
    os.chdir(path)
    #os.rmdir(result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=False)
    for file in MATRICES:
        dcs = DPU_CLUSTER_SIZES[file]
        for NUMA_MODE in zip(NUMA_MODES_ID, NUMA_MODES):
            run_cmd = NUMA_MODE[1] + "bin/spmv_host " + \
                " -f " + input_path + file
            for dt in DATATYPES:
                for r in NR_RANKS:
                    for t in NR_TASKLETS:
                        # Balance Rows across tasklets
                        os.system("make clean")
                        make_cmd = "make NR_RANKS=" + str(r) + " DPU_CLUSTER_SIZE=" + str(dcs) + " NR_TASKLETS=" + str(
                            t) + " TYPE=" + dt.upper() + " BLNC_TSKLT_ROW=1 BLNC_TSKLT_NNZ=0"
                        print(make_cmd)
                        os.system(make_cmd)
                        temp_file = str(file[:-4])
                        temp_file = temp_file.replace("_", "-")
                        r_cmd = run_cmd + " >> " + result_path + \
                            NUMA_MODE[0] + "_" + temp_file
                        r_cmd = r_cmd + "_" + dt + "_tl" + \
                            str(t) + "_ranks" + str(r) + "_dcs" + str(dcs) + "_row.out"
                        ret = os.system(r_cmd)
                        print(ret)
                        if ret != 0:
                          print('FUNCTIONAL FAIL FOR {}'.format(r_cmd))
                          exit()

                        # Balance NNZs across tasklets
                        #os.system("make clean")
                        #make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " BLNC_TSKLT_ROW=0 BLNC_TSKLT_NNZ=1"
                        # os.system(make_cmd)
                        #r_cmd = run_cmd + " >> " + result_path + temp_file
                        #r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_nnz.out"
                        # os.system(r_cmd)


def main():
    input_path = sys.argv[1]
    run(input_path)


if __name__ == "__main__":
    main()
