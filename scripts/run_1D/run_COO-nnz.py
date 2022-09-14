import os 
import sys
import glob
import getpass

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import benchmark_defs
from  benchmark_defs import *

result_path = "results/"

def run(input_path):

    pwd = os.getcwd()
    path = pwd.replace("scripts/run_1D", "") + "spmv/1D/COO-nnz"
    os.chdir(path)
    os.makedirs(result_path, exist_ok=True)
    for file in MATRICES:
        print(file)
        for NUMA_MODE in zip(NUMA_MODES_ID, NUMA_MODES):
            run_cmd = NUMA_MODE[1] + "bin/spmv_host " + " -f " + input_path + file
            for dt in DATATYPES:
                for r in NR_DPUS:
                    for t in NR_TASKLETS:
                        # Coarse-grained
                        #os.system("make clean")
                        #make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=1 FGLOCK=0 LOCKFREE=0"
                        #os.system(make_cmd)
                        temp_file = str(file[:-4]) 
                        temp_file = temp_file.replace("_", "-") 
                        #r_cmd = run_cmd + " >> " + result_path + temp_file 
                        #r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_cg.out" 
                        #os.system(r_cmd)


                        ## Fine-grained
                        #os.system("make clean")
                        #make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=0 FGLOCK=1 LOCKFREE=0"
                        #os.system(make_cmd)
                        #r_cmd = run_cmd + " >> " + result_path + temp_file
                        #r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_fg.out" 
                        #os.system(r_cmd)       


                        # Lock-free
                        os.system("make clean")
                        make_cmd = "make NR_DPUS=" + str(r) + " NR_TASKLETS=" + str(t) + " TYPE=" + dt.upper() + " CGLOCK=0 FGLOCK=0 LOCKFREE=1"
                        os.system(make_cmd)
                        r_cmd = run_cmd + " >> "  + result_path + NUMA_MODE[0] + "_" + temp_file
                        r_cmd = r_cmd + "_" + dt + "_tl" + str(t) + "_dpus" + str(r) +  "_lf.out" 
                        print(r_cmd)
                        os.system(r_cmd)       



def main():
    input_path = sys.argv[1]
    run(input_path)

if __name__ == "__main__":
    main()
