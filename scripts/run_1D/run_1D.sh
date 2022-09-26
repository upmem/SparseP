#!/bin/dash
# cc -o bin/spmv_host host/app.c -g -Isupport  -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DINT32 -DBLNC_TSKLT_NNZ=1 -DNR_TASKLETS=16 -DNR_DPUS=64
# dpu-profiling functions -o dpu.json -a -A   --   ./bin/spmv_host  -f /home/upmemstaff/dgerin/SparseP/inputs/rajat31.mtx
# dgerin@upmemcloud7:~/SparseP/spmv/1D/COO-nnz$ cc -o bin/spmv_host host/app.c -g -Isupport  -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DINT32 -DNR_TASKLETS=16 -DNR_DPUS=1024
# dgerin@upmemcloud7:~/SparseP/spmv/1D/COO-nnz$ bin/spmv_host  -f /home/upmemstaff/dgerin/SparseP/inputs/rajat31.mtx
## FIXME
input_path="~/SparseP/inputs/"
# cd ~/SparseP/spmv/1D/COO-nnz-rgrn/
# cc -o bin/spmv_host host/app.c -g -Isupport -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DINT32 -DNR_TASKLETS=16 -DNR_DPUS=64 -DBLNC_TSKLT_ROW=1 -DBLNC_TSKLT_NNZ=0
# bin/spmv_host  -f /home/upmemstaff/dgerin/SparseP/inputs/rajat31.mtx
python3 run_COO-nnz-rgrn.py ${input_path}
python3 run_legacy_COO-nnz-rgrn.py ${input_path}
echo '====reimpl====='
python3 ~/SparseP/plot/plot1D.py ~/SparseP/spmv/1D/COO-nnz-rgrn/ results
python3 ~/SparseP/plot/plot_exec_time_reduced.py ~/SparseP/spmv/1D/COO-nnz-rgrn/results_output.txt int32 > reimpl.res
echo '====legacy====='
python3 ~/SparseP/plot/plot1D.py ~/SparseP/spmv/1D/legacy-COO-nnz-rgrn/ results
python3 ~/SparseP/plot/plot_exec_time_reduced.py ~/SparseP/spmv/1D/legacy-COO-nnz-rgrn/results_output.txt int32 > legacy.res

#python3 run_CSR-nnz.py ${input_path}
#python3 ~/SparseP/plot/plot1D.py ~/SparseP/spmv/1D/COO-nnz/ results
#python3 ~/SparseP/plot/plot_exec_time_reduced.py ~/SparseP/spmv/1D/COO-nnz/results_output.txt int32

#python3 run_CSR-row.py ${input_path}
#python3 run_COO-row.py ${input_path}
#python3 run_COO-nnz-rgrn.py ${input_path}
#python3 run_COO-nnz.py ${input_path}
#python3 run_BCSR-block.py ${input_path}
#python3 run_BCSR-nnz.py ${input_path}
#python3 run_BCOO-block.py ${input_path}
#python3 run_BCOO-nnz.py ${input_path}

