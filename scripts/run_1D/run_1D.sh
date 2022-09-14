#!/bin/dash

## FIXME
input_path="/home/upmemstaff/dgerin/SparseP/inputs/"

python3 run_COO-nnz.py ${input_path}
python3 ~/SparseP/plot/plot1D.py ~/SparseP/spmv/1D/COO-nnz/ results
python3 ~/SparseP/plot/plot_exec_time_reduced.py ~/SparseP/spmv/1D/COO-nnz/results_output.txt int32

#python3 run_CSR-nnz.py ${input_path}
#python3 run_CSR-row.py ${input_path}
#python3 run_COO-row.py ${input_path}
#python3 run_COO-nnz-rgrn.py ${input_path}
#python3 run_COO-nnz.py ${input_path}
#python3 run_BCSR-block.py ${input_path}
#python3 run_BCSR-nnz.py ${input_path}
#python3 run_BCOO-block.py ${input_path}
#python3 run_BCOO-nnz.py ${input_path}

