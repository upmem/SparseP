#!/bin/dash

## FIXME
input_path="/home/upmemstaff/dgerin/SparseP/inputs/"

python3 run_CSR.py ${input_path}
python3 run_COO-rgrn.py ${input_path}
python3 run_COO.py ${input_path}
python3 run_BCSR.py ${input_path}
python3 run_BCOO.py ${input_path}

