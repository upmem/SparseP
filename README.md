## SparseP Software Package v1.0

<p align="center">
  <img width="600" height="338" src="https://github.com/CMU-SAFARI/SparseP/blob/main/images/SparseP-logo.png">
</p>

[<i>SparseP</i>](https://arxiv.org/pdf/2201.05072.pdf) software package is a collection of efficient Sparse Matrix Vector Multiplication (SpMV) kernels for real-world Processing-In-Memory (PIM) architectures. <i>SparseP</i> is written in C programming language and provides 25 SpMV kernels. <i>SparseP</i> can be useful to developers of solver libraries and of scientific applications to attain high performance and energy efficiency of the SpMV kernel on real-world PIM architectures, and to software, architecture and system researchers to improve multiple aspects of future PIM hardware and software.

<i>SparseP</i> is developed to evaluate, analyze, and characterize the first publicly-available real-world PIM architecture, the [UPMEM](https://www.upmem.com/) PIM architecture. The UPMEM PIM architecture is a near-bank PIM architecture that supports several PIM-enabled memory chips connected to a host CPU via memory channels. Each memory chip comprises multiple general-purpose in-order cores, called DRAM Processing Units (DPUs), each of them is tightly coupled with a DRAM bank. For a thorough characterization analysis of the UPMEM PIM architecture, please see our [CUT'21](https://people.inf.ethz.ch/omutlu/pub/Benchmarking-Memory-Centric-Computing-Systems_cut21.pdf) and [arXiv'21](https://arxiv.org/pdf/2105.03814.pdf) papers, and check out the [PrIM benchmark suite](https://github.com/CMU-SAFARI/prim-benchmarks), a collection of diverse workloads to characterize real-world PIM architectures.

<i>SparseP</i> efficiently maps the SpMV execution kernel on near-bank PIM systems and supports:
* a wide range of data types: 
	* 8-bit integer (INT8)
	* 16-bit integer (INT16)
	* 32-bit integer (INT32)
	* 64-bit integer (INT64)
	* 32-bit float (FP32)
	* 64-bit float data types (FP64)
* two types of well-crafted data partitioning techniques: 
	* the 1D-partitioned kernels (1D), where the matrix is horizontally partitioned across PIM cores, and the whole input vector is copied into the DRAM bank of each PIM core.
	* the 2D-partitioned kernels (2D), where the matrix is split in 2D tiles, the number of which is equal to the number of PIM cores, and a subset of the elements of the input vector is copied into the DRAM bank of each PIM core.
* the most popular compressed matrix storage formats:
	* Compressed Sparse Row (CSR)
	* Coordinate Format (COO)
	* Block Compressed Sparse Row (BCSR)
	* Block Coordinate Format (BCOO)
* various load balancing schemes across PIM cores:
	* load-balance either the rows (rows) or the non-zero elements (nnzs) for the CSR and COO formats
	* load-balance either the blocks (blocks) or the non-zero elements (nnzs) for the BCSR and BCOO formats
* several load balancing schemes across threads within a multithreaded PIM core
	* load-balance either the rows (rows) or the non-zero elements (nnzs) for the CSR and COO formats
	* load-balance either the blocks (blocks) or the non-zero elements (nnzs) for the BCSR and BCOO formats
* three synchronization approaches among parallel threads within a PIM core
	* coarse-grained locking (lb-cg)
	* fine-grained locking (lb-fg)
	* lock-free (lf)

## Cite <i>SparseP</i>

Please cite the following papers if you find this repository useful:

Christina Giannoula, Ivan Fernandez, Juan Gómez-Luna, Nectarios Koziris, Georgios Goumas, and Onur Mutlu, "[SparseP: Towards Efficient Sparse Matrix Vector Multiplication on Real Processing-In-Memory Systems](https://arxiv.org/pdf/2201.05072.pdf)", arXiv:2201.05072 [cs.AR], 2022.


Bibtex entries for citation:
```
@article{Giannoula2022SparsePPomacs,
	author={Christina Giannoula and Ivan Fernandez and Juan Gómez-Luna and Nectarios Koziris and Georgios Goumas and Onur Mutlu},
	title={SparseP: Towards Efficient Sparse Matrix Vector Multiplication on Real Processing-In-Memory Architectures}, 
	year = {2022},
	publisher = {Association for Computing Machinery},
	volume = {6},
	number = {1},
	url = {https://doi.org/10.1145/3508041},
	doi = {10.1145/3508041},
	journal = {Proc. ACM Meas. Anal. Comput. Syst.},
	articleno = {21},
}
```

```
@inproceedings{Giannoula2022SparsePSigmetrics,
	author={Christina Giannoula and Ivan Fernandez and Juan Gómez-Luna and Nectarios Koziris and Georgios Goumas and Onur Mutlu},
	title={Towards Efficient Sparse Matrix Vector Multiplication on Real Processing-In-Memory Architectures}, 
	year = {2022},
        isbn = {9781450391412},
	publisher = {Association for Computing Machinery},
	url = {https://doi.org/10.1145/3489048.3522661},
	doi = {10.1145/3489048.3522661},
        booktitle = {Abstract Proceedings of the 2022 ACM SIGMETRICS/IFIP PERFORMANCE Joint International Conference on Measurement and Modeling of Computer Systems},
        pages = {33–34},
        numpages = {2},
        location = {Mumbai, India},
        series = {SIGMETRICS/PERFORMANCE '22}
}
```


```
@misc{Giannoula2022SparseParXiv
      title={SparseP: Towards Efficient Sparse Matrix Vector Multiplication on Real Processing-In-Memory Systems}, 
      author={Christina Giannoula and Ivan Fernandez and Juan Gómez-Luna and Nectarios Koziris and Georgios Goumas and Onur Mutlu},
      year={2022},
      eprint={2201.05072},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```

## Repository Structure
We point out next the repository structure and some important folders and files.<br> 
The "spmv" directory includes all the 1D- and 2D-partitioned SpMV kernels of <i>SparseP</i> software package, i.e., 9 1D-partitioned and 16 2D-partitioned SpMV kernels.<br> 
The "inputs" directory includes a bash script to download matrix files (in mtx format) from the [Suite Sparse Matrix Collection](https://sparse.tamu.edu/).<br> 
The "scripts" directory includes python3 scripts to run experiments for the 1D- and 2D-partitioned SparseP kernels.<br>

```
.
+-- LICENSE
+-- README.md
+-- images/ 
+-- inputs/ 
+-- scripts/ 
|   +-- run_1DPU/
|   +-- run_1D/
|   +-- run_2D/
+-- spmv/ 
|   +-- 1D/
|	|	+-- CSR-row/
|	|	+-- CSR-nnz/
|	|	+-- COO-row/
|	|	+-- COO-nnz-rgrn/
|	|	+-- COO-nnz/
|	|	+-- BCSR-block/
|	|	+-- BCSR-nnz/
|	|	+-- BCOO-block/
|	|	+-- BCOO-nnz/
|   +-- 2D/
|	|	+-- DCSR/
|	|	+-- DCOO/
|	|	+-- DBCSR/
|	|	+-- DBCOO/
|	|	+-- RBDCSR/
|	|	+-- RBDCOO/
|	|	+-- RBDBCSR-block/
|	|	+-- RBDBCSR-nnz/
|	|	+-- RBDBCOO-block/
|	|	+-- RBDBCOO-nnz/
|	|	+-- BDCSR/
|	|	+-- BDCOO/
|	|	+-- BDBCSR-block/
|	|	+-- BDBCSR-nnz/
|	|	+-- BDBCOO-block/
|	|	+-- BDBCOO-nnz/
```

The following table summarizes the SpMV PIM kernels provided by the SparseP software package. 

<p align="center">
  <img width="800" height="600" src="https://github.com/CMU-SAFARI/SparseP/blob/main/images/SparseP-kernels.png">
</p>

## Requirements 

Running <i>SparseP</i> requires installing the [UPMEM SDK](https://sdk.upmem.com). The <i>SparseP</i> SpMV kernels are designed to run on a server with real UPMEM modules, but they also run on the functional simulator included in the UPMEM SDK.

## Running <i> SparseP </i>

### Clone the Git Repository

```sh
git clone https://github.com/CMU-SAFARI/SparseP.git

cd SparseP
```

### Download Input Matrix Files

```sh
cd inputs

./download_matrices.sh 
```


### Run the SpMV on one DPU (multiple tasklets)

```sh
cd scripts/run_1DPU

## FIXME input_path="/path/to/matrices"
## To use this script, update the path to the input matrix files
./run_1DPU.sh 
```


### Run the 1D-Partitioned SpMV Kernels

```sh
cd scripts/run_1D

## FIXME input_path="/path/to/matrices"
## To use this script, update the path to the input matrix files
./run_1D.sh 
```


### Run the 2D-Partitioned SpMV Kernels

```sh
cd scripts/run_2D

## FIXME input_path="/path/to/matrices"
## To use this script, update the path to the input matrix files
./run_2D.sh 
```




### Run an SpMV Kernel
Inside each SpMV kernel, one can compile and run each SpMV kernel with different configurations. Every Makefile accepts several input parameters:

```sh
cd spmv/1D/CSR-row

# Compile the CSR-row kernel for 32 DPUs, 16 tasklets (i.e., software threads) per DPU and the 32-bit integer data type
NR_DPUS=32 NR_TASKLETS=16 TYPE=INT32 make all
```

For help instructions:

```sh
# Input parameters
./bin/spmv_host -h
```

Run the SpMV kernel:

```sh
# Run the SpMV kernel
./bin/spmv_host -f /path/to/input/matrix/file.mtx
```



## Support
For any suggestions for improvement, any issues related to the <i>SparseP</i> SpMV kernels or for reporting bugs, please contact Christina Giannoula at christina.giann\<at\>gmail.com.
