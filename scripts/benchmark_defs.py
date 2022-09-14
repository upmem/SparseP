NR_DPUS_START = 64
NR_DPUS = [64]

MAX_DPUS = 1500
NITER = int(MAX_DPUS/128) - 1
for i in range(NITER):
   NR_DPUS.append( 128*(i+1))
NR_DPUS=[64,1024]
NR_TASKLETS = [16]
DATATYPES = ["int32"]
print('test with DPUs ', NR_DPUS)

MATRICES = [
    "ldoor.mtx",
    # "af_shell1.mtx",
    # "roadNet-TX.mtx",
    # "parabolic_fem.mtx",
    # "poisson3Db.mtx",
    # "delaunay_n19.mtx",
    # "com-Youtube.mtx",
    # "pkustk14.mtx",
    # "wing_nodal.mtx",
    # "delaunay_n13.mtx",
    # "pkustk08.mtx",
    # "raefsky4.mtx",
    # "rajat31.mtx"
]

MATRIXDICT_NAMES = {
    0: "ldoor",
    # 1: "af-shell1",
    # 2: "roadNet-TX",
    # 3: "parabolic-fem",
    # 4: "poisson3Db",
    # 5 : "delaunay-n19" ,
    # 6 : "com-Youtube" ,
    # 7 : "pkustk14",
    # 8 : "wing-nodal",
    # 9 : "delaunay-n13",
    # 10 : "pkustk08",
    # 11 : "raefsky4",
    # 12 : "rajat31"
}

MATRIXDICT = {
    "ldoor": 0,
    # "af-shell1": 1,
    # "roadNet-TX": 2,
    # "parabolic-fem": 3,
    # "poisson3Db": 4,
    # "delaunay-n19" : 5,
    # "com-Youtube" : 6,
    # "pkustk14" : 7,
    # "wing-nodal" : 8,
    # "delaunay-n13" : 9,
    # "pkustk08" : 10,
    # "raefsky4" : 11,
    # "rajat31" : 12
}

NUMA_MODES_ID = ["numadefault", "numa12"]
NUMA_MODES = [
    "",
    "UPMEM_PROFILE='nrThreadPerPool=8' numactl --interleave=0,1 "
]

#NUMA_MODES_ID = ["numadefault", "numaall", "numa12"]
#NUMA_MODES = [
#    "", "numactl --interleave=all ",
#    "UPMEM_PROFILE='nrThreadPerPool=8' numactl --interleave=0,1 "
#]