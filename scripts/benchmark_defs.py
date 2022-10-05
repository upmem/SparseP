NR_RANKS = []

MAX_RANKS = 10
for i in range(MAX_RANKS):
   NR_RANKS.append(i+1)

# WARNING NR_RANKS must always 1 eg : NR_RANKS = [1]
#NR_RANKS =[1]#, 16 , 32]
NR_RANKS =[1, 16, 32]

NR_TASKLETS = [16]
DATATYPES = ["int32"]
print('test with RANKs ', NR_RANKS)

DPU_CLUSTER_SIZES = {
    "ldoor.mtx" : 8,
    "af_shell1.mtx" : 8,
    "roadNet-TX.mtx": 8,
    "parabolic_fem.mtx": 8,
    "poisson3Db.mtx": 8,
    "delaunay_n19.mtx": 8,
    "com-Youtube.mtx": 8,
    "pkustk14.mtx": 8,
    "rajat31.mtx": 8,
    "raefsky4.mtx": 8,
    "delaunay_n13.mtx": 8,
    "pkustk08.mtx": 8,
    "wing_nodal.mtx": 8
}

MATRICES = [
    "ldoor.mtx",
    "af_shell1.mtx",
    "roadNet-TX.mtx",
    "parabolic_fem.mtx",
    "poisson3Db.mtx",
    "delaunay_n19.mtx",
    "com-Youtube.mtx",
    "pkustk14.mtx",
    "rajat31.mtx",
    "raefsky4.mtx",
    "delaunay_n13.mtx",
    "pkustk08.mtx",
    "wing_nodal.mtx"
]

MATRIXDICT_NAMES = {
    0 : "ldoor",
    1 : "af-shell1",
    2 : "roadNet-TX",
    3 : "parabolic-fem",
    4 : "poisson3Db",
    5 : "delaunay-n19",
    6 : "com-Youtube",
    7 : "pkustk14",
    8 : "rajat31",
    9 : "raefsky4",
    10 : "delaunay-n13",
    11 : "pkustk08",
    12 : "wing-nodal"
}

MATRIXDICT = {
  "ldoor"         : 0,
  "af-shell1"     : 1,
  "roadNet-TX"    : 2,
  "parabolic-fem" : 3,
  "poisson3Db"    : 4,
  "delaunay-n19"  : 5,
  "com-Youtube"   : 6,
  "pkustk14"      : 7,
  "rajat31"       : 8,
  "raefsky4"      : 9,
  "delaunay-n13"  : 10,
  "pkustk08"      : 11,
  "wing-nodal"    : 12
}

NUMA_MODES_ID = ["numadefault"]#, "numaall", "numa01"]
NUMA_MODES = [""]
#  , "numactl --interleave=all ",
#  "numactl --interleave=0,1 "
#]

