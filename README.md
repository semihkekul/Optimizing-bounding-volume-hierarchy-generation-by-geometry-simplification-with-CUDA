# OPTIMIZING BOUNDING VOLUME HIERARCHY GENERATION BY GEOMETRY SIMPLIFICATION WITH CUDA
**This was my unfinished thesis project but I converted to an informal personal project!**

In computer games, handling a large number of objects with accurate collision detection is a complex task. A common solution is to divide the process into two phases:

Broad Phase: This step quickly filters out object pairs that are unlikely to collide, often using a Bounding Volume Hierarchy (BVH) to avoid checking every object against every other.
Narrow Phase: This step performs detailed collision checks only on the filtered pairs from the broad phase.
Recently, Graphics Processing Units (GPUs) have been increasingly used instead of traditional Central Processing Units (CPUs) to speed up this process.

In this thesis, a geometric simplification method is enhanced using Level of Detail (LOD) optimization during remeshing. Additionally, a parallel algorithm is used in the broad phase to efficiently build the BVH structure.

## Results
![Generated bounding volumes](./docs/bunny.png) 


## References 
1 [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees  , Tero Karras](./docs/Maximizing_Parallelism_in_the_Construction_of_BVHs_Octrees_and_kd_Trees.pdf) 
