#pragma once
#define M_E        2.71828182845904523536
#define M_LOG2E    1.44269504088896340736
#define M_LOG10E   0.434294481903251827651
#define M_LN2      0.693147180559945309417
#define M_LN10     2.30258509299404568402
#define M_PI       3.14159265358979323846f
#define M_PI_2     1.57079632679489661923f
#define M_PI_4     0.785398163397448309616
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2    1.41421356237309504880
#define M_SQRT1_2  0.707106781186547524401


#include <cstdlib>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <host_defines.h>
#include <vector_functions.hpp>
#include <vector>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <pplinterface.h>
#include <winscard.h>
typedef unsigned int uint;

struct  uSeq3
{
	uint first, second, third;
};



typedef float3 Vertex;
typedef float3 Normal;

typedef float4 Vertex4;

struct Matrix
{
	Vertex4 row1;
	Vertex4 row2;
	Vertex4 row3;
	Vertex4 row4;
};

struct Matrix3x3
{
	Vertex row1;
	Vertex row2;
	Vertex row3;
};


struct Matrix2x2
{
	float2 row1;
	float2 row2;
};


struct stAABB
{
	Vertex posMin;
	Vertex posMax;
};

struct GpuNode
{
	int	 objectId;
	GpuNode* parent;
	GpuNode* left;
	GpuNode* right;
	stAABB AABB;
	Vertex xn; // representative point
	float En;	// approximation error
	int flag;
	bool isLeaf;
};


struct stScene
{
	Vertex min;
	Vertex max;
};




struct GPUFace
{
	uSeq3 vertexIndices;
};

struct Face
{
	uint vertexIndices[3];
};

struct Attribute
{
	std::vector<uint>	indices;
	std::vector<Vertex>	vertices;
	std::vector<Vertex>	normals;
};

struct DistanceMax : public thrust::binary_function<uint,uint,uint>
{
	__host__ __device__
		uint operator()(const uint first, const uint second)
	{
		return first + second - 1;
	}
};

struct MatrixSum : public thrust::binary_function<Matrix, Matrix, Matrix>
{

	__host__ __device__
		Matrix operator()(const Matrix& M1, const Matrix& M2)
	{
		Matrix result;
		result.row1.x = M1.row1.x + M2.row1.x;
		result.row1.y = M1.row1.y + M2.row1.y;
		result.row1.z = M1.row1.z + M2.row1.z;
		result.row1.w = M1.row1.w + M2.row1.w;

		result.row2.x = M1.row2.x + M2.row2.x;
		result.row2.y = M1.row2.y + M2.row2.y;
		result.row2.z = M1.row2.z + M2.row2.z;
		result.row2.w = M1.row2.w + M2.row2.w;

		result.row3.x = M1.row3.x + M2.row3.x;
		result.row3.y = M1.row3.y + M2.row3.y;
		result.row3.z = M1.row3.z + M2.row3.z;
		result.row3.w = M1.row3.w + M2.row3.w;

		result.row4.x = M1.row4.x + M2.row4.x;
		result.row4.y = M1.row4.y + M2.row4.y;
		result.row4.z = M1.row4.z + M2.row4.z;
		result.row4.w = M1.row4.w + M2.row4.w;

		return result;
	}
};

struct VertexSum : public thrust::binary_function<Vertex, Vertex, Vertex>
{

	__host__ __device__
		Vertex operator()(const Vertex& U, const Vertex& V)
	{
		Vertex result;


		result.x = U.x + V.x;
		result.y = U.y + V.y;
		result.z = U.z + V.z;

		/*
		printf("result\n%f %f %f \n %f %f %f\n %f %f %f\n\n",
		result.row1.x,
		result.row1.y,
		result.row1.z,
		result.row2.x,
		result.row2.y,
		result.row2.z,
		result.row3.x,
		result.row3.y,
		result.row3.z);
		*/

		return result;
	}
};


float RandomFloat(float min, float max);




// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__global__ void mortonKernel(const Vertex *dev_vertices, const uint noOfVertices, uint* dev_mortonCodes, stScene scene);

__global__ void CalculateFaceQuadricKernel(Vertex* dev_vertices,
	Matrix* dev_Quadrics,
	GPUFace* dev_GpuFaces,
	uint faceCount
);

__global__ void VertexToMortonMapGenerateKernel(uint* dev_MortonToVertexMap, uint* dev_VertexToMortonMapping, uint vertexCount);

//@brief the first occurence of each morton code is flagged
//@param dev_sortedMortonCodes; input; sorted morton codes
//@param vertexCount; input; 
//@param dev_MortonSortedToClMapping; output; sorted morton code index  -> Cl index ; 
// nth mortonCode is mapped to dev_MortonSortedToClMapping[n]
//__global__ void MarkFirstOccurencesKernel(uint* dev_sortedMortonCodes, int vertexCount, uint* dev_MortonSortedToClMapping);

__global__ void MarkFirstOccurencesKernel(uint* dev_data, uint dataCount, uint* dev_output);

__global__ void SumLeafQuadricKernel(Matrix* dev_Ql, //Ql
	Vertex* dev_totalSumVertexOFCl,
	uint* dev_vertexCountOfCl,
	Vertex* dev_vertices,
	uint* dev_MortonSortedToClMapping,
	uint* dev_VertexToMortonMapping,
	Matrix* dev_QuadricsOfFaces,  //Qt
	GPUFace* dev_GpuFaces,
	const uint faceCount);

__global__  void generateKdTreeKernel(Matrix* dev_QScan,
	Vertex* dev_TotalVertexSumScan,
	uint* dev_vertexCountScan,
	uint* dev_Cl,
	uint clCount,
	GpuNode*     leafNodes,
	GpuNode* internalNodes);

__global__ void treeTraverselKernel(uint2* dev_P, uint clCount, GpuNode* dev_InternalNodes, GpuNode* dev_leafNodes,float2 theta, Vertex eyePosition, float distanceRange);

__global__ void markTrianglesKernel(GPUFace* dev_GpuFaces, uint faceCount,
	uint2* dev_P,
	uint* dev_MarkedFaceIndices,
	uint* dev_MortonSortedToClMapping,
	uint* dev_VertexToMortonMapping,
	uint* dev_MortonToVertexMapping,
	uint* PToVPrime);


__global__ void PToVPrimeKernel(uint clCount, uint2* P, uint* PToVPrime);

__global__ void doesExistsKernel(uint2* dev_data, uint dataCount, uint* dev_output);
__global__ void doesExistsKernel(uint* dev_data, uint dataCount, uint* dev_output);

__global__ void calculateFaceCenterKernel(const GPUFace* dev_primitives, Vertex* dev_primitiveCenters, uint primitiveCount, Vertex* dev_remeshVertices);

__global__ void setIncreasingIdKernel(uint* data, uint count);

__global__ void generateAABBKernel(GpuNode* dev_leafNodeBuffer, Vertex* vertices, GPUFace* primitives, uint primitiveCount);

__global__  void generateHierarchyKernel(uint* sortedMortonCodes, GpuNode*     leafNodes, GpuNode* internalNodes, uint primitiveCount);

__global__ void boundingBoxCalculationKernel(GpuNode* leafNodes, uint primitiveCount);


__global__ void VtoVPrimeKernel(Vertex* VPrime, GpuNode* dev_leafNodes,GpuNode* dev_internalNodes, uint clCount, uint* MortonToV, uint2* P, uint* PToVPrime);

__global__ void remeshKernel(GPUFace* Tprime, GPUFace* T, uint* primitiveIndices, uint Tcount, uint* VToMorton, uint* MortonToCl, uint2* P, uint* PToVPrime);

__global__ void dumpPaths(GpuNode* I, uint count);

__global__ void CalculateRepresentativePointsOfLeavesKernel(Matrix* dev_Ql,
	Vertex* dev_totalSumVertexOFCl,
	uint* dev_vertexCountOfCl,
	GpuNode* dev_leafNodes,
	uint clCount);