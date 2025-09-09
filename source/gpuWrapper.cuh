#pragma once
#include "cuda.cuh"
#include "GameObject.h"

#define PrintFunction std::cout <<std::endl <<"........... "<<__FUNCTION__ <<" ........... "<< std::endl<<std::endl

#define PrintLine std::cout <<std::endl <<"........... "<<__FUNCTION__ <<" : "<<__LINE__<<" ........... "<< std::endl<<std::endl

#define CUDAErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }

void GpuAssert(cudaError_t code, const char *file, int line);



static struct stSimplificationParameters
{
	uint vertexCount;
	uint faceCount;

	uint* dev_MortonCodes;

	uint* dev_MortonSortedToClMapping;
	uint* dev_indicesOfMortonCodes;

	GpuNode *dev_leafNodes;
	GpuNode *dev_internalNodes;

	Vertex* dev_totalSumVertexOFCl;
	uint* dev_vertexCountOfCl;
	uint* dev_vertexCountScan;

	Matrix* dev_QScan;
	Vertex* dev_TotalVertexSumScan;

	Vertex* dev_vertices;

	uint* dev_MortonToVertexMapping;
	uint* dev_VertexToMortonMapping;
	GPUFace* dev_gpuFaces;

	Matrix* dev_quadricsOfFaces;
	Matrix* dev_Ql;

	uint2* dev_P;
	uint* dev_IndicesOfFacesPrime;
	uint* dev_MarkedFaceIndices;

	uint* dev_PToVPrime;

	Vertex* dev_verticesPrime;
	
	GPUFace* faces;

	uint clCount;

	float2 theta;

	Vertex eyePosition;

	float distanceRange;

	void cleanUp() const
	{
		CUDAErrorCheck(cudaFree(dev_MortonCodes));
		CUDAErrorCheck(cudaFree(dev_MortonSortedToClMapping));
		CUDAErrorCheck(cudaFree(dev_indicesOfMortonCodes));
		CUDAErrorCheck(cudaFree(dev_leafNodes));
		CUDAErrorCheck(cudaFree(dev_internalNodes));
		CUDAErrorCheck(cudaFree(dev_totalSumVertexOFCl));
		CUDAErrorCheck(cudaFree(dev_vertexCountOfCl));
		CUDAErrorCheck(cudaFree(dev_vertexCountScan));
		CUDAErrorCheck(cudaFree(dev_QScan));
		CUDAErrorCheck(cudaFree(dev_TotalVertexSumScan));
		CUDAErrorCheck(cudaFree(dev_vertices));
		CUDAErrorCheck(cudaFree(dev_MortonToVertexMapping));
		CUDAErrorCheck(cudaFree(dev_VertexToMortonMapping));
		CUDAErrorCheck(cudaFree(dev_gpuFaces));
		CUDAErrorCheck(cudaFree(dev_quadricsOfFaces));
		CUDAErrorCheck(cudaFree(dev_Ql));
		CUDAErrorCheck(cudaFree(dev_P));
		CUDAErrorCheck(cudaFree(dev_IndicesOfFacesPrime));
		CUDAErrorCheck(cudaFree(dev_MarkedFaceIndices));
		//CUDAErrorCheck(cudaFree(dev_PToVPrime));
		CUDAErrorCheck(cudaFree(dev_verticesPrime));

		delete[] faces;
	}

} gSimplificationParameters;




static struct stBVHParameters
{

	uint* dev_MortonCodes;
	Vertex* dev_Vertices;

	GPUFace* dev_primitives;  // faces 
	Vertex* dev_primitiveCenters;

	uint* dev_ids;

	stAABB* dev_aabbs;

	GpuNode *dev_leafNodes;
	GpuNode *dev_internalNodes;

	uint primitiveCount;
	uint vertexCount;

} gBVHParameters;

struct stParameters
{
	std::string objFileName;
	float2 theta;
	Vertex eyePosition;
	float distanceRange;
};


template <typename T>
void printArray(T* arr, int count, std::string name)
{
	int i = 0;
	std::cout << std::endl << "=============== " << name.c_str() << " ============================" << std::endl;
	while (i < count)
	{
		std::cout << arr[i] << "\n";
		++i;
	}
	std::cout << std::endl << "======================================================" << std::endl;
}






void initCUDA();



void cudaMain(std::vector<Vertex>&	vertices,
	stScene&							scene,
	std::vector<Face>&				faces,
	stParameters&					params
);


uint getGameObjectTriangles(GameObject* gameObject, std::vector<stAABB>& aabbs);