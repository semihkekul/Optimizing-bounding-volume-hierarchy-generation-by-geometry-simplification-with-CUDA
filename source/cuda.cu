
#include "cuda.cuh"
#define PRINT_IDX 67348

extern __device__ int __clzll(long long int 	x);

__device__ void printMatrix(const Matrix& mtx)
{
	printf("%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n\n",
		mtx.row1.x,
		mtx.row1.y,
		mtx.row1.z,
		mtx.row1.w,
		mtx.row2.x,
		mtx.row2.y,
		mtx.row2.z,
		mtx.row2.w,
		mtx.row3.x,
		mtx.row3.y,
		mtx.row3.z,
		mtx.row3.w,
		mtx.row4.x,
		mtx.row4.y,
		mtx.row4.z,
		mtx.row4.w);
}

#define SIGN(v) (0 < (v)) - ((v) < 0)
#define CONCAT(index) ((unsigned long long)sortedMortonCodes[(index)]) << 32 |(unsigned long long)(index) 

#ifdef MY_WAY
__device__  int delta(int i, int j, unsigned int *sortedMortonCodes, int numObjects)
{
	if (i < 0 || i >= numObjects || j < 0 || j >= numObjects) 
	{
		return 0;
	}
	// in case of duplicates
	long long mortonCodeI = CONCAT(i);
	long long mortonCodeJ = CONCAT(j);

	//printf("delta i %d j %d  %ld %ld\n", i,j,mortonCodeI, mortonCodeJ);

	return __clzll(mortonCodeI ^ mortonCodeJ);
}
#endif

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__  unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

__device__ void normalize(Vertex& V)
{
	float d = sqrtf(powf(V.x, 2) + powf(V.y, 2) + powf(V.z, 2));
	V.x /= d;
	V.y /= d;
	V.z /= d;
}

// vector . vector
__device__  float Dot(const Vertex4& v1, const Vertex4& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

// vector . vector
__device__  float Dot(const Vertex& v1, const Vertex& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// vector X vector
__device__  Vertex Cross(const Vertex& u, const Vertex& v)
{
	return make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

// vector + vector
__device__  Vertex Sum(const Vertex& u, const Vertex& v)
{
	return make_float3(u.x + v.x, u.y + v.y, u.z + v.z);
}

// vector * scalar
__device__  Vertex Product(const Vertex& u, float n)
{
	return make_float3(u.x * n, u.y * n, u.z *n);
}

__device__  Vertex4 Vertex_x_Matrix(Vertex4& v, Matrix& mtx)
{
	return make_float4(Dot(v, mtx.row1), Dot(v, mtx.row2), Dot(v, mtx.row3), Dot(v, mtx.row4));
}

__device__ float Distance(const Vertex& u, const Vertex& v)
{
	Vertex k = make_float3(u.x - v.x, u.y - v.y, u.z - v.z);

	return sqrtf(powf(k.x, 2) + powf(k.y, 2) + powf(k.z, 2));
}

__global__ void setIncreasingIdKernel(uint* data, uint count )
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= count)
	{
		return;
	}

	data[idx] = idx;
}




// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__global__ void mortonKernel(const Vertex *dev_vertices, const uint noOfVertices, uint* dev_mortonCodes, stScene scene)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= noOfVertices)
	{
		return;
	}
	//if (idx == 61030 || idx == 29558)
	//	printf("%d %.9g %.9g %.9g\n", idx,dev_vertices[idx].x, dev_vertices[idx].y, dev_vertices[idx].z);
	//  A good way to assign the Morton code for a given object is to use the centroid point of its bounding box, 
	//  and express it relative to the bounding box of the scene (https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/)
	Vertex morton;
	morton.x = (dev_vertices[idx].x - scene.min.x) / (scene.max.x - scene.min.x);
	morton.y = (dev_vertices[idx].y - scene.min.y) / (scene.max.y - scene.min.y);
	morton.z = (dev_vertices[idx].z - scene.min.z) / (scene.max.z - scene.min.z);


	float x = fminf(fmaxf(morton.x * 1024.0f, 0.0f), 1023.0f);
	float y = fminf(fmaxf(morton.y * 1024.0f, 0.0f), 1023.0f);
	float z = fminf(fmaxf(morton.z * 1024.0f, 0.0f), 1023.0f);
	uint xx = expandBits(static_cast<uint>(x));
	uint yy = expandBits(static_cast<uint>(y));
	uint zz = expandBits(static_cast<uint>(z));
	dev_mortonCodes[idx] = xx * 4 + yy * 2 + zz;

}

__device__ float det3(float a, float b, float c, float d, float e, float f, float g, float h, float i)
{
	return ( (a)*(e)*(i) + (b)*(f)*(g) + (c)*(d)*(h) - (c)*(e)*(g) - (b)*(d)*(i) - (a)*(f)*(h) );
}

__device__ float det4(float a, float b, float c, float d, float e, float f, float g, float h, float  i, float j, float k, float l, float  m, float n, float o, float p)
{
	////printf("%f, %f, %f, %f ; %f, %f, %f, %f ; %f, %f, %f, %f ; %f, %f, %f, %f\n", a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
	float _1 = (a)* det3((f), (g), (h), (j), (k), (l), (n), (o), (p));
	float _2 = -(b)* det3((e), (g), (h), (i), (k), (l), (m), (o), (p));
	float _3 = (c)* det3((e), (f), (h), (i), (j), (l), (m), (n), (p));
	float _4 = - (d)* det3((e), (f), (g), (i), (j), (k), (m), (n), (o));
	////printf("%f %f %f %f \n", _1, _2, _3, _4);
	return _1 + _2 + _3 + _4;

}
__device__ void Adjugate(float a, float b, float c, float d, float e, float f, float g, float h, float  i, float j, float k, float l, float  m, float n, float o, float p, Matrix& A)
{
	A.row1.x = +1 * det3((f), (g), (h), (j), (k), (l), (n), (o), (p));
	A.row1.y = -1 * det3((e), (g), (h), (i), (k), (l), (m), (o), (p));
	A.row1.z = +1 * det3((e), (f), (h), (i), (j), (l), (m), (n), (p));
	A.row1.w = -1 * det3((e), (f), (g), (i), (j), (k), (m), (n), (o));

	A.row2.x = -1 * det3((b), (c), (d), (j), (k), (l), (n), (o), (p));
	A.row2.y = +1 * det3((a), (c), (d), (i), (k), (l), (m), (o), (p));
	A.row2.z = -1 * det3((a), (b), (d), (i), (j), (l), (m), (n), (p));
	A.row2.w = +1 * det3((a), (b), (c), (i), (j), (k), (m), (n), (o));

	A.row3.x = +1 * det3((b), (c), (d), (f), (g), (h), (n), (o), (p));
	A.row3.y = -1 * det3((a), (c), (d), (e), (g), (h), (m), (o), (p));
	A.row3.z = +1 * det3((a), (b), (d), (e), (f), (h), (m), (n), (p));
	A.row3.w = -1 * det3((a), (b), (c), (e), (f), (g), (m), (n), (o));

	A.row4.x = -1 * det3((b), (c), (d), (f), (g), (h), (j), (k), (l));
	A.row4.y = +1 * det3((a), (c), (d), (e), (g), (h), (i), (k), (l));
	A.row4.z = -1 * det3((a), (b), (d), (e), (f), (h), (i), (j), (l));
	A.row4.w = +1 * det3((a), (b), (c), (e), (f), (g), (i), (j), (k));
}

__device__ void Transpose(Matrix& A, Matrix& T)
{
	T.row1.x = A.row1.x;
	T.row2.x = A.row1.y;
	T.row3.x = A.row1.z;
	T.row4.x = A.row1.w;
	T.row1.y = A.row2.x;
	T.row2.y = A.row2.y;
	T.row3.y = A.row2.z;
	T.row4.y = A.row2.w;
	T.row1.z = A.row3.x;
	T.row2.z = A.row3.y;
	T.row3.z = A.row3.z;
	T.row4.z = A.row3.w;
	T.row1.w = A.row4.x;
	T.row2.w = A.row4.y;
	T.row3.w = A.row4.z;
	T.row4.w = A.row4.w;
}

__device__ void ResolveLinearSystem(Matrix& Q, float detQ, uint idx)
{

	// Q*X = [0,0,0,1]
	detQ = 1.0f / detQ;
	//if (idx == PRINT_IDX)
	//{
	//	printf("detQ %.9g\n", detQ);
	//}
	Matrix adj;
	Adjugate(Q.row1.x, Q.row1.y, Q.row1.z, Q.row1.w,
		Q.row2.x, Q.row2.y, Q.row2.z, Q.row2.w,
		Q.row3.x, Q.row3.y, Q.row3.z, Q.row3.w,
		Q.row4.x, Q.row4.y, Q.row4.z, Q.row4.w, adj);

	Transpose(adj, Q);
	Q.row1.x = detQ * Q.row1.x;
	Q.row2.x = detQ * Q.row2.x;
	Q.row3.x = detQ * Q.row3.x;
	Q.row4.x = detQ * Q.row4.x;
	Q.row1.y = detQ * Q.row1.y;
	Q.row2.y = detQ * Q.row2.y;
	Q.row3.y = detQ * Q.row3.y;
	Q.row4.y = detQ * Q.row4.y;
	Q.row1.z = detQ * Q.row1.z;
	Q.row2.z = detQ * Q.row2.z;
	Q.row3.z = detQ * Q.row3.z;
	Q.row4.z = detQ * Q.row4.z;
	Q.row1.w = detQ * Q.row1.w;
	Q.row2.w = detQ * Q.row2.w;
	Q.row3.w = detQ * Q.row3.w;
	Q.row4.w = detQ * Q.row4.w;


}


__device__ void CalculateRepresentativePoint(Matrix Q, Vertex& xn, uint idx)
{
	
	Q.row4.x = 0;
	Q.row4.y = 0;
	Q.row4.z = 0;
	Q.row4.w = 1;

	float det = det4(   Q.row1.x, Q.row1.y, Q.row1.z, Q.row1.w,
						Q.row2.x, Q.row2.y, Q.row2.z, Q.row2.w,
						Q.row3.x, Q.row3.y, Q.row3.z, Q.row3.w,
						0, 0, 0, 1);

	//if (idx == PRINT_IDX)
	//{
	//	printMatrix(Q);
	//}

	const float MIN_DETERMINANT_VALUE = 1e-2;
	
	if(abs(det) <= MIN_DETERMINANT_VALUE)
	{
		return;  //xn stays the same
	}
	//if (idx == PRINT_IDX)
	//{
	//	printf("det %.9g\n", det);
	//}

	ResolveLinearSystem(Q,det,idx);

	//if (idx == PRINT_IDX)
	//{
	//	printMatrix(Q);
	//}
}

__global__ void CalculateRepresentativePointsOfLeavesKernel(Matrix* dev_Ql, 
	Vertex* dev_totalSumVertexOFCl,
	uint* dev_vertexCountOfCl,
	GpuNode* dev_leafNodes,
	uint clCount)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= clCount)
	{
		return;
	}
	uint count = dev_vertexCountOfCl[idx];
	dev_leafNodes[idx].xn.x = dev_totalSumVertexOFCl[idx].x / count;
	dev_leafNodes[idx].xn.y = dev_totalSumVertexOFCl[idx].y / count;
	dev_leafNodes[idx].xn.z = dev_totalSumVertexOFCl[idx].z / count;
	//printMatrix(dev_Ql[idx]);
	CalculateRepresentativePoint(dev_Ql[idx], dev_leafNodes[idx].xn, -1);
	//if ((dev_leafNodes[idx].xn.x) > 1.22018 /*|| abs(dev_leafNodes[idx].xn.y) > 3.0f || abs(dev_leafNodes[idx].xn.z) > 3.0f*/)
	//{
		//printf("idx=%d xn=[%f %f %f]\n", idx, dev_leafNodes[idx].xn.x, dev_leafNodes[idx].xn.y, dev_leafNodes[idx].xn.z);
	//}
}

__global__ void CalculateFaceQuadricKernel(Vertex* dev_vertices,
	Matrix* dev_Quadrics,
	GPUFace* dev_GpuFaces,
	uint faceCount
)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= faceCount)
	{
		return;
	}
	
	Vertex u = dev_vertices[dev_GpuFaces[idx].vertexIndices.first];
	Vertex v = dev_vertices[dev_GpuFaces[idx].vertexIndices.second];
	Vertex z = dev_vertices[dev_GpuFaces[idx].vertexIndices.third];
	//if(dev_GpuFaces[idx].vertexIndices.first == 2 || dev_GpuFaces[idx].vertexIndices.second == 2 || dev_GpuFaces[idx].vertexIndices.third == 2)
	//printf("idx %d =>%d %f %f %f , %d %f %f %f, %d %f %f %f\n", idx, dev_GpuFaces[idx].vertexIndices.first,u.x, u.y, u.z,
	//	dev_GpuFaces[idx].vertexIndices.second,v.x, v.y, v.z, 
	//	dev_GpuFaces[idx].vertexIndices.third, z.x, z.y,z.z);

	// calculate two face vectors
	Vertex k = Sum(u, Product(v, -1));
	Vertex l = Sum(z, Product(v, -1));


	// cross gives the normal vector
	Vertex normal = Cross(k, l);
	normalize(normal);

	float a = normal.x;
	float b = normal.y;
	float c = normal.z;

	//ax + by + cz + d = 0 
	float d = -(u.x *  a + u.y * b + u.z * c);

	//printf("%f %f %f\n", powf(a,2) + powf(b,2) + powf(c,2) , u.x *  a + u.y * b + u.z * c, v.x *  a + v.y * b + v.z * c);
	////printf("%f %f %f -> %f\n", k.x * a, k.y * b, k.z* c,d);
	//if (dev_GpuFaces[idx].vertexIndices.first == 2 || dev_GpuFaces[idx].vertexIndices.second == 2 || dev_GpuFaces[idx].vertexIndices.third == 2)
	//	printf("idx %d  x %f y %f z %f  a %f b %f c %f d %f\n", idx, normal.x, normal.y, normal.z,a, b, c, d);

	Matrix* mtx = &dev_Quadrics[idx];
	// calculate Kp matrix (fundemental error quadric)
	mtx->row1 = make_float4(powf(a, 2), a * b, a * c, a * d);
	mtx->row2 = make_float4(a * b, powf(b, 2), b * c, b * d);
	mtx->row3 = make_float4(a * c, b * c, powf(c, 2), c * d);
	mtx->row4 = make_float4(a * d, b * d, c * d, powf(d, 2));


	/*if (dev_GpuFaces[idx].vertexIndices.first == 2 || dev_GpuFaces[idx].vertexIndices.second == 2 || dev_GpuFaces[idx].vertexIndices.third == 2)

		printf("idx %d\n %f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n\n",idx,
			mtx->row1.x,
			mtx->row1.y,
			mtx->row1.z,
			mtx->row1.w,
			mtx->row2.x,
			mtx->row2.y,
			mtx->row2.z,
			mtx->row2.w,
			mtx->row3.x,
			mtx->row3.y,
			mtx->row3.z,
			mtx->row3.w,
			mtx->row4.x,
			mtx->row4.y,
			mtx->row4.z,
			mtx->row4.w);*/
}

__global__ void VertexToMortonMapGenerateKernel(uint* dev_VertexToMortonMapping, uint*  dev_MortonToVertexMap, uint vertexCount)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= vertexCount)
	{
		return;
	}

	uint id = dev_MortonToVertexMap[idx]; // this ids are sorted w.r.t morton codes

	dev_VertexToMortonMapping[id] = idx;

	//if(idx == 67348 || idx == 67453)
	//	printf("dev_VertexToMortonMapping[%d] = %d\n",id,idx);

}

__global__ void MarkFirstOccurencesKernel(uint* dev_data, uint dataCount, uint* dev_output)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= dataCount)
	{
		return;
	}

	if (idx == 0)
	{
		dev_output[idx] = 0;
		return;
	}

	if (dev_data[idx] == dev_data[idx - 1])
	{
		dev_output[idx] = 0; // not first occurence
	}
	else dev_output[idx] = 1; // first occurence
}

__global__ void doesExistsKernel(uint2* dev_data, uint dataCount, uint* dev_output)
{

	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= dataCount)
	{
		return;
	}
	/*if (dev_output[dev_data[idx].y] == 1)
	{
		printf("idx %d dev_data[idx].y %d isLeaf %d dev_output[dev_data[idx].y]=%d\n", idx, dev_data[idx].y, dev_data[idx].x, dev_output[dev_data[idx].y]);
	}*/
	dev_output[dev_data[idx].y] = 1;
}

__global__ void doesExistsKernel(uint* dev_data, uint dataCount, uint* dev_output)
{

	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= dataCount)
	{
		return;
	}
	//printf("idx %d dev_data[idx] %d dev_output[dev_data[idx]]%d\n", idx, dev_data[idx], dev_output[dev_data[idx]]);
	dev_output[dev_data[idx]] = 1;
}


__global__ void SumLeafQuadricKernel(Matrix* dev_Ql, //Ql
	Vertex* dev_totalSumVertexOFCl,
	uint* dev_vertexCountOfCl,
	Vertex* dev_vertices,
	uint* dev_MortonSortedToClMapping,
	uint* dev_VertexToMortonMapping,
	Matrix* dev_QuadricsOfFaces,  //Qt
	GPUFace* dev_GpuFaces,
	const uint faceCount)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= faceCount)
	{
		return;
	}

	// get the vertex indices of face (a face has 3 vertices)
	uSeq3* verticesOfFace = &dev_GpuFaces[idx].vertexIndices;



	// the vertex index of face is for whole Mesh so we have saved a mapping before; dev_VertexToClMapping
	uint firstClOfFaceIdx = dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->first]];
	uint secondClOfFaceIdx = dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->second]];
	uint thirdClOfFaceIdx = dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->third]];

	
	/*
	////printf("idx=%d ; vertices Of Face %d %d %d ; cl of face %d %d %d\n",
	idx,
	verticesOfFace->first,
	verticesOfFace->second,
	verticesOfFace->third,
	firstClOfFaceIdx,
	secondClOfFaceIdx,
	thirdClOfFaceIdx);
	*/

	//  For every leaf node l, we compute a
	//	4x4 symmetric quadric matrix Ql following Garland and Heckbert[1997].To do so, we compute the face quadric Qt of each
	//	triangle t of M and sum it to the leaf quadrics of the vertices of t
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row1.x), dev_QuadricsOfFaces[idx].row1.x);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row1.y), dev_QuadricsOfFaces[idx].row1.y);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row1.z), dev_QuadricsOfFaces[idx].row1.z);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row1.w), dev_QuadricsOfFaces[idx].row1.w);

	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row2.x), dev_QuadricsOfFaces[idx].row2.x);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row2.y), dev_QuadricsOfFaces[idx].row2.y);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row2.z), dev_QuadricsOfFaces[idx].row2.z);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row2.w), dev_QuadricsOfFaces[idx].row2.w);

	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row3.x), dev_QuadricsOfFaces[idx].row3.x);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row3.y), dev_QuadricsOfFaces[idx].row3.y);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row3.z), dev_QuadricsOfFaces[idx].row3.z);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row3.w), dev_QuadricsOfFaces[idx].row3.w);

	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row4.x), dev_QuadricsOfFaces[idx].row4.x);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row4.y), dev_QuadricsOfFaces[idx].row4.y);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row4.z), dev_QuadricsOfFaces[idx].row4.z);
	atomicAdd(&(dev_Ql[firstClOfFaceIdx].row4.w), dev_QuadricsOfFaces[idx].row4.w);

	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row1.x), dev_QuadricsOfFaces[idx].row1.x);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row1.y), dev_QuadricsOfFaces[idx].row1.y);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row1.z), dev_QuadricsOfFaces[idx].row1.z);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row1.w), dev_QuadricsOfFaces[idx].row1.w);

	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row2.x), dev_QuadricsOfFaces[idx].row2.x);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row2.y), dev_QuadricsOfFaces[idx].row2.y);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row2.z), dev_QuadricsOfFaces[idx].row2.z);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row2.w), dev_QuadricsOfFaces[idx].row2.w);

	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row3.x), dev_QuadricsOfFaces[idx].row3.x);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row3.y), dev_QuadricsOfFaces[idx].row3.y);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row3.z), dev_QuadricsOfFaces[idx].row3.z);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row3.w), dev_QuadricsOfFaces[idx].row3.w);

	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row4.x), dev_QuadricsOfFaces[idx].row4.x);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row4.y), dev_QuadricsOfFaces[idx].row4.y);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row4.z), dev_QuadricsOfFaces[idx].row4.z);
	atomicAdd(&(dev_Ql[secondClOfFaceIdx].row4.w), dev_QuadricsOfFaces[idx].row4.w);

	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row1.x), dev_QuadricsOfFaces[idx].row1.x);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row1.y), dev_QuadricsOfFaces[idx].row1.y);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row1.z), dev_QuadricsOfFaces[idx].row1.z);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row1.w), dev_QuadricsOfFaces[idx].row1.w);

	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row2.x), dev_QuadricsOfFaces[idx].row2.x);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row2.y), dev_QuadricsOfFaces[idx].row2.y);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row2.z), dev_QuadricsOfFaces[idx].row2.z);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row2.w), dev_QuadricsOfFaces[idx].row2.w);

	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row3.x), dev_QuadricsOfFaces[idx].row3.x);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row3.y), dev_QuadricsOfFaces[idx].row3.y);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row3.z), dev_QuadricsOfFaces[idx].row3.z);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row3.w), dev_QuadricsOfFaces[idx].row3.w);

	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row4.x), dev_QuadricsOfFaces[idx].row4.x);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row4.y), dev_QuadricsOfFaces[idx].row4.y);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row4.z), dev_QuadricsOfFaces[idx].row4.z);
	atomicAdd(&(dev_Ql[thirdClOfFaceIdx].row4.w), dev_QuadricsOfFaces[idx].row4.w);


	/*
	*printf("face[%d]\n%f %f %f \n %f %f %f\n %f %f %f\n\n",idx,
	dev_QuadricsOfFaces[idx].row1.x,
	dev_QuadricsOfFaces[idx].row1.y,
	dev_QuadricsOfFaces[idx].row1.z,
	dev_QuadricsOfFaces[idx].row2.x,
	dev_QuadricsOfFaces[idx].row2.y,
	dev_QuadricsOfFaces[idx].row2.z,
	dev_QuadricsOfFaces[idx].row3.x,
	dev_QuadricsOfFaces[idx].row3.y,
	dev_QuadricsOfFaces[idx].row3.z);
	*/

	// we store the total sum vertex of each leaf node, its number of vertices to get mean vertex in the next step
	atomicAdd(&dev_totalSumVertexOFCl[firstClOfFaceIdx].x, dev_vertices[verticesOfFace->first].x);
	atomicAdd(&dev_totalSumVertexOFCl[firstClOfFaceIdx].y, dev_vertices[verticesOfFace->first].y);
	atomicAdd(&dev_totalSumVertexOFCl[firstClOfFaceIdx].z, dev_vertices[verticesOfFace->first].z);
	atomicAdd(&dev_vertexCountOfCl[firstClOfFaceIdx], 1);

	atomicAdd(&dev_totalSumVertexOFCl[secondClOfFaceIdx].x, dev_vertices[verticesOfFace->second].x);
	atomicAdd(&dev_totalSumVertexOFCl[secondClOfFaceIdx].y, dev_vertices[verticesOfFace->second].y);
	atomicAdd(&dev_totalSumVertexOFCl[secondClOfFaceIdx].z, dev_vertices[verticesOfFace->second].z);
	atomicAdd(&dev_vertexCountOfCl[secondClOfFaceIdx], 1);

	atomicAdd(&dev_totalSumVertexOFCl[thirdClOfFaceIdx].x, dev_vertices[verticesOfFace->third].x);
	atomicAdd(&dev_totalSumVertexOFCl[thirdClOfFaceIdx].y, dev_vertices[verticesOfFace->third].y);
	atomicAdd(&dev_totalSumVertexOFCl[thirdClOfFaceIdx].z, dev_vertices[verticesOfFace->third].z);
	atomicAdd(&dev_vertexCountOfCl[thirdClOfFaceIdx], 1);


	/*if (firstClOfFaceIdx == PRINT_IDX ||
		secondClOfFaceIdx == PRINT_IDX ||
		thirdClOfFaceIdx == PRINT_IDX)
	{
		
		Matrix& mtx = dev_Ql[firstClOfFaceIdx];
			printf("%d \n %f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n\n",idx,
				mtx.row1.x,
				mtx.row1.y,
				mtx.row1.z,
				mtx.row1.w,
				mtx.row2.x,
				mtx.row2.y,
				mtx.row2.z,
				mtx.row2.w,
				mtx.row3.x,
				mtx.row3.y,
				mtx.row3.z,
				mtx.row3.w,
				mtx.row4.x,
				mtx.row4.y,
				mtx.row4.z,
				mtx.row4.w);
			mtx = dev_Ql[secondClOfFaceIdx];
			printf("%d \n %f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n\n", idx,
				mtx.row1.x,
				mtx.row1.y,
				mtx.row1.z,
				mtx.row1.w,
				mtx.row2.x,
				mtx.row2.y,
				mtx.row2.z,
				mtx.row2.w,
				mtx.row3.x,
				mtx.row3.y,
				mtx.row3.z,
				mtx.row3.w,
				mtx.row4.x,
				mtx.row4.y,
				mtx.row4.z,
				mtx.row4.w);

			mtx = dev_Ql[thirdClOfFaceIdx];
			printf("%d \n %f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n%f  %f  %f %f \n\n", idx,
				mtx.row1.x,
				mtx.row1.y,
				mtx.row1.z,
				mtx.row1.w,
				mtx.row2.x,
				mtx.row2.y,
				mtx.row2.z,
				mtx.row2.w,
				mtx.row3.x,
				mtx.row3.y,
				mtx.row3.z,
				mtx.row3.w,
				mtx.row4.x,
				mtx.row4.y,
				mtx.row4.z,
				mtx.row4.w);

	}*/
	
}



__device__ void SumMatrix(Matrix& result, const Matrix& M1, const Matrix& M2)
{


	result.row1.x = M1.row1.x + M2.row1.x;
	result.row1.y = M1.row1.y + M2.row1.y;
	result.row1.z = M1.row1.z + M2.row1.z;

	result.row2.x = M1.row2.x + M2.row2.x;
	result.row2.y = M1.row2.y + M2.row2.y;
	result.row2.z = M1.row2.z + M2.row2.z;

	result.row3.x = M1.row3.x + M2.row3.x;
	result.row3.y = M1.row3.y + M2.row3.y;
	result.row3.z = M1.row3.z + M2.row3.z;

}

// En = xn * Qn * xn^T
__device__ float CalculateError(Vertex& v, Matrix& Qn)
{


	Vertex4 temp = make_float4(
		v.x * Qn.row1.x + v.y * Qn.row2.x + v.z * Qn.row3.x + 1 * Qn.row4.x,
		v.x * Qn.row1.y + v.y * Qn.row2.y + v.z * Qn.row3.y + 1 * Qn.row4.y,
		v.x * Qn.row1.z + v.y * Qn.row2.z + v.z * Qn.row3.z + 1 * Qn.row4.z,
		v.x * Qn.row1.w + v.y * Qn.row2.w + v.z * Qn.row3.w + 1 * Qn.row4.w
	);

	float retVal = temp.x * v.x + temp.y * v.y + temp.z * v.z + temp.w * 1;
	//if(retVal <= 0.00000f)
	//printf("%f -> %f %f %f %f and %f %f %f %f \n",retVal,
	//	temp.x,
	//	temp.y,
	//	temp.z,
	//	temp.w,
	//	xn.x,
	//	xn.y,
	//	xn.z,
	//	xn.w
	//	);
	return retVal;
}
__device__ inline int clz64(unsigned long long val) {
	unsigned int left = val >> 32;
	if (left == 0) {
		unsigned int right = (unsigned int)(val & 0xFFFFFFFFu);
		return 32 + ::__clz(right);
	}
	else {
		return ::__clz(left);
	}
}
__device__ int delta(int i, int j, unsigned int *sortedMortonCodes, int numObjects) {
	if (i < 0 || i >= numObjects || j < 0 || j >= numObjects) {
		return 0;
	}
	unsigned long long mortonCodeI = ((unsigned long long)sortedMortonCodes[i]) << 32 | (unsigned long long)i;
	unsigned long long mortonCodeJ = ((unsigned long long)sortedMortonCodes[j]) << 32 | (unsigned long long)j;
	if (mortonCodeI == mortonCodeJ) {
		return clz64((unsigned long long)i ^ (unsigned long long)j);
	}
	return clz64(mortonCodeI ^ mortonCodeJ);
}

__device__ inline int sign(int val) {
	return (0 < val) - (val < 0);
}

__device__ inline float2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int i) {
	float2 range;
	int d = sign(delta(i, i + 1, sortedMortonCodes, numObjects) - delta(i, i - 1, sortedMortonCodes, numObjects));

	int deltaMin = delta(i, i - d, sortedMortonCodes, numObjects);
	int lMax = 2;
	while (delta(i, i + lMax * d, sortedMortonCodes, numObjects) > deltaMin) {
		lMax = lMax << 1;
	}
	int l = 0;
	int t = lMax / 2;
	while (t >= 1) {
		if (delta(i, i + (l + t) * d, sortedMortonCodes, numObjects) > deltaMin) {
			l += t;
		}
		t = t / 2;
	}
	int j = i + l * d;

	range.x = min(i, j);
	range.y = max(i, j);

	return range;
}
__device__ inline int findSplit(unsigned int* sortedMortonCodes,
	int           first,
	int           last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned long long firstCode = (unsigned long long)sortedMortonCodes[first] << 32 | (unsigned long long)first;
	unsigned long long lastCode = (unsigned long long)sortedMortonCodes[last] << 32 | (unsigned long long)last;

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = clz64(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned long long splitCode = (unsigned long long)sortedMortonCodes[newSplit] << 32 | (unsigned long long)newSplit;
			int splitPrefix = clz64(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}


__global__  void generateKdTreeKernel(Matrix* dev_QScan,
	Vertex* dev_TotalVertexSumScan,
	uint* dev_vertexCountScan,
	uint* dev_Cl,
	uint clCount,
	GpuNode*     leafNodes,
	GpuNode* internalNodes)
{
#ifdef MY_WAY
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= clCount - 1) return;

	int i = idx;  // i is in [0 ,  n - 2]

				  // Determine the direction of the range (+1 or -1)
	int d = SIGN(delta(i, i + 1, dev_Cl, clCount) - delta(i, i - 1, dev_Cl, clCount));

	// Compute upper bound for the lenght of range
	int deltaMin = delta(i, i - d, dev_Cl, clCount);
	int lMax = 2;
	while (delta(i, i + lMax * d, dev_Cl, clCount) > deltaMin)
	{
		lMax = lMax << 1;    // lMax = lMax * 2
	}

	// Find the other end using binary search 
	int l = 0;
	int t = lMax / 2;
	while (t >= 1)
	{
		if (delta(i, i + (l + t) * d, dev_Cl, clCount) > deltaMin)
		{
			l += t;
		}
		t = t / 2;
	}
	int j = i + l * d;

	/*int2 range;
	range.x = min(i, j);
	range.y = max(i, j);

	i = range.x;
	j = range.y;
	*/
	//printf("i=%d  j=%d\n", i, j);

	// Find the split position using binary search

	int deltaNode = delta(i, j, dev_Cl, clCount);

	int gamma = i;

	int s = 0;
	t = l / 2;
	
	for (; t > 0; t /= 2)
	{
		if (delta(i, i + (s + t) * d, dev_Cl, clCount) > deltaNode)
		{
			s += t;
		}
	}
	gamma = i + s * d + min(d, 0);
#endif
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= clCount - 1) return;
	float2 range = determineRange(dev_Cl, clCount, idx);
	int first = range.x;
	int last = range.y;

	int gamma = findSplit(dev_Cl, first, last);

	// Output  child pointers
	GpuNode* leftChild;

	int i = first;
	int j = last;
	//if(idx == 30472) printf("idx=%d gamma=[%d] i=[%d] j=[%d]\n", idx, gamma, i, j);

	if (gamma == min(i, j))
	{
		leftChild = leafNodes + gamma;
		leftChild->isLeaf = true;
		leftChild->left = NULL;
		leftChild->right = NULL;
		leftChild->objectId = gamma;
		
		//if (idx == PRINT_IDX)printf("left of .%d. is L .%d.\n", idx, gamma);
	}
	else
	{
		leftChild = internalNodes + gamma;
		leftChild->objectId = gamma;
		
		//if (idx == PRINT_IDX)printf("left of .%d. is I .%d.\n", idx, gamma);
	}

	GpuNode* rightChild;
	if (gamma + 1 == max(i, j))
	{
		rightChild = leafNodes + (gamma + 1);
		rightChild->isLeaf = true;
		rightChild->left = NULL;
		rightChild->right = NULL;
		rightChild->objectId = gamma + 1 ;
		//if (idx == PRINT_IDX)printf("right of .%d. is L .%d.\n", idx, gamma +1);
	}
	else
	{
		rightChild = internalNodes + (gamma + 1);
		rightChild->objectId = gamma + 1;
		//if (idx == PRINT_IDX)printf("right of .%d. is I .%d.\n", idx, gamma +1);
	}

	// Record parent-child relationships.
	internalNodes[idx].objectId = idx;
	internalNodes[idx].left = leftChild;
	internalNodes[idx].right = rightChild;

	leftChild->parent = internalNodes + idx;
	rightChild->parent = internalNodes + idx;

	internalNodes[idx].objectId = idx;

	uint r2 = max(i, j);
	uint r1 = min(i, j);

	//if(idx == 86723) printf("r2 r1 %d %d\n", r2, r1);

	//*“Morton integral” allows
	//*	to compute any sum of attributes for consecutive leaves with only
	//*	two memory accesses; for instance in the case of the quadric of a
	//*	node n covering leaves r1 to r2 :
	//* Qn = Qscan[r2] − Qscan[r1 − 1]
	Matrix Qn;
	Matrix& M2 = dev_QScan[r2];
	Matrix& M1 = dev_QScan[r1];



	Qn.row1.x = M2.row1.x - M1.row1.x;
	Qn.row1.y = M2.row1.y - M1.row1.y;
	Qn.row1.z = M2.row1.z - M1.row1.z;
	Qn.row1.w = M2.row1.w - M1.row1.w;

	Qn.row2.x = M2.row2.x - M1.row2.x;
	Qn.row2.y = M2.row2.y - M1.row2.y;
	Qn.row2.z = M2.row2.z - M1.row2.z;
	Qn.row2.w = M2.row2.w - M1.row2.w;

	Qn.row3.x = M2.row3.x - M1.row3.x;
	Qn.row3.y = M2.row3.y - M1.row3.y;
	Qn.row3.z = M2.row3.z - M1.row3.z;
	Qn.row3.w = M2.row3.w - M1.row3.w;

	Qn.row4.x = M2.row4.x - M1.row4.x;
	Qn.row4.y = M2.row4.y - M1.row4.y;
	Qn.row4.z = M2.row4.z - M1.row4.z;
	Qn.row4.w = M2.row4.w - M1.row4.w;

	uint& c2 = dev_vertexCountScan[r2];
	uint& c1 = dev_vertexCountScan[r1];

	uint c = c2 - c1;

	Vertex& xn = internalNodes[idx].xn;
	Vertex& v2 = dev_TotalVertexSumScan[r2];
	Vertex& v1 = dev_TotalVertexSumScan[r1];
	
	xn.x = (v2.x - v1.x) ;
	xn.y = (v2.y - v1.y) ;
	xn.z = (v2.z - v1.z) ;
	xn.x /= c;
	xn.y /= c;
	xn.z /= c;
	//if (idx == PRINT_IDX)
	//{
	//	printf("idx=%d xn=[%.9g %.9g %.9g] c=%d\n", idx, xn.x, xn.y, xn.z, c);
	//	printf("idx=%d v2=[%f %f %f] v1=[%.9g %.9g %.9g] \n", idx, v2.x, v2.y, v2.z, v1.x, v1.y, v1.z);
	//	printf("idx=%d r2 %d r1 %d\n", idx, r2, r1);
	//}

	
	CalculateRepresentativePoint(Qn, xn, idx);
	



	/*if(idx == PRINT_IDX) printf("idx=%d xn=[%.9g %.9g %.9g]\n", idx, xn.x, xn.y, xn.z);*/
	
	

	internalNodes[idx].xn.x = xn.x;
	internalNodes[idx].xn.y = xn.y;
	internalNodes[idx].xn.z = xn.z;

	// En = xn Qn xnT
	//if(idx == PRINT_IDX) 		printf("idx %d %d %d %f %f %f\n", idx, c2,c1 , xn.x, xn.y, xn.z);
	
	internalNodes[idx].En = CalculateError(xn, Qn);
	
//	if (idx == PRINT_IDX) printf("internalNode[%d].En=%f\n", idx, internalNodes[idx].En);
}


__global__ void treeTraverselKernel(uint2* dev_P, uint clCount, GpuNode* dev_InternalNodes, GpuNode* dev_leafNodes, float2 theta, Vertex eyePosition, float distanceRange)
{
	float maxTheta = theta.x;
	float thetaRange = fabs(theta.x - theta.y);
	float calculatedTheta = theta.x;
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= clCount)
	{
		return;
	}

	uint c = 0;

	float error = FLT_MAX;

	GpuNode* node = &dev_InternalNodes[0]; // root;
	
	bool isLeaf = false;
	int i = 0;


	while(error > calculatedTheta)
	{
		
		GpuNode* r = node->right;
		GpuNode* l = node->left;
		isLeaf = node->isLeaf;
	
		//printf("idx %d node->objectId %d left %d right %d Leaf? %d %d\n", idx, node->objectId, l->objectId, r->objectId,l->isLeaf, r->isLeaf);
		if (node->isLeaf)
		{
			//c = node->objectId;
			break; //leaf
		}

		if(idx > l->objectId)
		{
			//printf("idx %d i %d c %d l->objectId %d r->objectId %d => go R \n", idx, i++, c, l->objectId,r->objectId);
			node = r;
		}
		else
		{
			// printf("idx %d i %d c %d l->objectId %d r->objectId %d => go L \n", idx, i++, c, l->objectId, r->objectId);
			node = l;
		}
		error = node->En;
		c = node->objectId;
		//printf("idx=%d  c=%d error=%f l=%d r=%d \n", idx,c, error, l->objectId , r->objectId);

		calculatedTheta = (Distance(eyePosition, node->xn) / distanceRange) * thetaRange + theta.y;
		if(calculatedTheta > theta.x || calculatedTheta < theta.y)
		printf("%f %f %f \n", calculatedTheta, distanceRange, thetaRange);

		++i;
	}
	dev_P[idx] = make_uint2(node->isLeaf, c) ;
	// printf("idx=%d isLeaf=%d  id=%d\n", idx, dev_P[idx].x, dev_P[idx].y);
}



__global__ void markTrianglesKernel(GPUFace* dev_GpuFaces, uint faceCount,
	uint2* dev_P,
	uint* dev_MarkedFaceIndices,
	uint* dev_MortonSortedToClMapping,
	uint* dev_VertexToMortonMapping,
	uint* dev_MortonToVertexMapping,
	uint* dev_PToVPrime)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= faceCount)
	{
		return;
	}

	// get the vertex indices of face (a face has 3 vertices)
	uSeq3* verticesOfFace = &dev_GpuFaces[idx].vertexIndices;



	// the vertex index of face is for whole Mesh so we have saved a mapping before; dev_VertexToClMapping
	uint firstPOfFaceIdx =   dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->first]];
	uint secondPOfFaceIdx = dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->second]];
	uint thirdPOfFaceIdx = dev_MortonSortedToClMapping[dev_VertexToMortonMapping[verticesOfFace->third]];

	/*firstPOfFaceIdx = dev_MortonToVertexMapping[firstPOfFaceIdx];
	secondPOfFaceIdx = dev_MortonToVertexMapping[secondPOfFaceIdx];
	thirdPOfFaceIdx = dev_MortonToVertexMapping[thirdPOfFaceIdx];*/

	

	uint p1 = dev_P[firstPOfFaceIdx].y;
	uint p2 = dev_P[secondPOfFaceIdx].y;
	uint p3 = dev_P[thirdPOfFaceIdx].y;

	uint v1 = dev_PToVPrime[p1];
	uint v2 = dev_PToVPrime[p2];
	uint v3 = dev_PToVPrime[p3];


	/*if(idx == 20578)
	printf("idx=%d \n v1 %d v2 %d v3 %d\n p1 %d p2 %d p3 %d\n cl of face %d %d %d\nvertices Of Face %d %d %d\n ",
		idx, 
		v1,v2,v3,
		p1, p2, p3,
		firstPOfFaceIdx,
		secondPOfFaceIdx,
		thirdPOfFaceIdx,
		verticesOfFace->first,
		verticesOfFace->second,
		verticesOfFace->third
		);*/

	bool v12 = v1 == v2;
	bool v13 = v1 == v3;
	bool v23 = v2 == v3;

	if (v12 || v13 || v23)  // triangle is not in the remesh
	{ 
		dev_MarkedFaceIndices[idx] = 0;
		//printf("%d is not in remesh %d %d %d\n",idx,p1,p2,p3);
	}
	else
	{
		dev_MarkedFaceIndices[idx] = 1;
	}

	//printf("dev_MarkedFaceIndices[%d] is %d\n", idx, dev_MarkedFaceIndices[idx]);
}

__global__ void calculateFaceCenterKernel(const GPUFace* dev_primitives, Vertex* dev_primitiveCenters , uint primitiveCount, Vertex* dev_remeshVertices)
{

	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= primitiveCount)
	{
		return;
	}

	Vertex v1 = dev_remeshVertices[dev_primitives[idx].vertexIndices.first];
	Vertex v2 = dev_remeshVertices[dev_primitives[idx].vertexIndices.second];
	Vertex v3 = dev_remeshVertices[dev_primitives[idx].vertexIndices.third];

	dev_primitiveCenters[idx] = make_float3((v1.x + v2.x + v3.x) / 3.0f,
											(v1.y + v2.y + v3.y) / 3.0f,
											(v1.z + v2.z + v3.z) / 3.0f);


	/*////printf("calculateFaceCenterKernel: %d , vertex index: %d %d %d\n x = %f %f %f \n y = %f %f %f \n z = %f %f %f \n c = %f %f %f \n"
		, idx, dev_primitives[idx].vertexIndices.first, dev_primitives[idx].vertexIndices.second, dev_primitives[idx].vertexIndices.third,
		v1.x ,v2.x , v3.x, v1.y, v2.y, v3.y, v1.z, v2.z, v3.z, 
		dev_primitiveCenters[idx].x, dev_primitiveCenters[idx].y, dev_primitiveCenters[idx].z);
	*/
}

__device__ void SetBoundary(Vertex& vertex, 
	float& minX, float& minY, float& minZ,
	float& maxX, float& maxY, float& maxZ)
{
	if (vertex.x < minX)
	{
		minX = vertex.x;
	}
	if (vertex.y < minY)
	{
		minY = vertex.y;
	}
	if (vertex.z < minZ)
	{
		minZ = vertex.z;
	}

	if (vertex.x > maxX)
	{
		maxX = vertex.x;
	}
	if (vertex.y > maxY)
	{
		maxY = vertex.y;
	}
	if (vertex.z > maxZ)
	{
		maxZ = vertex.z;
	}
}

__global__ void generateAABBKernel(GpuNode* dev_leafNodeBuffer, Vertex* vertices, GPUFace* primitives,uint primitiveCount)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= primitiveCount) return;


	uint a = primitives[idx].vertexIndices.first;
	uint b = primitives[idx].vertexIndices.second;
	uint c = primitives[idx].vertexIndices.third;

	Vertex Va = vertices[a];
	Vertex Vb = vertices[b];
	Vertex Vc = vertices[c];

	float maxX = -FLT_MAX;
	float minX = FLT_MAX;

	float maxY = -FLT_MAX;
	float minY = FLT_MAX;

	float maxZ = -FLT_MAX;
	float minZ = FLT_MAX;

		
		
	SetBoundary(Va, minX, minY, minZ, maxX, maxY, maxZ);
	SetBoundary(Vb, minX, minY, minZ, maxX, maxY, maxZ);
	SetBoundary(Vc, minX, minY, minZ, maxX, maxY, maxZ);
		
	stAABB& aabb = dev_leafNodeBuffer[idx].AABB;

	aabb.posMin.x = minX;
	aabb.posMin.y = minY;
	aabb.posMin.z = minZ;

	aabb.posMax.x = maxX;
	aabb.posMax.y = maxY;
	aabb.posMax.z = maxZ;
}




__global__  void generateHierarchyKernel(uint* sortedMortonCodes, 
	GpuNode*     leafNodes, GpuNode* internalNodes, uint primitiveCount)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= primitiveCount - 1) return;

	int i = idx;  // i is in [0 ,  n - 2]

	// Determine the direction of the range (+1 or -1)
	int deltaRight = delta(i, i + 1, sortedMortonCodes, primitiveCount);
	int deltaLeft = delta(i, i - 1, sortedMortonCodes, primitiveCount);
	int d = SIGN( deltaRight - deltaLeft );



	


	// Compute upper bound for the lenght of range
	int deltaMin = delta(i, i - d, sortedMortonCodes, primitiveCount);
	int lMax = 2;
	while (delta(i, i + lMax * d, sortedMortonCodes, primitiveCount) > deltaMin)
	{
		lMax = lMax << 1;    // lMax = lMax * 2
	}

	// Find the other end using binary search 
	int l = 0;
	int t = lMax / 2;
	while (t >= 1)
	{
		if (delta(i, i + (l + t) * d, sortedMortonCodes, primitiveCount) > deltaMin)
		{
			l += t;
		}
		t = t / 2;
	}
	int j = i + l * d;

	//printf("idx[%d] i=%d  j=%d  x=%f y=%f\n",idx, i, j,range.x,range.y);

	// Find the split position using binary search
	
	int deltaNode = delta(i, j, sortedMortonCodes, primitiveCount);

	int gamma = i;

	int s = 0;
	t = l / 2;
	//printf("t=%d l=%d\n", t, l);
	for (; t > 0; t /= 2)
	{
		//printf("t=%d\n", t);
		if (delta(i, i + (s + t) * d, sortedMortonCodes, primitiveCount) > deltaNode)
		{
			s += t;
		}
	}
	gamma = i + s * d + min(d, 0);


	//printf("idx[%d] gamma=%d  \n", idx, gamma);
	

	// Output  child pointers
	GpuNode* leftChild;


	if (gamma == min(i, j))
	{
		//printf("idx=%d left child leaf %d\n", idx,gamma);
		leftChild = &leafNodes[gamma];
		leftChild->left = NULL;
		leftChild->right = NULL;
	}
	else
	{
		//printf("idx=%d left child interal %d\n", idx, gamma);
		leftChild = &internalNodes[gamma];
	}

	GpuNode* rightChild;
	if (gamma + 1 == max(i, j))
	{
		//printf("idx=%d right child leaf %d\n", idx, gamma+1);
		rightChild = &leafNodes[gamma + 1];
		rightChild->left = NULL;
		rightChild->right = NULL;
	}
	else
	{
		//printf("idx=%d right child internal %d\n",idx, gamma+1);
		rightChild = &internalNodes[gamma + 1];
	}

	// Record parent-child relationships.
	//internalNodes[idx].idx = idx;
	internalNodes[idx].left = leftChild;
	internalNodes[idx].right = rightChild;

	leftChild->parent = &internalNodes[idx];
	rightChild->parent = &internalNodes[idx];

	internalNodes[idx].objectId = idx;
}


//To find the bounding box of a given node, the thread simply looks up the bounding boxes of its children and calculates their union
__global__ void boundingBoxCalculationKernel(GpuNode* leafNodes, uint primitiveCount)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= primitiveCount) return;

	GpuNode* node = &leafNodes[idx];

	while (node != NULL)
	{
		if (atomicExch(&node->flag, 1) == 0) //To avoid duplicate work, the idea is to use an atomic flag per node to terminate the first thread that enters it, while letting the second one through
		{
			return;
		}

		if (node->left != NULL)
		{
			node->AABB.posMin.x = fminf(node->AABB.posMin.x, node->left->AABB.posMin.x);
			node->AABB.posMin.y = fminf(node->AABB.posMin.y, node->left->AABB.posMin.y);
			node->AABB.posMin.z = fminf(node->AABB.posMin.z, node->left->AABB.posMin.z);
			node->AABB.posMax.x = fmaxf(node->AABB.posMax.x, node->left->AABB.posMax.x);
			node->AABB.posMax.y = fmaxf(node->AABB.posMax.y, node->left->AABB.posMax.y);
			node->AABB.posMax.z = fmaxf(node->AABB.posMax.z, node->left->AABB.posMax.z);
		}
		if (node->right != NULL)
		{
			node->AABB.posMin.x = fminf(node->AABB.posMin.x, node->right->AABB.posMin.x);
			node->AABB.posMin.y = fminf(node->AABB.posMin.y, node->right->AABB.posMin.y);
			node->AABB.posMin.z = fminf(node->AABB.posMin.z, node->right->AABB.posMin.z);
			node->AABB.posMax.x = fmaxf(node->AABB.posMax.x, node->right->AABB.posMax.x);
			node->AABB.posMax.y = fmaxf(node->AABB.posMax.y, node->right->AABB.posMax.y);
			node->AABB.posMax.z = fmaxf(node->AABB.posMax.z, node->right->AABB.posMax.z);
		}

		node = node->parent;
	}
}


__global__ void remeshKernel(GPUFace* Tprime, GPUFace* T, uint* primitiveIndices, uint Tcount,uint* VToMorton, uint* MortonToCl, uint2* dev_P, uint* dev_PToVPrime)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= Tcount) return;

	int Tidx = primitiveIndices[idx];

	uSeq3& VofT = T[Tidx].vertexIndices;
	
	uint a1 = VToMorton[VofT.first];
	uint a2 = VToMorton[VofT.second];
	uint a3 = VToMorton[VofT.third];


	uint b1 = MortonToCl[a1];
	uint b2 = MortonToCl[a2];
	uint b3 = MortonToCl[a3];

	uint c1 = dev_P[b1].y;
	uint c2 = dev_P[b2].y;
	uint c3 = dev_P[b3].y;

	uint d1 = dev_PToVPrime[c1];
	uint d2 = dev_PToVPrime[c2];
	uint d3 = dev_PToVPrime[c3];
	/*if(idx == 171)
		printf("idx %d Tidx %d \n V1=%d V2=%d V3=%d\na1: %d a2: %d a3: %d \nb1: %d b2: %d b3: %d \nc1: %d c2: %d c3: %d\nd1: %d d2: %d d3: %d\n", idx, Tidx, 
			VofT.first, VofT.second, VofT.third, 
			a1, a2, a3,
			b1, b2, b3, 
			c1, c2, c3,
			d1, d2, d3);*/

	
	 Tprime[idx].vertexIndices.first  = d1;
	 Tprime[idx].vertexIndices.second = d2;
	 Tprime[idx].vertexIndices.third  = d3;
}

__global__ void VtoVPrimeKernel(Vertex* VPrime, GpuNode* dev_leafNodes, GpuNode* dev_internalNodes, uint clCount, uint* MortonToV, uint2* P, uint* dev_PToVPrime)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= clCount) return;

	uint pIdx = P[idx].y;

	uint dev_PToVPrimeIdx = dev_PToVPrime[idx];

	
	if(P[idx].x == 1)
	{
		VPrime[dev_PToVPrimeIdx].x = dev_leafNodes[pIdx].xn.x;
		VPrime[dev_PToVPrimeIdx].y = dev_leafNodes[pIdx].xn.y;
		VPrime[dev_PToVPrimeIdx].z = dev_leafNodes[pIdx].xn.z;
	}
	else
	{
		VPrime[dev_PToVPrimeIdx].x = dev_internalNodes[pIdx].xn.x;
		VPrime[dev_PToVPrimeIdx].y = dev_internalNodes[pIdx].xn.y;
		VPrime[dev_PToVPrimeIdx].z = dev_internalNodes[pIdx].xn.z;
	}

}


__global__  void dumpPaths(GpuNode* I, uint count) 
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= count) return;
	GpuNode* node = &I[0];

	while(node->isLeaf == false)
	{
		
		if (idx > node->left->objectId)
		{
			////printf("idx %d left %d rght %d => go R \n", idx, node->left->objectId, node->right->objectId);
			node = node->right;
		}
		else
		{
			////printf("idx %d left %d rght %d => go L \n", idx, node->left->objectId, node->right->objectId);
			node = node->left;
		}
	}
	////printf("idx %d is reached %d\n", idx, node->objectId);
}

__global__ void PToVPrimeKernel(uint clCount, uint2* P, uint* PToVPrime)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= clCount) return;

	PToVPrime[idx] = P[idx].y;
}