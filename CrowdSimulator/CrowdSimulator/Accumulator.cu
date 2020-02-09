#include "Accumulator.h"
#include "CudaHelper.h"
#include <cuda_runtime.h>
#include <cassert>
#include <device_launch_parameters.h>
#include <math.h>


void Accumulator::ToolSystem()
{
	assert(m_accumulationResult == NULL);
	cudaMalloc<AccuResult>(&m_accumulationResult, sizeof(AccuResult));
}

void Accumulator::FreeResources()
{
	cudaFree(m_accumulationResult);
	m_accumulationResult = NULL;
}


__global__ void ResetCounters(AccuResult* accuResult)
{
	accuResult->m_sum = 0.0f;
	accuResult->m_maximum = 0.0f;
}

__global__ void CountValues(float* arrayToProcess, size_t stride,  AccuResult* resultStructure)
{
	__shared__ float maximum[2][32][32];
	__shared__ float sum[2][32][32];

	int xBase = (blockIdx.x * 32 + threadIdx.x) * 3 + 1;
	int yBase = (blockIdx.y * 32 + threadIdx.y) * 3 + 1;

	float locSum = 0.0f;
	float locMax = 0.0f;
	for(int j = 0; j < 3; ++j)
		for(int i = 0;  i < 3; ++i)
		{
			float locVal = arrayToProcess[(xBase + i) + (yBase + j) * stride];
			locSum += locVal;
			locMax = fmaxf(locMax, locVal);
		}

	maximum[0][threadIdx.x][threadIdx.y] = locMax;
	sum[0][threadIdx.x][threadIdx.y] = locSum;

	__syncthreads();

	// Do a parallel reduce on that.
	int granularity = 16;
	int sourceBuffer = 0;
	for(int gen = 0; gen < 5; ++gen)
	{
		if ((threadIdx.x < granularity) && (threadIdx.y < granularity))
		{
			locSum = 0.0f;
			locMax = 0.0f;
			for(int i = 0; i < 2; ++i)
				for(int j = 0; j < 2; ++j)
				{
					locSum += sum[sourceBuffer][threadIdx.x + granularity * i][threadIdx.y + granularity * j];
					locMax = fmaxf(locMax, maximum[sourceBuffer][threadIdx.x + granularity * i][threadIdx.y + granularity * j]);
				}
			
			maximum[1 - sourceBuffer][threadIdx.x][threadIdx.y] = locMax;
			sum[1 - sourceBuffer][threadIdx.x][threadIdx.y] = locSum;
		}

		granularity /= 2;
		sourceBuffer = 1 - sourceBuffer;
		__syncthreads();
	}

	// If we are not the first thread of the block we are done here.
	if ((threadIdx.x != 0) || (threadIdx.y != 0))
		return;

	// Copy over our result.
	atomicAdd(&(resultStructure->m_sum), sum[sourceBuffer][0][0]);
	// Atmic max only exists for ints. We do a compare and swap construct here.
	int old = *((int *) &(resultStructure->m_maximum));
	while(true)
	{
		float attempt = fmaxf(resultStructure->m_maximum, maximum[sourceBuffer][0][0]);
		int converted = *((int *) &(attempt));
		int current = atomicCAS((int *) &(resultStructure->m_maximum), old, converted);
		if (current == old)
			return;
		
		// Do a new attempt.
		old = current;
	}
	
}

void Accumulator::ProcessField(FloatArray fieldToProcess)
{
	assert(m_accumulationResult);
	// WARNING: Method assumes a grid dimension multiple of three in the moment.
	assert(gNumOfBlocks % 3 == 0);
	
	ResetCounters << <1, 1 >> > (m_accumulationResult);
	CountValues << <dim3(gNumOfBlocks / 3, gNumOfBlocks / 3), dim3(32, 32) >> > (fieldToProcess.m_array, fieldToProcess.m_stride, m_accumulationResult);
}

void Accumulator::GetResult(float& sum, float& maximum)
{
	AccuResult result;
	cudaDeviceSynchronize();
	cudaMemcpy(&result, m_accumulationResult, sizeof(AccuResult), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	sum = result.m_sum;
	maximum = result.m_maximum;
	
}




