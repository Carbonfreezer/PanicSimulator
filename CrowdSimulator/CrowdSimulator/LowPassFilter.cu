#include "LowPassFilter.h"
#include "TransferHelper.h"
#include "CudaHelper.h"
#include <crt/host_defines.h>
#include <cassert>
#include <device_launch_parameters.h>

__constant__ float convolutionKernel[3][3] = {
	{1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f},
	{1.0f / 8.0f, 1.0f / 4.0f, 1.0f / 8.0f} ,
	{1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f}
};


void LowPassFilter::PrepareModule()
{
	m_result = TransferHelper::ReserveFloatMemory();
}

__global__ void CudaFilter(float* input, float* output, size_t stride, unsigned int* blocked, size_t blockedStride)
{
	// We want to low pass filter the stuff.
	__shared__ float valueBuffer[gBlockSize + 2][gBlockSize + 2];
	__shared__ unsigned int blockedBuffer[gBlockSize + 2][gBlockSize + 2];

	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;


	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	valueBuffer[xScan][yScan] = input[xOrigin + yOrigin * stride];
	blockedBuffer[xScan][yScan] = blocked[xOrigin + yOrigin * blockedStride];
	
	// Deal with the die cases-
	if (threadIdx.x == 0)
	{
		valueBuffer[xScan - 1][yScan] = input[xOrigin - 1 + yOrigin * stride];
		blockedBuffer[xScan - 1][yScan] = blocked[xOrigin - 1 + yOrigin * blockedStride];
	}
	if (threadIdx.x == 31)
	{
		valueBuffer[xScan + 1][yScan] = input[xOrigin + 1 + yOrigin * stride];
		blockedBuffer[xScan + 1][yScan] = blocked[xOrigin + 1 + yOrigin * blockedStride];
	}
	if (threadIdx.y == 0)
	{
		valueBuffer[xScan][yScan - 1] = input[xOrigin + (yOrigin - 1) * stride];
		blockedBuffer[xScan][yScan - 1] = blocked[xOrigin + (yOrigin - 1) * blockedStride];
	}
	if (threadIdx.y == 31)
	{
		valueBuffer[xScan][yScan + 1] = input[xOrigin + (yOrigin + 1) * stride];
		blockedBuffer[xScan][yScan + 1] = blocked[xOrigin + (yOrigin + 1) * blockedStride];
	}
	

	// Deal with the corner cases.
	if ((threadIdx.x == 0) && (threadIdx.y == 0))
	{
		
		valueBuffer[0][0] = input[(xOrigin - 1) + (yOrigin - 1) * stride];
		blockedBuffer[0][0] = blocked[(xOrigin - 1) + (yOrigin - 1) * blockedStride];

		valueBuffer[gBlockSize + 1][0] = input[(xOrigin + gBlockSize) + ( yOrigin - 1) * stride];
		blockedBuffer[gBlockSize + 1][0] = blocked[(xOrigin + gBlockSize) + (yOrigin - 1) * blockedStride];

		valueBuffer[0][gBlockSize + 1] = input[(xOrigin - 1) + (yOrigin + gBlockSize) * stride];
		blockedBuffer[0][gBlockSize + 1] = blocked[(xOrigin - 1) + (yOrigin  + gBlockSize) * blockedStride];

		valueBuffer[gBlockSize + 1][gBlockSize + 1] = input[(xOrigin + gBlockSize) + (yOrigin + gBlockSize) * stride];
		blockedBuffer[gBlockSize + 1][gBlockSize + 1] = blocked[(xOrigin + gBlockSize) + (yOrigin + gBlockSize) * blockedStride];
		


	}

	__syncthreads();

	if (blockedBuffer[xScan][yScan])
	{
		output[xOrigin + yOrigin * stride] = valueBuffer[xScan][yScan];
		return;
	}

	// Now we build the low pass filter over the 9 elements.
	float accumulatedWeight = 0.0f;
	float accumulatedResult = 0.0f;

	for (int j = -1; j < 2; ++j)
		for (int i = -1; i < 2; ++i)
		{
			if ((blockedBuffer[xScan + i][yScan + j]) || (xOrigin + i == 0) || (yOrigin + j == 0) || (xOrigin + i == gGridSizeExternal - 1) || (yOrigin + j == gGridSizeExternal - 1))
				continue;
			
			float localWeight = convolutionKernel[i + 1][j + 1];
			accumulatedWeight += localWeight;
			accumulatedResult += localWeight * valueBuffer[xScan + i][yScan + j];
		}

	accumulatedResult /= accumulatedWeight;

	output[xOrigin + yOrigin * stride] = accumulatedResult;
}

void LowPassFilter::Filter(FloatArray inputField, UnsignedArray blockedElements)
{
	assert(inputField.m_stride == m_result.m_stride);
	CudaFilter CUDA_DECORATOR_LOGIC (inputField.m_array, m_result.m_array, m_result.m_stride, blockedElements.m_array, blockedElements.m_stride);
}


