#include "CrowdPressure.h"
#include "TransferHelper.h"
#include "DataBase.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>

void CrowdPressure::ToolSystem()
{
	m_pressureArray = TransferHelper::ReserveFloatMemory();
}

__global__ void ComputeCrowdPressureCuda(size_t strides, float* densityField, float* velocityField, unsigned int* wallData,  float* result)
{

	__shared__ float inputBuffer[gBlockSize + 2][gBlockSize + 2];
	__shared__ unsigned int locallyBlockedField[gBlockSize + 2][gBlockSize + 2];

	
	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	int totalIndex = xOrigin + yOrigin * strides;

	inputBuffer[xScan][yScan] = velocityField[totalIndex];
	locallyBlockedField[xScan][yScan] = wallData[totalIndex];

	if (threadIdx.x == 0)
	{
		inputBuffer[0][yScan] = velocityField[(xOrigin - 1) + yOrigin * strides];
		locallyBlockedField[0][yScan] = wallData[(xOrigin - 1) + yOrigin * strides];
	}
	if (threadIdx.x == 31)
	{
		inputBuffer[xScan + 1][yScan] = velocityField[(xOrigin + 1) + yOrigin * strides];
		locallyBlockedField[xScan + 1][yScan] = wallData[(xOrigin + 1) + yOrigin * strides];
	}
	if (threadIdx.y == 0)
	{
		inputBuffer[xScan][0] = velocityField[xOrigin + (yOrigin - 1)* strides];
		locallyBlockedField[xScan][0] = wallData[xOrigin + (yOrigin - 1)* strides];
	}
	if (threadIdx.y == 31)
	{
		inputBuffer[xScan][yScan + 1] = velocityField[xOrigin + (yOrigin + 1) * strides];
		locallyBlockedField[xScan][yScan + 1] = wallData[xOrigin + (yOrigin + 1) * strides];
	}

	__syncthreads();

	bool leftValid = (locallyBlockedField[xScan - 1][yScan] != 0);
	bool rightValid = (locallyBlockedField[xScan + 1][yScan] != 0);
	float xGrad = 0.0f;

	if (leftValid && rightValid)
	{
		xGrad = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan - 1][yScan]) / (2.0f * gCellSize);
	}
	else if (rightValid)
	{
		xGrad = (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / (gCellSize);
	}
	else if (leftValid)
	{
		xGrad = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan]) / ( gCellSize);

	}



	bool topValid = (locallyBlockedField[xScan][yScan - 1] == 0);
	bool bottomValid = (locallyBlockedField[xScan][yScan + 1] == 0);
	float yGrad = 0.0f;

	if (topValid && bottomValid)
	{
		yGrad = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan - 1]) / (2.0f * gCellSize);
	}
	else if (topValid)
	{
		yGrad = (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize);
	}
	else if (bottomValid)
	{
		yGrad = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	
	
	
	result[totalIndex] = densityField[totalIndex] * (xGrad * xGrad + yGrad * yGrad);
}

void CrowdPressure::ComputeCrowdPressure(FloatArray density, FloatArray velocity, DataBase* dataBase)
{
	
	UnsignedArray wallData = dataBase->GetWallData();
	
	assert(density.m_stride == velocity.m_stride);
	assert(density.m_stride == m_pressureArray.m_stride);
	assert(density.m_stride == wallData.m_stride);
	
	ComputeCrowdPressureCuda CUDA_DECORATOR_LOGIC (density.m_stride, density.m_array, velocity.m_array,  wallData.m_array   ,m_pressureArray.m_array);
	
}
