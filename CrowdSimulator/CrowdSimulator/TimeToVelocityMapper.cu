#include "TimeToVelocityMapper.h"

#include "GlobalConstants.h"
#include "CudaHelper.h"
#include <cassert>
#include <device_launch_parameters.h>
#include <math.h>
#include "TransferHelper.h"

void TimeToVelocityMapper::PreprareModule()
{
	assert(m_velocityXResult.m_array == NULL);
	assert(m_velocityYResult.m_array == NULL);

	m_velocityXResult = TransferHelper::ReserveFloatMemory();
	m_velocityYResult = TransferHelper::ReserveFloatMemory();

	m_extremPoint = TransferHelper::ReserveUnsignedMemory();

	assert(m_velocityXResult.m_stride == m_velocityYResult.m_stride);
	assert(m_extremPoint.m_stride == m_velocityXResult.m_stride);
}

void TimeToVelocityMapper::FreeResources()
{
	m_velocityXResult.FreeArray();
	m_velocityYResult.FreeArray();
	m_extremPoint.FreeArray();
}

__global__ void ComputeVelocityCuda(float* inputField,  unsigned int* blockedField, unsigned int* targetField,
                                    float* velocityX, float* velocityY,
                                    unsigned int* extremPointInfo, size_t strides)
{
	__shared__ float inputBuffer[gBlockSize + 2][gBlockSize + 2];
	__shared__ unsigned int locallyBlockedField[gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	inputBuffer[xScan][yScan] = inputField[xOrigin + yOrigin * strides];
	locallyBlockedField[xScan][yScan] = blockedField[xOrigin + yOrigin * strides];

	if (threadIdx.x == 0)
	{
		inputBuffer[0][yScan] = inputField[(xOrigin - 1) + yOrigin * strides];
		locallyBlockedField[0][yScan] = blockedField[(xOrigin - 1) + yOrigin * strides];
	}
	if (threadIdx.x == 31)
	{
		inputBuffer[xScan + 1][yScan] = inputField[(xOrigin + 1) + yOrigin * strides];
		locallyBlockedField[xScan + 1][yScan] = blockedField[(xOrigin + 1) + yOrigin * strides];
	}
	if (threadIdx.y == 0)
	{
		inputBuffer[xScan][0] = inputField[xOrigin + (yOrigin - 1)* strides];
		locallyBlockedField[xScan][0] = blockedField[xOrigin + (yOrigin - 1)* strides];
	}
	if (threadIdx.y == 31)
	{
		inputBuffer[xScan][yScan + 1] = inputField[xOrigin + (yOrigin + 1) * strides];
		locallyBlockedField[xScan][yScan + 1] = blockedField[xOrigin + (yOrigin + 1) * strides];
	}

	__syncthreads();

	

	int localExtremum = 0;

	// We start with the x component.
	bool leftValid = ((xOrigin != 1) && (locallyBlockedField[xScan - 1][yScan] == 0));
	bool rightValid = ((xOrigin != gGridSizeExternal - 2) && (locallyBlockedField[xScan + 1][yScan] == 0));
	float xDerivative = 0.0f;

	if (rightValid && leftValid)
	{
		float currentElement = inputBuffer[xScan][yScan];
		float rightElement = inputBuffer[xScan + 1][yScan];
		float leftElement = inputBuffer[xScan - 1][yScan];
		xDerivative = (rightElement - leftElement) / (2.0f * gCellSize);

		if (((currentElement > rightElement) && (currentElement > leftElement)) ||
			((currentElement < rightElement) && (currentElement < leftElement)))
			localExtremum = 1;

	}
	else if (rightValid)
	{
		xDerivative = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	else if (leftValid)
	{
		xDerivative = (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / (gCellSize);
	}


	// Now the same with the y component.
	bool topValid = ((yOrigin != 1) && (locallyBlockedField[xScan][yScan - 1] == 0));
	bool bottomValid = ((yOrigin != gGridSizeExternal - 2) && (locallyBlockedField[xScan][yScan + 1] == 0));
	float yDerivative = 0.0f;

	if (topValid && bottomValid)
	{
		float currentElement = inputBuffer[xScan][yScan];
		float topElement = inputBuffer[xScan][yScan + 1];
		float bottomElement = inputBuffer[xScan][yScan - 1];
		yDerivative =  (topElement - bottomElement) / (2.0f * gCellSize);

		if (((currentElement > topElement) && (currentElement > bottomElement)) ||
			((currentElement < topElement) && (currentElement < bottomElement)))
			localExtremum += 2;
	}
	else if (bottomValid)
	{
		yDerivative  = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	else if (topValid)
	{
		yDerivative =  (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize);
	}



	float xVelocity, yVelocity;
	
	// The magnitude of the time gradient is 1.0f / magnitude velocity.
	float sqrMag = xDerivative * xDerivative + yDerivative * yDerivative;

	// Velocity is in opposite of the gradient.
	xVelocity = -xDerivative / sqrMag;
	yVelocity = -yDerivative / sqrMag;

	if (isnan(yVelocity))
		yVelocity = 0.0f;

	if (isnan(xVelocity))
		xVelocity = 0.0f;

	if (targetField[xOrigin + yOrigin * strides] != 0)
	{
		xVelocity = 0.0f;
		yVelocity = 0.0f;
	}

	velocityX[xOrigin + yOrigin * strides] = xVelocity;
	velocityY[xOrigin + yOrigin * strides] = yVelocity;

	extremPointInfo[xOrigin + yOrigin * strides] = localExtremum;

}

void TimeToVelocityMapper::ComputeVelocity(FloatArray inputField, UnsignedArray blockedElements,
                                           UnsignedArray targetElements)
{
	assert(m_velocityXResult.m_array);
	assert(m_velocityXResult.m_array);
	assert(inputField.m_stride == blockedElements.m_stride);
	assert(inputField.m_stride == m_velocityXResult.m_stride);
	assert(targetElements.m_stride == m_velocityXResult.m_stride);
	ComputeVelocityCuda CUDA_DECORATOR_LOGIC(inputField.m_array, blockedElements.m_array, targetElements.m_array,
	                                         m_velocityXResult.m_array, m_velocityYResult.m_array,
	                                         m_extremPoint.m_array, m_velocityXResult.m_stride);

}




