#include "GradientModule.h"
#include "GlobalConstants.h"
#include "CudaHelper.h"
#include <cassert>
#include <device_launch_parameters.h>
#include <math.h>
#include "VisualizationHelper.h"
#include "TransferHelper.h"

void GradientModule::PreprareModule()
{
	assert(m_gradientResultX.m_array == NULL);
	assert(m_gradientResultY.m_array == NULL);


						
	m_gradientResultX = TransferHelper::ReserveFloatMemory();
	m_gradientResultY = TransferHelper::ReserveFloatMemory();

	m_extremPoint = TransferHelper::ReserveUnsignedMemory();

	assert(m_gradientResultX.m_stride == m_gradientResultY.m_stride);
	assert(m_extremPoint .m_stride == m_gradientResultX.m_stride);
}

__global__ void ComputeGradientCuda(float* inputField, size_t inputStride, unsigned int* blockedField, size_t blockedStride, float* gradientX, float* gradientY,
                       unsigned int* extremPointInfo, size_t gradientStride)
{
	__shared__ float inputBuffer[gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;
	
	inputBuffer[xScan][yScan] =  inputField[xOrigin + yOrigin  * inputStride];

	if (threadIdx.x == 0)
		inputBuffer[0][yScan] = inputField[(xOrigin - 1) + yOrigin  * inputStride];
	if (threadIdx.x == 31)
		inputBuffer[xScan + 1][yScan] =  inputField[(xOrigin + 1) + yOrigin  * inputStride];
	if (threadIdx.y == 0)
		inputBuffer[xScan][0]  = inputField[xOrigin  + (yOrigin - 1)* inputStride];
	if (threadIdx.y == 31)
		inputBuffer [xScan][yScan + 1]  = inputField[ xOrigin  + (yOrigin + 1) * inputStride];


	
	__syncthreads();

	if (blockedField[xOrigin + yOrigin * blockedStride] != 0)
	{
		gradientX[xOrigin + yOrigin * gradientStride] = 0.0f;
		gradientY[xOrigin + yOrigin * gradientStride] = 0.0f;
		return;
	}

	int localExtremum = 0;

	// We start with the x component.
	bool leftValid = ((xOrigin != 1) && (blockedField[(xOrigin - 1) + yOrigin * blockedStride]) == 0);
	bool rightValid = ((xOrigin != gGridSizeExternal - 2) && (blockedField[(xOrigin + 1) + yOrigin * blockedStride]) == 0);
	float result = 0.0f;

	
	
	if (rightValid && leftValid)
	{
		float currentElement = inputBuffer[xScan][yScan];
		float rightElement = inputBuffer[xScan + 1][yScan];
		float leftElement = inputBuffer[xScan - 1][yScan];
		result = ( rightElement - leftElement ) / (2.0f * gCellSize);

		if (((currentElement > rightElement) && (currentElement > leftElement)) ||
			((currentElement < rightElement) && (currentElement < leftElement)))
			localExtremum = 1;
		
	} else if (rightValid)
	{
		result =  (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan])  / ( gCellSize);
		// Left is invalid in that case, we do not want to get a component pointing to the right.
		result = fminf(0.0f, result);
	} else if (leftValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / (gCellSize);
		// Right is invalid in that case.
		result = fmaxf(0.0f, result);
	}

	if (isnan(result))
		result = 0.0f;

	gradientX[xOrigin  + yOrigin  * gradientStride] = result;

	
	// Now the same with the y component.
	bool topValid = ((yOrigin != 1) && (blockedField[xOrigin  + (yOrigin - 1) * blockedStride]) == 0);
	bool bottomValid = ((yOrigin != gGridSizeExternal - 2) && (blockedField[xOrigin + (yOrigin + 1) * blockedStride]) == 0);
	result = 0.0f;

	
	
	if (topValid && bottomValid)
	{
		float currentElement = inputBuffer[xScan][yScan];
		float topElement = inputBuffer[xScan][yScan + 1];
		float bottomElement = inputBuffer[xScan][yScan - 1];
		result = (topElement - bottomElement) / (2.0f * gCellSize);

		if (((currentElement > topElement) && (currentElement > bottomElement)) ||
			((currentElement < topElement) && (currentElement < bottomElement)))
			localExtremum += 2;
	}
	else if (bottomValid)
	{
		result = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize);
		result = fminf(0.0f, result);
	}
	else if (topValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize);
		result = fmaxf(0.0f, result);
	}

	if (isnan(result))
		result = 0.0f;

	
	gradientY[xOrigin + yOrigin * gradientStride] = result;

	extremPointInfo[xOrigin + yOrigin * gradientStride] = localExtremum;
	
}

void GradientModule::ComputeGradient(FloatArray inputField, UnsignedArray blockedElements)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);
	ComputeGradientCuda CUDA_DECORATOR_LOGIC (inputField.m_array, inputField.m_stride, blockedElements.m_array, blockedElements.m_stride, 
		m_gradientResultX.m_array, m_gradientResultY.m_array, m_extremPoint.m_array,  m_gradientResultX.m_stride);
}




void GradientModule::VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);

	FloatArray fieldToVisualize = component ? m_gradientResultY : m_gradientResultX;
	VisualizationHelper::VisualizeScalarFieldWithNegative(fieldToVisualize, maxExepctedValue,  textureMemory);
}
