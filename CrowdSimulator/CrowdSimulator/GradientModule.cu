#include "GradientModule.h"
#include "GlobalConstants.h"
#include "CudaHelper.h"
#include <cassert>
#include <device_launch_parameters.h>

void GradientModule::PreprareModule()
{
	assert(m_gradientResultX == NULL);
	assert(m_gradientResultY == NULL);

	m_gradientResultX = m_transferHelper.ReserveFloatMemory(m_gradientStride);
	m_gradientResultY = m_transferHelper.ReserveFloatMemory(m_gradientStride);
}

__global__ void ComputeGradientCuda(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride, float* gradientX, float* gradientY,
                       size_t gradientStride)
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

	if (wallField[xOrigin + yOrigin * wallStride] != 0)
	{
		gradientX[xOrigin + yOrigin * gradientStride] = 0.0f;
		gradientY[xOrigin + yOrigin * gradientStride] = 0.0f;
		return;
	}


	// We start with the x component.
	bool leftValid = ((xOrigin != 1) && (wallField[(xOrigin - 1) + yOrigin * wallStride]) == 0);
	bool rightValid = ((xOrigin != gGridSizeExternal - 2) && (wallField[(xOrigin + 1) + yOrigin * wallStride]) == 0);
	float result = 0.0f;

	
	
	if (rightValid && leftValid)
	{
		result = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan - 1][yScan]) / (2.0f * gCellSize);
	} else if (rightValid)
	{
		result =  (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan])  / ( gCellSize);
	} else if (leftValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / (gCellSize);
	}

	gradientX[xOrigin  + yOrigin  * gradientStride] = result;

	
	// Now the same with the y component.
	bool topValid = ((yOrigin != 1) && (wallField[xOrigin  + (yOrigin - 1) * wallStride]) == 0);
	bool bottomValid = ((yOrigin != gGridSizeExternal - 2) && (wallField[xOrigin + (yOrigin + 1) * wallStride]) == 0);
	result = 0.0f;

	if (topValid && bottomValid)
	{
		result = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan - 1]) / (2.0f * gCellSize);
	}
	else if (bottomValid)
	{
		result = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	else if (topValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize);
	}

	gradientY[xOrigin + yOrigin * gradientStride] = result;
	
}

void GradientModule::ComputeGradient(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride)
{
	ComputeGradientCuda CUDA_DECORATOR_LOGIC (inputField, inputStride, wallField, wallStride, m_gradientResultX, m_gradientResultY, m_gradientStride);
}


__global__ void CompueXDerivativeCuda(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride, float* gradientX,
	size_t gradientStride)
{
	__shared__ float inputBuffer[gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;


	inputBuffer[xScan][yScan] = inputField[xOrigin + yOrigin * inputStride];


	if (threadIdx.x == 0)
		inputBuffer[0][yScan] = inputField[(xOrigin - 1) + yOrigin * inputStride];
	if (threadIdx.x == 31)
		inputBuffer[xScan + 1][yScan] = inputField[(xOrigin + 1) + yOrigin * inputStride];
	if (threadIdx.y == 0)
		inputBuffer[xScan][0] = inputField[xOrigin + (yOrigin - 1)* inputStride];
	if (threadIdx.y == 31)
		inputBuffer[xScan][yScan + 1] = inputField[xOrigin + (yOrigin + 1) * inputStride];



	__syncthreads();

	if (wallField[xOrigin + yOrigin * wallStride] != 0)
	{
		gradientX[xOrigin + yOrigin * gradientStride] = 0.0f;
		return;
	}

	// We start with the x component.
	bool leftValid = ((xOrigin != 1) && (wallField[(xOrigin - 1) + yOrigin * wallStride]) == 0);
	bool rightValid = ((xOrigin != gGridSizeExternal - 2) && (wallField[(xOrigin + 1) + yOrigin * wallStride]) == 0);
	float result = 0.0f;

	if (rightValid && leftValid)
	{
		result = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan - 1][yScan]) / (2.0f * gCellSize);
	}
	else if (rightValid)
	{
		result = result = (inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	else if (leftValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / (gCellSize);
	}

	gradientX[xOrigin + yOrigin * gradientStride] = result;

	
}



void GradientModule::ComputeXDerivative(float* inputField, size_t inputStride, unsigned int* wallField,
	size_t wallStride)
{
	CompueXDerivativeCuda CUDA_DECORATOR_LOGIC (inputField, inputStride, wallField, wallStride, m_gradientResultX, m_gradientStride);
}


__global__ void CompueYDerivativeCuda(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride, float* gradientY,
	size_t gradientStride)
{
	__shared__ float inputBuffer[gBlockSize + 2][gBlockSize + 2];


	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;


	inputBuffer[xScan][yScan] = inputField[xOrigin + yOrigin * inputStride];


	if (threadIdx.x == 0)
		inputBuffer[0][yScan] = inputField[(xOrigin - 1) + yOrigin * inputStride];
	if (threadIdx.x == 31)
		inputBuffer[xScan + 1][yScan] = inputField[(xOrigin + 1) + yOrigin * inputStride];
	if (threadIdx.y == 0)
		inputBuffer[xScan][0] = inputField[xOrigin + (yOrigin - 1)* inputStride];
	if (threadIdx.y == 31)
		inputBuffer[xScan][yScan + 1] = inputField[xOrigin + (yOrigin + 1) * inputStride];



	__syncthreads();


	if (wallField[xOrigin + yOrigin * wallStride] != 0)
	{
		gradientY[xOrigin + yOrigin * gradientStride] = 0.0f;
		return;
	}
	
	// Now the same with the y component.
	bool topValid = ((yOrigin != 1) && (wallField[xOrigin + (yOrigin - 1) * wallStride]) == 0);
	bool bottomValid = ((yOrigin != gGridSizeExternal - 2) && (wallField[xOrigin + (yOrigin + 1) * wallStride]) == 0);
	float result = 0.0f;

	if (topValid && bottomValid)
	{
		result = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan - 1]) / (2.0f * gCellSize);
	}
	else if (bottomValid)
	{
		result = (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize);
	}
	else if (topValid)
	{
		result = (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize);
	}

	gradientY[xOrigin + yOrigin * gradientStride] = result;
}


void GradientModule::ComputeYDerivative(float* inputField, size_t inputStride, unsigned int* wallField,
	size_t wallStride)
{
	CompueYDerivativeCuda CUDA_DECORATOR_LOGIC(inputField, inputStride, wallField, wallStride, m_gradientResultY, m_gradientStride);
}


void GradientModule::VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory)
{
	float* fieldToVisualize = component ? m_gradientResultY : m_gradientResultX;
	m_transferHelper.VisualizeScalarFieldWithNegative(fieldToVisualize, maxExepctedValue, m_gradientStride, textureMemory);
}
