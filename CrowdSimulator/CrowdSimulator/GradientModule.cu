#include "GradientModule.h"
#include "GlobalConstants.h"
#include "CudaHelper.h"
#include "DataBase.h"
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

	assert(m_gradientResultX.m_stride == m_gradientResultY.m_stride);
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

void GradientModule::ComputeGradient(FloatArray inputField, DataBase* dataBase)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);
	ComputeGradientCuda CUDA_DECORATOR_LOGIC (inputField.m_array, inputField.m_stride, dataBase->GetWallData().m_array, dataBase->GetWallData().m_stride, 
		m_gradientResultX.m_array, m_gradientResultY.m_array, m_gradientResultX.m_stride);
}


__global__ void ComputeGradientCudaXDivergence(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride, float* gradientX, 
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
		result = fminf((inputBuffer[xScan + 1][yScan] - 0.0f) / (2.0f *gCellSize) ,(inputBuffer[xScan + 1][yScan] - inputBuffer[xScan][yScan]) / (gCellSize));
	}
	else if (leftValid)
	{
		result = fminf (( 0.0f - inputBuffer[xScan - 1][yScan]) / (2.0f * gCellSize), (inputBuffer[xScan][yScan] - inputBuffer[xScan - 1][yScan]) / gCellSize);
	}

	gradientX[xOrigin + yOrigin * gradientStride] = result;


	
}




void GradientModule::ComputeGradientXForDivergence(FloatArray inputField, DataBase* dataBase)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);
	ComputeGradientCudaXDivergence CUDA_DECORATOR_LOGIC(inputField.m_array, inputField.m_stride, dataBase->GetWallData().m_array, dataBase->GetWallData().m_stride,
		m_gradientResultX.m_array,  m_gradientResultX.m_stride);
}


__global__ void ComputeGradientCudaYDivergence(float* inputField, size_t inputStride, unsigned int* wallField, size_t wallStride, float* gradientY,
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
		result = fminf((inputBuffer[xScan][yScan + 1] - 0.0f) / (2.0f * gCellSize), (inputBuffer[xScan][yScan + 1] - inputBuffer[xScan][yScan]) / (gCellSize));
	}
	else if (topValid)
	{
		result = fminf((0.0f - inputBuffer[xScan][yScan - 1]) / (2.0f * gCellSize), (inputBuffer[xScan][yScan] - inputBuffer[xScan][yScan - 1]) / (gCellSize));
	}

	gradientY[xOrigin + yOrigin * gradientStride] = result;

}


void GradientModule::ComputeGradientYForDivergence(FloatArray inputField, DataBase* dataBase)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);
	ComputeGradientCudaYDivergence CUDA_DECORATOR_LOGIC(inputField.m_array, inputField.m_stride, dataBase->GetWallData().m_array, dataBase->GetWallData().m_stride,
		m_gradientResultY.m_array, m_gradientResultX.m_stride);
}


void GradientModule::VisualizeField(int component, float maxExepctedValue, uchar4* textureMemory)
{
	assert(m_gradientResultX.m_array);
	assert(m_gradientResultY.m_array);

	FloatArray fieldToVisualize = component ? m_gradientResultY : m_gradientResultX;
	VisualizationHelper::VisualizeScalarFieldWithNegative(fieldToVisualize, maxExepctedValue,  textureMemory);
}
