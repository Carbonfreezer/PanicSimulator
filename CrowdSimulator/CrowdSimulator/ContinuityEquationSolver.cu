#include "ContinuityEquationSolver.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>
#include "TransferHelper.h"
#include "DataBase.h"




void ContinuityEquationSolver::PrepareSolver()
{
	m_gradientIconal.PreprareModule();
	m_resultBuffer = TransferHelper::ReserveFloatMemory();
}


__global__ void IntegrateCuda(float timePassed,  size_t strides, float* density,  
								float* derivativeX, float* derivativeY,
							float* result)
{
	__shared__ float xDiv[gBlockSize + 2][gBlockSize + 2];
	__shared__ float yDiv[gBlockSize + 2][gBlockSize + 2];

	

	// Prefill the data.

	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	float factor;
	
	float xDivPure;
	float yDivPure;
	float localDensity;

	float xGrad, yGrad;

	xGrad = derivativeX[xOrigin + yOrigin * strides];
	yGrad = derivativeY[xOrigin + yOrigin * strides];


	// If both are zero we get an error here.
	factor = 1.0f / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);
	localDensity = density[xOrigin + yOrigin * strides];


	// Deal with the inf * zero situation.
	if (factor == 0.0f)
	{
		xDivPure = 0.0f;
		yDivPure = 0.0f;
	}
	else
	{
		xDivPure = xGrad * factor;
		yDivPure = yGrad * factor;
	}
	
	xDiv[xScan][yScan] = xDivPure * localDensity;
	yDiv[xScan][yScan] = yDivPure * localDensity;

	// Copy over the border lines, we do not need the corner elements.
	// We also do not need the density on the border lines.
	if (threadIdx.x == 0)
	{
		xGrad = derivativeX[(xOrigin - 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin - 1) + yOrigin * strides];
		
		factor  = density[(xOrigin - 1) + yOrigin * strides] /(xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan - 1][yScan] = 0.0f;
			yDiv[xScan - 1][yScan] = 0.0f;
		}
		else
		{
			xDiv[xScan - 1][yScan] = xGrad * factor;
			yDiv[xScan - 1][yScan] = yGrad * factor;
		}
		
	}
		
	if (threadIdx.x == 31)
	{
		xGrad = derivativeX[(xOrigin + 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin + 1) + yOrigin * strides];
		
		factor = density[(xOrigin + 1) + yOrigin * strides] / (xGrad * xGrad + yGrad * yGrad  + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan + 1][yScan] = 0.0f;
			yDiv[xScan + 1][yScan] = 0.0f;

		}
		else
		{
			xDiv[xScan + 1][yScan] = xGrad * factor;
			yDiv[xScan + 1][yScan] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 0)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin - 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin - 1) * strides];

		factor = density[(xOrigin)+(yOrigin - 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan][yScan - 1] = 0.0f;
			yDiv[xScan][yScan - 1] = 0.0f;

		}
		else
		{
			xDiv[xScan][yScan - 1] = xGrad * factor;
			yDiv[xScan][yScan - 1] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 31)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin + 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin + 1) * strides];

		factor = density[(xOrigin)+(yOrigin + 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);

		if (factor == 0.0f)
		{
			xDiv[xScan][yScan + 1] = 0.0f;
			yDiv[xScan][yScan + 1] = 0.0f;
		}
		else
		{
			xDiv[xScan][yScan + 1] = xGrad * factor;
			yDiv[xScan][yScan + 1] = yGrad * factor;
		}
	
	}
		

	__syncthreads();

	
	float xDerivative;
	if (xDivPure >= 0.0f)
		xDerivative = (xDiv[xScan + 1][yScan] - xDiv[xScan][yScan]) / (gCellSize);
	else
		xDerivative = (xDiv[xScan][yScan] - xDiv[xScan - 1][yScan]) / (gCellSize);

	
	float yDerivative;
	if (yDivPure >= 0.0f)
		yDerivative = (yDiv[xScan][yScan + 1] - yDiv[xScan][yScan]) / (gCellSize);
	else
		yDerivative = (yDiv[xScan][yScan] - yDiv[xScan ][yScan - 1]) / (gCellSize);


	float finalValue  = localDensity + timePassed * (xDerivative + yDerivative);
	finalValue = fmaxf(0.0f, finalValue);
	finalValue = fminf(gMaximumDensity, finalValue);

	result[xOrigin + yOrigin * strides] = finalValue;
	
	
	
}


void ContinuityEquationSolver::IntegrateEquation(float timePassed, FloatArray density,  FloatArray timeToDestination, DataBase* dataBase)
{

	UnsignedArray wallData = dataBase->GetWallData();
	// First we need the gradient of the iconal equation.
	m_gradientIconal.ComputeGradient(timeToDestination, wallData);

	FloatArray gradX = m_gradientIconal.GetXComponent();
	FloatArray gradY = m_gradientIconal.GetYComponent();
	assert(gradX.m_stride == gradY.m_stride);
	assert(gradX.m_stride == density.m_stride);
	assert(gradX.m_stride == m_resultBuffer.m_stride);
	assert(gradX.m_stride == wallData.m_stride);

	
	IntegrateCuda CUDA_DECORATOR_LOGIC (timePassed, gradX.m_stride, density.m_array,  gradX.m_array, gradY.m_array, m_resultBuffer.m_array);

	// TODO: Make double buffer later on.

	TransferHelper::CopyDataFromTo(m_resultBuffer, density);
		
}

