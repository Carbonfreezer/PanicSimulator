#include "ContinuityEquationSolver.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>
#include "TransferHelper.h"
#include "DataBase.h"




void ContinuityEquationSolver::PrepareSolver()
{
	m_gradienEikonal.PreprareModule();
	m_derivativeBuffer = TransferHelper::ReserveFloatMemory();
}


__global__ void IntegrateEquationCuda(float deltaTime,  size_t strides, float* density,  
								float* derivativeX, float* derivativeY, unsigned int* extremPoint,
								unsigned int* wallInformation,
							float* result)
{
	__shared__ float xMassTransport[gBlockSize + 2][gBlockSize + 2];
	__shared__ float yMassTransport[gBlockSize + 2][gBlockSize + 2];

	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;

	float factor;	
	float xVelocity;
	float yVelocity;
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
		xVelocity = 0.0f;
		yVelocity = 0.0f;
	}
	else
	{
		xVelocity = xGrad * factor;
		yVelocity = yGrad * factor;
	}
	
	xMassTransport[xScan][yScan] = xVelocity * localDensity;
	yMassTransport[xScan][yScan] = yVelocity * localDensity;

	// Copy over the border lines, we do not need the corner elements.
	// We also do not need the density on the border lines.
	if (threadIdx.x == 0)
	{
		xGrad = derivativeX[(xOrigin - 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin - 1) + yOrigin * strides];
		
		factor  = density[(xOrigin - 1) + yOrigin * strides] /(xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);
		if (factor == 0.0f)
		{
			xMassTransport[xScan - 1][yScan] = 0.0f;
			yMassTransport[xScan - 1][yScan] = 0.0f;
		}
		else
		{
			xMassTransport[xScan - 1][yScan] = xGrad * factor;
			yMassTransport[xScan - 1][yScan] = yGrad * factor;
		}
	}
		
	if (threadIdx.x == 31)
	{
		xGrad = derivativeX[(xOrigin + 1) + yOrigin * strides];
		yGrad = derivativeY[(xOrigin + 1) + yOrigin * strides];
		
		factor = density[(xOrigin + 1) + yOrigin * strides] / (xGrad * xGrad + yGrad * yGrad  + FLT_EPSILON);
		if (factor == 0.0f)
		{
			xMassTransport[xScan + 1][yScan] = 0.0f;
			yMassTransport[xScan + 1][yScan] = 0.0f;

		}
		else
		{
			xMassTransport[xScan + 1][yScan] = xGrad * factor;
			yMassTransport[xScan + 1][yScan] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 0)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin - 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin - 1) * strides];

		factor = density[(xOrigin)+(yOrigin - 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);
		if (factor == 0.0f)
		{
			xMassTransport[xScan][yScan - 1] = 0.0f;
			yMassTransport[xScan][yScan - 1] = 0.0f;

		}
		else
		{
			xMassTransport[xScan][yScan - 1] = xGrad * factor;
			yMassTransport[xScan][yScan - 1] = yGrad * factor;

		}
	}
		
	if (threadIdx.y == 31)
	{
		xGrad = derivativeX[(xOrigin)+(yOrigin + 1) * strides];
		yGrad = derivativeY[(xOrigin)+(yOrigin + 1) * strides];

		factor = density[(xOrigin)+(yOrigin + 1) * strides] / (xGrad * xGrad + yGrad * yGrad + FLT_EPSILON);
		if (factor == 0.0f)
		{
			xMassTransport[xScan][yScan + 1] = 0.0f;
			yMassTransport[xScan][yScan + 1] = 0.0f;
		}
		else
		{
			xMassTransport[xScan][yScan + 1] = xGrad * factor;
			yMassTransport[xScan][yScan + 1] = yGrad * factor;
		}
	
	}
	__syncthreads();

	// No divergence on wall.
	if (wallInformation[xOrigin + yOrigin * strides])
	{
		result[xOrigin + yOrigin * strides] = 0.0f;
		return;
	}

	unsigned int extremPointInfo = extremPoint[xOrigin + yOrigin * strides];

	float xDerivative;
	// In this case we have a confluence or defluencing point and take the central difference quotient.
	if (extremPointInfo % 2 == 1)
	{
		xDerivative = (xMassTransport[xScan + 1][yScan] - xMassTransport[xScan - 1][yScan]) / ( 2.0f * gCellSize);
	}
	else
	{
		if (xVelocity >= 0.0f)
			xDerivative = (xMassTransport[xScan + 1][yScan] - xMassTransport[xScan][yScan]) / (gCellSize);
		else
			xDerivative = (xMassTransport[xScan][yScan] - xMassTransport[xScan - 1][yScan]) / (gCellSize);
	}
	
	float yDerivative;
	if (extremPointInfo / 2 == 1)
	{
		yDerivative = (yMassTransport[xScan][yScan + 1] - yMassTransport[xScan][yScan - 1]) / (2.0f *  gCellSize);
	}
	else
	{
		if (yVelocity >= 0.0f)
			yDerivative = (yMassTransport[xScan][yScan + 1] - yMassTransport[xScan][yScan]) / (gCellSize);
		else
			yDerivative = (yMassTransport[xScan][yScan] - yMassTransport[xScan][yScan - 1]) / (gCellSize);
	}


	float buffer = density[xOrigin + yOrigin * strides] + deltaTime * (xDerivative + yDerivative);
	buffer = fmaxf(0.0f, buffer);
	buffer = fminf(buffer, gMaximumDensity);
	result[xOrigin + yOrigin * strides] = buffer ;
}



void ContinuityEquationSolver::IntegrateEquation(float timePassed, FloatArray density,  FloatArray timeToDestination, DataBase* dataBase)
{

	UnsignedArray wallData = dataBase->GetWallData();
	// First we need the gradient of the iconal equation.
	m_gradienEikonal.ComputeGradient(timeToDestination, wallData);

	FloatArray gradX = m_gradienEikonal.GetXComponent();
	FloatArray gradY = m_gradienEikonal.GetYComponent();
	UnsignedArray extremPoint = m_gradienEikonal.GerExtremPointInformation();
	assert(gradX.m_stride == gradY.m_stride);
	assert(gradX.m_stride == density.m_stride);
	assert(gradX.m_stride == m_derivativeBuffer.m_stride);
	assert(gradX.m_stride == wallData.m_stride);
	assert(gradX.m_stride == extremPoint.m_stride);

	bool isTerminated = false;
	float localTimeStep;
	while(!isTerminated)
	{
		if (timePassed > gMaximumStepsizeContinuitySolver)
		{
			localTimeStep = gMaximumStepsizeContinuitySolver;
			timePassed -= gMaximumStepsizeContinuitySolver;
		}
		else
		{
			isTerminated = true;
			localTimeStep = timePassed;
		}

		IntegrateEquationCuda CUDA_DECORATOR_LOGIC (localTimeStep, density.m_stride, density.m_array, gradX.m_array, gradY.m_array,extremPoint.m_array,wallData.m_array,  m_derivativeBuffer.m_array);
		TransferHelper::CopyDataFromTo(m_derivativeBuffer, density);
		
	}

		
}


