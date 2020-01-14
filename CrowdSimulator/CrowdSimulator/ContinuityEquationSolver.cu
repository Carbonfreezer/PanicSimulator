#include "ContinuityEquationSolver.h"
#include <cassert>
#include "CudaHelper.h"
#include <device_launch_parameters.h>
#include <math.h>
#include "TransferHelper.h"
#include "DataBase.h"




void ContinuityEquationSolver::PrepareSolver()
{
	m_timeToVelocity.PreprareModule();
	m_currentDensityField[0] = TransferHelper::ReserveFloatMemory();
	m_currentDensityField[1] = TransferHelper::ReserveFloatMemory();
	m_currentFieldValid = 0;
}

void ContinuityEquationSolver::FreeResources()
{
	m_currentDensityField[0].FreeArray();
	m_currentDensityField[1].FreeArray();
	m_timeToVelocity.FreeResources();
}


__global__ void IntegrateEquationCuda(float deltaTime,  size_t strides, float* density,  
                                      float* velocityX, float* velocityY, unsigned int* extremPoint,
                                      unsigned int* wallData,
                                      float* result)
{
	__shared__ float xMassTransport[gBlockSize + 2][gBlockSize + 2];
	__shared__ float yMassTransport[gBlockSize + 2][gBlockSize + 2];
	__shared__ unsigned int wallBuffer[gBlockSize + 2][gBlockSize + 2];

	// We keep tack of the pixel  we are responsible for.
	int xOrigin = threadIdx.x + gBlockSize * blockIdx.x + 1;
	int yOrigin = threadIdx.y + gBlockSize * blockIdx.y + 1;

	int xScan = threadIdx.x + 1;
	int yScan = threadIdx.y + 1;



	wallBuffer[xScan][yScan] = wallData[xOrigin + yOrigin * strides];
	float localDensity = density[xOrigin + yOrigin * strides];

	float xVelocity = velocityX[xOrigin + yOrigin * strides];
	float yVelocity = velocityY[xOrigin + yOrigin * strides];
	xMassTransport[xScan][yScan] = xVelocity * localDensity;
	yMassTransport[xScan][yScan] = yVelocity * localDensity;

	// Copy over the border lines, we do not need the corner elements.
	// We also do not need the density on the border lines.
	if (threadIdx.x == 0)
	{
		localDensity = density[xOrigin - 1 + yOrigin * strides];
		xMassTransport[xScan - 1][yScan] = velocityX[xOrigin - 1 + yOrigin * strides] * localDensity;
		yMassTransport[xScan - 1][yScan] = velocityY[xOrigin - 1 + yOrigin * strides] * localDensity;
		wallBuffer[xScan - 1][yScan] = wallData[(xOrigin - 1) + yOrigin * strides];
	}
		
	if (threadIdx.x == 31)
	{
		localDensity = density[xOrigin + 1 + yOrigin * strides];
		xMassTransport[xScan + 1][yScan] = velocityX[xOrigin + 1 + yOrigin * strides] * localDensity;
		yMassTransport[xScan + 1][yScan] = velocityY[xOrigin + 1 + yOrigin * strides] * localDensity;
		wallBuffer[xScan + 1][yScan] = wallData[(xOrigin + 1) + yOrigin * strides];
	}
		
	if (threadIdx.y == 0)
	{
		localDensity = density[xOrigin + (yOrigin - 1) * strides];
		xMassTransport[xScan][yScan - 1] = velocityX[xOrigin + (yOrigin - 1) * strides] * localDensity;
		yMassTransport[xScan][yScan - 1] = velocityY[xOrigin + (yOrigin - 1) * strides] * localDensity;
		wallBuffer[xScan][yScan - 1] = wallData[xOrigin + (yOrigin - 1) * strides];
	}
		
	if (threadIdx.y == 31)
	{
		localDensity = density[xOrigin + (yOrigin + 1) * strides];
		xMassTransport[xScan][yScan + 1] = velocityX[xOrigin + (yOrigin + 1) * strides] * localDensity;
		yMassTransport[xScan][yScan + 1] = velocityY[xOrigin + (yOrigin + 1) * strides] * localDensity;
		wallBuffer[xScan][yScan + 1] = wallData[xOrigin + (yOrigin + 1) * strides];
	}
	__syncthreads();


	if (wallBuffer[xScan][yScan])
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
		if (xVelocity < 0.0f)
		{
			// If we run into a wall we only accumulate but do not loose.
			if (wallBuffer[xScan - 1][yScan])
				xDerivative = (xMassTransport[xScan + 1][yScan]) / (gCellSize);
			else
				xDerivative = (xMassTransport[xScan + 1][yScan] - xMassTransport[xScan][yScan]) / (gCellSize);
		}
			
		else
		{
			if (wallBuffer[xScan + 1][yScan])
				xDerivative = ( - xMassTransport[xScan - 1][yScan]) / (gCellSize);
			else
				xDerivative = (xMassTransport[xScan][yScan] - xMassTransport[xScan - 1][yScan]) / (gCellSize);
		}
			
	}
	
	float yDerivative;
	if (extremPointInfo / 2 == 1)
	{
		yDerivative = (yMassTransport[xScan][yScan + 1] - yMassTransport[xScan][yScan - 1]) / (2.0f *  gCellSize);
	}
	else
	{
		if (yVelocity < 0.0f)
		{
			if (wallBuffer[xScan][yScan - 1])
				yDerivative = (yMassTransport[xScan][yScan + 1]) / (gCellSize);
			else
				yDerivative = (yMassTransport[xScan][yScan + 1] - yMassTransport[xScan][yScan]) / (gCellSize);
		}
		else
		{
			if (wallBuffer[xScan][yScan + 1])
				yDerivative = (- yMassTransport[xScan][yScan - 1]) / (gCellSize);
			else
				yDerivative = (yMassTransport[xScan][yScan] - yMassTransport[xScan][yScan - 1]) / (gCellSize);
		}
			
	}


	float buffer = density[xOrigin + yOrigin * strides] - deltaTime * (xDerivative + yDerivative);
	buffer = fmaxf(0.0f, buffer);

	// buffer = fminf(buffer, gMaximumDensity);
	result[xOrigin + yOrigin * strides] = buffer ;
}



void ContinuityEquationSolver::IntegrateEquation(float timePassed,   FloatArray timeToDestination, DataBase* dataBase)
{

	UnsignedArray wallData = dataBase->GetWallData();
	// First we need the gradient of the iconal equation.
	m_timeToVelocity.ComputeVelocity(timeToDestination, wallData, dataBase->GetTargetData());

	FloatArray velX = m_timeToVelocity.GetXComponent();
	FloatArray velY = m_timeToVelocity.GetYComponent();
	UnsignedArray extremPoint = m_timeToVelocity.GerExtremPointInformation();
	assert(velX.m_stride == velY.m_stride);
	assert(velX.m_stride == m_currentDensityField[0].m_stride);
	assert(velX.m_stride == wallData.m_stride);
	assert(velX.m_stride == extremPoint.m_stride);

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

		IntegrateEquationCuda CUDA_DECORATOR_LOGIC (localTimeStep, velX.m_stride,
		                                            m_currentDensityField[m_currentFieldValid].m_array, 
		                                            velX.m_array, velY.m_array,extremPoint.m_array, wallData.m_array,
		                                            m_currentDensityField[1 - m_currentFieldValid].m_array);

		m_currentFieldValid = 1 - m_currentFieldValid;
	}

		
}


